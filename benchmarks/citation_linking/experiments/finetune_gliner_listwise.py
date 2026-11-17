"""Train single-choice GLiNER linking with full candidate descriptions preserved."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

from benchmarks.citation_linking.experiments.gliner_data import shuffle_training_example


def read_split(
    directory: Path, split: str, *, allow_test: bool = False
) -> tuple[list[dict], list[dict]]:
    """Training reads train/validation only; ``allow_test`` is for the frozen OOF evaluator."""
    if split not in {"train", "validation"} | ({"test"} if allow_test else set()):
        raise ValueError("The training runner only reads train and validation")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete training-data export")
    loaded = []
    for filename in (f"{split}.jsonl", f"{split}.metadata.jsonl"):
        data = (directory / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["files_sha256"][filename]:
            raise ValueError(f"Export checksum mismatch: {filename}")
        loaded.append([json.loads(line) for line in data.splitlines() if line.strip()])
    examples, metadata = loaded
    if not examples or len(examples) != len(metadata):
        raise ValueError("Examples and metadata must be nonempty and aligned")
    for example, meta in zip(examples, metadata):
        cls = example["output"]["classifications"][0]
        if (
            meta["split"] != split
            or not meta["eligible"]
            or cls["true_label"] != [meta["target"]]
            or cls["labels"] != list(meta["candidate_ids"]) + ["none"]
            or set(cls["label_descriptions"]) != set(cls["labels"])
        ):
            raise ValueError(
                "Invalid split, label mapping, or missing candidate descriptions"
            )
    return examples, metadata


def smoke_subset(
    examples: list[dict], metadata: list[dict], size: int
) -> tuple[list, list]:
    """Keep short train examples from each available supervision source."""
    if size < 4:
        raise ValueError("Smoke tests require at least four examples")
    groups = {}
    for i in sorted(range(len(examples)), key=lambda i: len(json.dumps(examples[i]))):
        groups.setdefault(metadata[i]["reason"], []).append(i)
    if "gold_retrieved" not in groups or not any(
        reason in groups for reason in ("gold_not_retrieved", "annotated_no_record")
    ):
        raise ValueError("Smoke test needs both candidate and none targets")
    # Include a real choice among candidates, not only singleton classification.
    multi = next(
        (i for i in groups["gold_retrieved"] if len(metadata[i]["candidate_ids"]) > 1),
        None,
    )
    if multi is None:
        raise ValueError(
            "Smoke test requires a retrieved gold with competing candidates"
        )
    groups["gold_retrieved"].remove(multi)
    selected = [multi]
    while len(selected) < min(size, len(examples)):
        for group in groups.values():
            if group and len(selected) < size:
                selected.append(group.pop(0))
    return [examples[i] for i in selected], [metadata[i] for i in selected]


def prepare_example(model, example: dict, max_tokens: int):
    """Use deterministic schema preprocessing with supervised targets and no truncation."""
    batch = model.processor.collate_fn_inference(
        [(example["input"], copy.deepcopy(example["output"]))],
        max_len=None,
        architecture=model.architecture,
        build_targets=True,
        error_policy="raise",
    )
    cls = example["output"]["classifications"][0]
    expected = [int(label == cls["true_label"][0]) for label in cls["labels"]]
    if batch.structure_labels != [[expected]] or sum(expected) != 1:
        raise ValueError("Processor lost or reordered the single-label target")
    # GLiNER appends sentence punctuation to both training and inference text.
    if batch.original_texts[0] not in {example["input"], example["input"] + "."}:
        raise ValueError("Processor changed citation text")
    if batch.end_mappings[0][-1] != len(batch.original_texts[0]):
        raise ValueError("Processor truncated citation text")
    tokens = int(batch.attention_mask.sum())
    if tokens > max_tokens:
        raise ValueError(
            f"Input has {tokens} tokens, exceeding the explicit {max_tokens} budget; no truncation applied"
        )
    return batch


def logits_for_example(model, batch):
    """Use the same encoder, schema embeddings and classifier as Classifier.score."""
    import torch

    batch = batch.to(next(model.parameters()).device)
    encoded = model.encode_tokens(batch)
    _, schemas = model.processor.extract_embeddings_from_batch(
        encoded, batch.input_ids, batch
    )
    logits = model.classifier(torch.stack(schemas[0][0][1:])).squeeze(-1)
    if logits.ndim != 1 or len(logits) != len(batch.structure_labels[0][0]):
        raise ValueError("Classifier logits do not align with the candidate labels")
    return logits


def listwise_loss(
    logits,
    target: int,
    *,
    none_weight: float = 1.0,
    label_smoothing: float = 0.0,
    rank_weight: float = 0.0,
):
    """Cross entropy over candidates + ``none`` (last logit), with recall-protecting terms.

    ``none_weight`` scales examples whose target is ``none``. ``rank_weight`` adds a
    softmax over the candidates alone when a real candidate is correct, so the ranker
    keeps learning even when the none-vs-candidate decision dominates the full loss.
    """
    import torch
    import torch.nn.functional as F

    none_index = len(logits) - 1
    label = torch.tensor([target], device=logits.device)
    full = F.cross_entropy(logits[None], label, label_smoothing=label_smoothing)
    if target == none_index:
        return full * none_weight
    if rank_weight and none_index > 1:
        return full + rank_weight * F.cross_entropy(logits[:none_index][None], label)
    return full


def recall_selection(
    result: dict, tasks, metadata: list[dict], minimum_tp: int
) -> tuple[tuple, dict | None]:
    """Rank an epoch by max precision at TP >= minimum_tp, using only validation."""
    from benchmarks.citation_linking.experiments.gliner_eval import (
        select_policy,
        sweep_bias,
    )

    curve = sweep_bias(tasks, result["predictions"], metadata)
    selected = select_policy(curve, minimum_tp)
    if selected is None:
        return (0, int(curve.TP.max())), None
    return (1, selected["precision"], selected["recall"]), selected


def evaluate(
    model, examples: list[dict], metadata: list[dict], max_tokens: int
) -> dict:
    import torch

    model.eval()
    rows, losses = [], []
    with torch.no_grad():
        for example, meta in zip(examples, metadata):
            batch = prepare_example(model, example, max_tokens)
            logits = logits_for_example(model, batch)
            cls = example["output"]["classifications"][0]
            target = cls["labels"].index(meta["target"])
            loss = torch.nn.functional.cross_entropy(
                logits[None], torch.tensor([target], device=logits.device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite evaluation loss")
            losses.append(float(loss))
            probabilities = logits.softmax(-1).cpu().tolist()
            choice = cls["labels"][int(logits.argmax())]
            rows.append(
                {
                    "ref_id": meta["ref_id"],
                    "index": meta["index"],
                    "split": meta["split"],
                    "target": meta["target"],
                    "choice": choice,
                    "predicted_id": meta["candidate_ids"].get(choice),
                    "gold_id": meta["gold_id"],
                    "reason": meta["reason"],
                    "reference_seen_in_train": meta["reference_seen_in_train"],
                    "probabilities": dict(zip(cls["labels"], probabilities)),
                }
            )
    summary = {"loss": sum(losses) / len(losses), "examples": len(rows)}
    for name, subset in (
        ("all", rows),
        ("candidate", [r for r in rows if r["target"] != "none"]),
        ("none", [r for r in rows if r["target"] == "none"]),
        ("seen_reference", [r for r in rows if r["reference_seen_in_train"]]),
        ("unseen_reference", [r for r in rows if not r["reference_seen_in_train"]]),
    ):
        summary[name] = {
            "count": len(subset),
            "correct": sum(r["choice"] == r["target"] for r in subset),
        }
    return {"summary": summary, "predictions": rows}


def save_dev_metrics(
    result: dict, data_dir: Path, output_dir: Path, prefix: str
) -> None:
    """Include automatic empty-pool rejection in dev link metrics and threshold curves."""
    import pandas as pd
    from benchmarks.citation_linking.experiments.linking_eval import (
        KEY,
        score_predictions,
        summarize_scores,
    )

    audit = pd.read_json(data_dir / "audit.jsonl", lines=True, convert_dates=False)
    tasks = audit.loc[
        audit.split.eq("validation")
        & (audit.eligible | audit.reason.eq("empty_candidates"))
    ].copy()
    predictions = pd.DataFrame(result["predictions"])
    if not predictions.split.eq("validation").all():
        raise ValueError("Threshold curves require validation predictions only")
    expected = set(map(tuple, tasks.loc[tasks.eligible, KEY].values))
    if (
        set(map(tuple, predictions[KEY].values)) != expected
        or predictions.duplicated(KEY).any()
    ):
        raise ValueError(
            "Dev predictions must cover exactly the eligible validation tasks"
        )
    predictions["selected_score"] = predictions.apply(
        lambda row: row.probabilities[row.choice] if row.choice != "none" else 0.0,
        axis=1,
    )
    merged = tasks.merge(
        predictions[KEY + ["predicted_id", "selected_score"]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    scored = score_predictions(tasks, merged)
    tables = []
    for name, group in (
        ("all", scored),
        ("seen_reference", scored.loc[scored.reference_seen_in_train]),
        ("unseen_reference", scored.loc[~scored.reference_seen_in_train]),
    ):
        if len(group):
            table = summarize_scores(group, by=()).assign(scope=name)
            table["false_rejections_retrieved_gold"] = int(
                (group.reason.eq("gold_retrieved") & group.predicted_id.isna()).sum()
            )
            tables.append(table)
    tables.append(summarize_scores(scored, by=("index",)).assign(scope="per_index"))
    pd.concat(tables, ignore_index=True).to_csv(
        output_dir / f"{prefix}_dev_links.csv", index=False
    )
    thresholds = sorted(
        {
            0.0,
            1.0000001,
            *predictions.loc[
                predictions.predicted_id.notna(), "selected_score"
            ].tolist(),
        }
    )
    curve = []
    for threshold in thresholds:
        gated = merged.copy()
        gated.loc[gated.selected_score.lt(threshold), "predicted_id"] = None
        summary = (
            summarize_scores(score_predictions(tasks, gated), by=()).iloc[0].to_dict()
        )
        curve.append({"threshold": threshold, **summary})
    pd.DataFrame(curve).to_csv(
        output_dir / f"{prefix}_dev_precision_recall.csv", index=False
    )


def train(args: argparse.Namespace) -> dict:
    import torch
    from gliner2 import AutoExtractor
    from gliner2.classification import Classifier, ClassificationSchema
    from huggingface_hub import snapshot_download

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples, metadata = read_split(args.data_dir, "train")
    if args.smoke_size:
        examples, metadata = smoke_subset(examples, metadata, args.smoke_size)
        eval_examples, eval_metadata = examples, metadata
    elif args.train_only:
        # No held-out data: a small train slice only checks checkpoint reload.
        eval_examples, eval_metadata = examples[:40], metadata[:40]
    else:
        eval_examples, eval_metadata = read_split(args.data_dir, "validation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "mode": (
            "smoke_overfit"
            if args.smoke_size
            else "train_only_fixed_epochs" if args.train_only else "full_train"
        ),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("gliner2", "torch", "transformers")
        },
        "training_data_manifest_sha256": hashlib.sha256(
            (args.data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "objective": "single-label cross entropy over all candidates and none",
        "selection_scope": (
            "train overfit diagnostic"
            if args.smoke_size
            else (
                "none; final epoch of a fixed budget"
                if args.train_only
                else "validation loss"
            )
        ),
        "history": [],
    }

    def save_manifest():
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=lambda o: o.item()) + "\n"
        )

    save_manifest()
    try:
        model_path = Path(args.model)
        if not model_path.is_dir():
            model_path = Path(snapshot_download(args.model, revision=args.revision))
        manifest["model_snapshot"] = str(model_path)
        manifest["model_weights_sha256"] = hashlib.sha256(
            (model_path / "model.safetensors").read_bytes()
        ).hexdigest()
        model = AutoExtractor.from_pretrained(str(model_path)).to(args.device)
        batches = [
            prepare_example(model, example, args.max_tokens) for example in examples
        ]
        eval_batches = [
            prepare_example(model, example, args.max_tokens)
            for example in eval_examples
        ]
        manifest["tokens"] = {
            "train_max": max(int(b.attention_mask.sum()) for b in batches),
            "evaluation_max": max(int(b.attention_mask.sum()) for b in eval_batches),
            "truncated": 0,
        }
        # Check the differentiable scoring path against the pretrained baseline API.
        first = examples[0]
        labels = first["output"]["classifications"][0]["label_descriptions"]
        native = (
            Classifier(model)
            .eval()
            .classify(first["input"], ClassificationSchema().single("link", labels))
        )
        with torch.no_grad():
            probabilities = logits_for_example(model, batches[0]).softmax(-1).cpu()
        expected = torch.tensor(
            [dict(native.probabilities("link"))[label] for label in labels]
        )
        if not torch.allclose(probabilities, expected, atol=1e-5, rtol=1e-5):
            raise ValueError(
                "Training logits differ from baseline Classifier probabilities"
            )
        before = evaluate(model, eval_examples, eval_metadata, args.max_tokens)
        (args.output_dir / "before.json").write_text(json.dumps(before, indent=2))
        manifest["before"] = before["summary"]
        if not args.smoke_size and not args.train_only:
            save_dev_metrics(before, args.data_dir, args.output_dir, "pretrained")
        print(
            json.dumps({"before": before["summary"], "tokens": manifest["tokens"]}),
            flush=True,
        )
        encoder, head = [], []
        for name, parameter in model.named_parameters():
            if args.freeze_encoder and name.startswith("encoder."):
                parameter.requires_grad_(False)
            if parameter.requires_grad:
                (encoder if name.startswith("encoder.") else head).append(parameter)
        groups = [{"params": head, "lr": args.head_lr}]
        if encoder:
            groups.append({"params": encoder, "lr": args.encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        initial_weight = next(model.classifier.parameters()).detach().cpu().clone()
        best_loss = float("inf")
        best_key: tuple | None = None
        stale_epochs = 0
        by_recall = (
            args.select_by == "recall_policy"
            and not args.smoke_size
            and not args.train_only
        )
        if by_recall:
            import pandas as pd

            audit = pd.read_json(
                args.data_dir / "audit.jsonl", lines=True, convert_dates=False
            )
            dev_tasks = audit.loc[
                audit.split.eq("validation")
                & (audit.eligible | audit.reason.eq("empty_candidates"))
            ].copy()
            from benchmarks.citation_linking.experiments.gliner_eval import score_policy

            baseline, _ = score_policy(
                dev_tasks, before["predictions"], eval_metadata, 0.0
            )
            minimum_tp = int(baseline["TP"])
            manifest["selection_minimum_tp"] = minimum_tp
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(examples)))
            random.Random(args.seed + epoch).shuffle(order)
            losses = []
            for i in order:
                example, meta = shuffle_training_example(
                    examples[i], metadata[i], epoch=epoch, seed=args.seed
                )
                batch = prepare_example(model, example, args.max_tokens)
                optimizer.zero_grad(set_to_none=True)
                logits = logits_for_example(model, batch)
                labels = example["output"]["classifications"][0]["labels"]
                target = torch.tensor(
                    [labels.index(meta["target"])], device=logits.device
                )
                loss = listwise_loss(
                    logits,
                    int(target),
                    none_weight=args.none_weight,
                    label_smoothing=args.label_smoothing,
                    rank_weight=args.rank_weight,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                if len(losses) % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "completed": len(losses),
                                "total": len(order),
                                "train_loss": sum(losses) / len(losses),
                            }
                        ),
                        flush=True,
                    )
            result = evaluate(model, eval_examples, eval_metadata, args.max_tokens)
            summary = {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / len(losses),
                **result["summary"],
            }
            manifest["history"].append(summary)
            improved_loss = result["summary"]["loss"] < best_loss
            best_loss = min(best_loss, result["summary"]["loss"])
            if by_recall:
                key, policy = recall_selection(
                    result, dev_tasks, eval_metadata, minimum_tp
                )
                summary["selection_key"] = list(key)
                summary["selected_policy"] = policy
                improved = best_key is None or key > best_key
            elif args.train_only:
                key, policy = None, None
                improved = epoch + 1 == args.epochs
            else:
                key, policy = None, None
                improved = improved_loss
            if improved:
                best_key = key
                manifest["best_policy"] = policy
                stale_epochs = 0
                manifest["best_epoch"] = epoch + 1
                model.save_pretrained(str(args.output_dir / "best"))
                (args.output_dir / "best_predictions.json").write_text(
                    json.dumps(result, indent=2)
                )
            else:
                stale_epochs += 1
            print(json.dumps(summary, default=lambda o: o.item()), flush=True)
            save_manifest()
            if (
                args.smoke_size
                and result["summary"]["all"]["correct"] == len(examples)
                and best_loss < before["summary"]["loss"]
            ):
                break
            if (
                not args.smoke_size
                and not args.train_only
                and stale_epochs >= args.patience
            ):
                break
        manifest["classifier_weights_changed"] = not torch.equal(
            initial_weight, next(model.classifier.parameters()).detach().cpu()
        )
        del optimizer, model
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        model = AutoExtractor.from_pretrained(str(args.output_dir / "best")).to(
            args.device
        )
        reloaded = evaluate(model, eval_examples, eval_metadata, args.max_tokens)
        best = json.loads((args.output_dir / "best_predictions.json").read_text())
        if [r["choice"] for r in reloaded["predictions"]] != [
            r["choice"] for r in best["predictions"]
        ]:
            raise ValueError(
                "Reloaded checkpoint decisions differ from saved decisions"
            )
        if abs(reloaded["summary"]["loss"] - best["summary"]["loss"]) > 1e-5:
            raise ValueError("Reloaded checkpoint loss differs from saved loss")
        manifest["after"] = reloaded["summary"]
        manifest["checkpoint_reload_verified"] = True
        if not args.smoke_size and not args.train_only:
            save_dev_metrics(reloaded, args.data_dir, args.output_dir, "finetuned")
        if not manifest["classifier_weights_changed"]:
            raise ValueError("Training did not update classifier weights")
        if args.smoke_size and (
            reloaded["summary"]["all"]["correct"] != len(examples)
            or reloaded["summary"]["loss"] >= before["summary"]["loss"]
        ):
            raise ValueError(
                "Smoke test has not yet learned all selected training examples"
            )
        manifest["status"] = "complete"
        save_manifest()
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        save_manifest()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="fastino/GLiNER2.5-multi-Decide")
    parser.add_argument(
        "--revision", default="e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--smoke-size", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--select-by",
        choices=("recall_policy", "loss"),
        default="recall_policy",
        help="Epoch criterion: max validation precision at pretrained TP (recall_policy) or loss",
    )
    parser.add_argument("--none-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train all epochs on train only and keep the last; no validation selection",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.patience < 1
        or args.max_tokens < 1
        or args.smoke_size < 0
        or (args.encoder_lr <= 0 and not args.freeze_encoder)
        or args.head_lr <= 0
        or args.none_weight <= 0
        or args.rank_weight < 0
        or not 0 <= args.label_smoothing < 1
    ):
        parser.error(
            "Epochs, patience, token budget, and learning rates must be positive"
        )
    result = train(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_epoch": result["best_epoch"],
                "after": result["after"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
