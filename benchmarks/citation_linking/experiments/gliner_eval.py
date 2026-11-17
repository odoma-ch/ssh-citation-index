"""GLiNER none-logit bias policy and held-out evaluation against the pretrained model.

``predict`` scores a run's held-out test split with the pretrained model and the run's
best checkpoint; ``summarize`` rescales the saved probabilities with a frozen bias and
compares the two models task by task (exact McNemar). Test labels never select anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from benchmarks.citation_linking.experiments.finetune_gliner_listwise import (
    evaluate,
    read_split,
)
from benchmarks.citation_linking.experiments.linking_eval import (
    KEY,
    score_predictions,
    summarize_scores,
)
from benchmarks.citation_linking.experiments.rerankers import bias_none_probabilities


def score_policy(
    tasks: pd.DataFrame,
    rows: list[dict],
    metadata: list[dict],
    bias: float,
    split: str = "validation",
):
    """Score all approved tasks of one split, including automatic empty-list rejection.

    Bias selection must use ``validation``; ``test`` scores a frozen bias only.
    """
    if not tasks.split.eq(split).all():
        raise ValueError(f"Policy scoring only accepts {split} tasks")
    expected = set(map(tuple, tasks.loc[tasks.eligible, KEY].values))
    predictions = []
    if len(rows) != len(metadata):
        raise ValueError("Predictions and metadata must be aligned")
    for row, meta in zip(rows, metadata):
        if (
            row["split"] != split
            or meta["split"] != split
            or any(row[key] != meta[key] for key in KEY)
            or row["target"] != meta["target"]
            or set(row["probabilities"]) != set(meta["candidate_ids"]) | {"none"}
        ):
            raise ValueError("Invalid prediction or candidate-label mapping")
        probabilities = bias_none_probabilities(row["probabilities"], bias)
        choice = max(probabilities, key=probabilities.get) if bias else row["choice"]
        if choice not in probabilities or probabilities[choice] < max(
            probabilities.values()
        ):
            raise ValueError("Choice must have maximum adjusted probability")
        predictions.append(
            {
                **{key: row[key] for key in KEY},
                "predicted_id": meta["candidate_ids"].get(choice),
            }
        )
    predicted = pd.DataFrame(predictions)
    if (
        predicted.duplicated(KEY).any()
        or set(map(tuple, predicted[KEY].values)) != expected
    ):
        raise ValueError(f"Predictions must cover exactly eligible {split} tasks")
    scored = score_predictions(tasks, tasks[KEY].merge(predicted, on=KEY, how="left"))
    summary = summarize_scores(scored, by=()).iloc[0].to_dict()
    summary["none_logit_bias"] = bias
    summary["false_rejections_retrieved"] = int(
        (scored.reason.eq("gold_retrieved") & scored.predicted_id.isna()).sum()
    )
    return summary, scored


def sweep_bias(
    tasks: pd.DataFrame,
    rows: list[dict],
    metadata: list[dict],
    split: str = "validation",
) -> pd.DataFrame:
    margins = []
    for row in rows:
        probabilities = row["probabilities"]
        best = max(p for label, p in probabilities.items() if label != "none")
        none = probabilities["none"]
        if best > 0 and none > 0:
            margins.append(math.log(best) - math.log(none))
    boundaries = sorted(set(margins))
    biases = {0.0}
    if boundaries:
        # Midpoints keep operating decisions away from floating-point ties.
        biases.update((boundaries[0] - 1, boundaries[-1] + 1))
        biases.update((a + b) / 2 for a, b in zip(boundaries, boundaries[1:]))
    return pd.DataFrame(
        [score_policy(tasks, rows, metadata, bias, split)[0] for bias in sorted(biases)]
    )


def select_policy(curve: pd.DataFrame, minimum_tp: int) -> dict | None:
    """Precision first, then recall, then the smallest change to the native decision."""
    feasible = curve.loc[curve.TP.ge(minimum_tp) & curve.precision.notna()].copy()
    if feasible.empty:
        return None
    feasible["absolute_bias"] = feasible.none_logit_bias.abs()
    return (
        feasible.sort_values(
            ["precision", "recall", "absolute_bias", "none_logit_bias"],
            ascending=[False, False, True, True],
        )
        .drop(columns="absolute_bias")
        .iloc[0]
        .to_dict()
    )


MODELS = ("pretrained", "finetuned")


def split_tasks(data_dir: Path, split: str) -> pd.DataFrame:
    audit = pd.read_json(data_dir / "audit.jsonl", lines=True, convert_dates=False)
    return audit.loc[
        audit.split.eq(split) & (audit.eligible | audit.reason.eq("empty_candidates"))
    ].copy()


def held_out_tasks(data_dir: Path) -> pd.DataFrame:
    return split_tasks(data_dir, "test")


def run_manifest(run_dir: Path) -> tuple[dict, Path]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Training did not complete: {run_dir}")
    return manifest, Path(manifest["config"]["data_dir"])


def predict(data_dir: Path, run_dir: Path, device: str, max_tokens: int = 8192) -> dict:
    from gliner2 import AutoExtractor

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Training did not complete")
    policy = manifest.get("best_policy")
    bias = float(policy["none_logit_bias"]) if policy else 0.0
    examples, metadata = read_split(data_dir, "test", allow_test=True)
    tasks = held_out_tasks(data_dir)
    paths = {"pretrained": manifest["model_snapshot"], "finetuned": run_dir / "best"}
    report = {"frozen_bias": bias, "has_feasible_policy": policy is not None}
    for name, path in paths.items():
        model = AutoExtractor.from_pretrained(str(path)).to(device)
        result = evaluate(model, examples, metadata, max_tokens)
        del model
        (run_dir / f"oof_{name}_predictions.json").write_text(
            json.dumps(result["predictions"])
        )
        _, scored = score_policy(
            tasks,
            result["predictions"],
            metadata,
            bias if name == "finetuned" else 0.0,
            split="test",
        )
        scored.to_json(
            run_dir / f"oof_{name}_outcomes.jsonl", orient="records", lines=True
        )
        report[name] = summarize_scores(scored, by=()).iloc[0].to_dict()
    (run_dir / "oof_report.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.item()) + "\n"
    )
    return report


def paired_counts(pre: pd.DataFrame, ft: pd.DataFrame) -> dict:
    """Exact McNemar on tasks with a gold target (recall) and on all tasks (correct decision)."""
    from scipy.stats import binomtest

    key = ["ref_id", "index"]
    joined = pre[key + ["gold_id", "outcome"]].merge(
        ft[key + ["outcome"]], on=key, suffixes=("_pre", "_ft"), validate="one_to_one"
    )
    out = {}
    for name, subset in (
        ("gold_tasks", joined[joined.gold_id.notna()]),
        ("no_gold_tasks", joined[joined.gold_id.isna()]),
    ):
        good = "correct_link" if name == "gold_tasks" else "correct_abstain"
        pre_ok, ft_ok = subset.outcome_pre.eq(good), subset.outcome_ft.eq(good)
        only_ft, only_pre = int((ft_ok & ~pre_ok).sum()), int((pre_ok & ~ft_ok).sum())
        discordant = only_ft + only_pre
        out[name] = {
            "tasks": len(subset),
            "pretrained_correct": int(pre_ok.sum()),
            "finetuned_correct": int(ft_ok.sum()),
            "only_finetuned": only_ft,
            "only_pretrained": only_pre,
            "mcnemar_p": (
                float(binomtest(only_ft, discordant, 0.5).pvalue) if discordant else 1.0
            ),
        }
    return out


def summarize(
    run_dirs: list[Path],
    output: Path,
    bias: float | None = None,
) -> dict:
    """Rescore saved test probabilities; ``bias`` overrides each run's own bias."""
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in MODELS}
    for run_dir in run_dirs:
        manifest, data_dir = run_manifest(run_dir)
        policy = manifest.get("best_policy")
        fold_bias = (
            bias
            if bias is not None
            else float(policy["none_logit_bias"] if policy else 0.0)
        )
        metadata = read_split(data_dir, "test", allow_test=True)[1]
        tasks = held_out_tasks(data_dir)
        for name in MODELS:
            rows = json.loads((run_dir / f"oof_{name}_predictions.json").read_text())
            _, scored = score_policy(
                tasks,
                rows,
                metadata,
                fold_bias if name == "finetuned" else 0.0,
                split="test",
            )
            frames[name].append(scored.assign(fold=data_dir.name))
    frames = {
        name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()
    }
    rows = []
    for name, frame in frames.items():
        rows.append(summarize_scores(frame, by=()).assign(model=name, scope="all"))
        rows.append(
            summarize_scores(frame, by=("fold",)).assign(model=name, scope="per_fold")
        )
        rows.append(
            summarize_scores(frame, by=("index",)).assign(model=name, scope="per_index")
        )
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output / "oof_summary.csv", index=False)
    paired = paired_counts(frames["pretrained"], frames["finetuned"])
    (output / "oof_paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(table[table.scope.eq("all")].to_string(index=False))
    print(json.dumps(paired, indent=2))
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    s = sub.add_parser("summarize")
    s.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--bias", type=float, help="Frozen bias; default is each run's own")
    args = parser.parse_args()
    if args.command == "predict":
        print(json.dumps(predict(args.data_dir, args.run_dir, args.device), indent=2))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summarize(args.run_dirs, args.output_dir, args.bias)


if __name__ == "__main__":
    main()
