"""Export audited GLiNER listwise training data.

Subcommands: ``export`` builds the audited examples from a frozen retrieval pool,
``full`` re-exports every eligible task as train data (final model), and ``official``
re-cuts the examples with the published ``reference_disjoint`` split files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from benchmarks.citation_linking.dataset.build_annotation_dataset import (
    normalize_id,
    SPLITS,
)
from benchmarks.citation_linking.dataset.corrections import REVIEW_OVERRIDES
from benchmarks.citation_linking.experiments.linking_eval import (
    digest,
    KEY,
    prepare_pool,
)
from benchmarks.citation_linking.experiments.rerankers import gliner_listwise_request

ROOT = Path(__file__).resolve().parents[3]


def read_annotations(paths: list[Path]) -> dict[tuple[str, str], dict]:
    annotations = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[k] for k in KEY)
            if key in annotations:
                raise ValueError(f"Duplicate annotation: {key}")
            annotations[key] = row
    return annotations


def label_evidence(row: dict, annotation: dict | None) -> str:
    """Require consistent submitted annotation evidence, including release overrides."""
    if annotation is None:
        return "missing_annotation"
    if " ".join(annotation["original_ref_string"].split()) != " ".join(
        row["reference"].split()
    ):
        return "annotation_text_mismatch"
    if annotation.get("needs_review") is not False or annotation.get("review_reason"):
        return "annotation_requires_review"
    for question in ("is_match_correct", "no_match", "correct_id"):
        values = annotation.get(f"{question}.responses") or []
        statuses = annotation.get(f"{question}.responses.status") or []
        if len(values) != len(statuses):
            return "invalid_response_evidence"
        submitted = {v for v, s in zip(values, statuses) if s == "submitted"}
        if len(submitted) > 1:
            return "conflicting_responses"
        value = annotation.get(question)
        if value is not None and submitted != {value}:
            return "unsubmitted_response"

    label = annotation.get("label")
    if label == "no_record_in_index" and annotation.get("no_match") == "true":
        approved = None
    elif label == "correct_match" and annotation.get("is_match_correct") == "true":
        approved = normalize_id(annotation["matched_id"], row["index"])
    elif label == "wrong_match" and annotation.get("gold_id"):
        approved = normalize_id(annotation["gold_id"], row["index"])
    else:
        return "unresolved_annotation"
    approved = (
        REVIEW_OVERRIDES.get(row["ref_id"], {})
        .get("gold_id", {})
        .get(row["index"], approved)
    )
    gold = normalize_id(row["gold_id"], row["index"]) if row["gold_id"] else None
    if approved != gold:
        return "annotation_gold_mismatch"
    if gold is not None:
        return "approved_gold"
    if label == "no_record_in_index":
        return "annotated_no_record"
    return "cleared_in_review"


def shuffle_training_example(
    example: dict, metadata: dict, *, epoch: int, seed: int = 42
) -> tuple[dict, dict]:
    """Reassign anonymous candidate labels each epoch, keeping supervision aligned."""
    if metadata["split"] != "train" or epoch < 0:
        raise ValueError("Only training examples may be reshuffled; epoch must be >= 0")
    example, metadata = copy.deepcopy(example), copy.deepcopy(metadata)
    classification = example["output"]["classifications"][0]
    mapping = metadata["candidate_ids"]
    options = list(mapping)
    random.Random(digest([seed, epoch, metadata["ref_id"], metadata["index"]])).shuffle(
        options
    )
    renaming = {old: f"candidate_{i}" for i, old in enumerate(options, 1)}
    descriptions = classification["label_descriptions"]
    classification["label_descriptions"] = {
        **{renaming[old]: descriptions[old] for old in options},
        "none": descriptions["none"],
    }
    classification["labels"] = list(classification["label_descriptions"])
    target = classification["true_label"][0]
    classification["true_label"] = [renaming.get(target, target)]
    metadata["candidate_ids"] = {renaming[old]: mapping[old] for old in options}
    metadata["target"] = classification["true_label"][0]
    return example, metadata


def build_examples(
    pool_path: Path, annotations: dict, *, seed: int = 42, field_limit: int = 1000
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    tasks, pairs = prepare_pool(pool_path, seed=seed, field_limit=field_limit)
    groups = {key: group for key, group in pairs.groupby(KEY, sort=False)}
    examples = {split: [] for split in SPLITS}
    metadata = {split: [] for split in SPLITS}
    audit = []
    for row in tasks.to_dict("records"):
        if (
            row.get("injected_gold_id")
            or row.get("injected_metadata_source")
            or any(c.get("injected") or c.get("is_injected") for c in row["candidates"])
        ):
            raise ValueError(
                "Training export requires real retrieval, not oracle injection"
            )
        if row["n_candidates"] > 10:
            raise ValueError("Training export requires at most 10 retrieved candidates")
        key = tuple(row[k] for k in KEY)
        annotation = annotations.get(key)
        evidence = label_evidence(row, annotation)
        info = {
            **{k: row[k] for k in KEY + ["split", "gold_id", "n_candidates"]},
            "reference_sha256": digest(" ".join(row["reference"].split())),
            "label_evidence": evidence,
            "annotation_record_id": (annotation or {}).get("record_id"),
            "annotation_label": (annotation or {}).get("label"),
            "eligible": False,
            "target": None,
            "reason": evidence,
            "empty_candidates": not row["candidates"],
        }
        audit.append(info)
        if evidence not in {
            "approved_gold",
            "annotated_no_record",
            "cleared_in_review",
        }:
            continue
        if not row["candidates"]:
            info["reason"] = "empty_candidates"
            continue
        group = groups[key]
        text, labels = gliner_listwise_request(
            list(zip(group.citation_text, group.candidate_text))
        )
        mapping = {
            f"candidate_{i}": identifier
            for i, identifier in enumerate(group.candidate_id, 1)
        }
        gold = normalize_id(row["gold_id"], row["index"]) if row["gold_id"] else None
        target = next((label for label, cid in mapping.items() if cid == gold), "none")
        reason = (
            "gold_retrieved"
            if target != "none"
            else "gold_not_retrieved" if gold is not None else evidence
        )
        info.update(eligible=True, target=target, reason=reason)
        examples[row["split"]].append(
            {
                "input": text,
                "output": {
                    "classifications": [
                        {
                            "task": "link",
                            "labels": list(labels),
                            "true_label": [target],
                            "multi_label": False,
                            "label_descriptions": labels,
                        }
                    ]
                },
            }
        )
        metadata[row["split"]].append(
            {
                **info,
                "candidate_ids": mapping,
                "clipped_pair_fields": int(group.clipped_fields.sum()),
                "token_truncation": "not_checked",
            }
        )

    train_refs = {row["ref_id"] for row in metadata["train"]}
    train_texts = {row["reference_sha256"] for row in metadata["train"]}
    for row in audit + [r for split in SPLITS for r in metadata[split]]:
        row["reference_seen_in_train"] = row["ref_id"] in train_refs
        row["citation_text_seen_in_train"] = row["reference_sha256"] in train_texts
    return examples, metadata, audit


def export_training_data(
    pool_path: Path,
    annotation_paths: list[Path],
    output_dir: Path,
    *,
    seed: int = 42,
    field_limit: int = 1000,
) -> dict:
    if field_limit < 1:
        raise ValueError("field_limit must be positive")
    examples, metadata, audit = build_examples(
        pool_path,
        read_annotations(annotation_paths),
        seed=seed,
        field_limit=field_limit,
    )
    # A fresh directory prevents stale split files or manifests from masquerading as a run.
    output_dir.mkdir(parents=True, exist_ok=False)

    def write_rows(name: str, rows: list[dict]) -> None:
        (output_dir / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            )
        )

    for split in SPLITS:
        write_rows(f"{split}.jsonl", examples[split])
        write_rows(f"{split}.metadata.jsonl", metadata[split])
    write_rows("audit.jsonl", audit)
    write_rows(
        "needs_review.jsonl",
        [
            row
            for row in audit
            if not row["eligible"] and row["reason"] != "empty_candidates"
        ],
    )
    summary = {}
    for split in SPLITS:
        records = [row for row in audit if row["split"] == split]
        summary[split] = {
            "pool_tasks": len(records),
            "exported": len(examples[split]),
            "candidate_targets": sum(
                row["target"] not in {None, "none"} for row in records
            ),
            "none_targets": sum(row["target"] == "none" for row in records),
            "empty_candidates": sum(row["empty_candidates"] for row in records),
            "reasons": dict(Counter(row["reason"] for row in records)),
            "exported_seen_reference_tasks": sum(
                row["reference_seen_in_train"] for row in metadata[split]
            ),
            "exported_unseen_reference_tasks": sum(
                not row["reference_seen_in_train"] for row in metadata[split]
            ),
            "exported_seen_citation_text_tasks": sum(
                row["citation_text_seen_in_train"] for row in metadata[split]
            ),
            "exported_distinct_references": len(
                {row["ref_id"] for row in metadata[split]}
            ),
        }
    sources = [
        pool_path,
        *annotation_paths,
        Path(__file__),
        Path(__file__).with_name("linking_eval.py"),
        Path(__file__).with_name("rerankers.py"),
        Path(__file__).parents[1] / "dataset/build_annotation_dataset.py",
    ]
    manifest = {
        "status": "complete",
        "model_id": "fastino/GLiNER2.5-multi-Decide",
        "objective": "listwise_single_label_link",
        "pool": str(pool_path.resolve()),
        "seed": seed,
        "field_limit": field_limit,
        "sources_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.glob("*.jsonl"))
        },
        "summary": summary,
        "none_policy": "Nonempty pool with approved gold absent, or consistent submitted no_record_in_index annotation, or gold cleared in review.",
        "negative_evidence_limit": "Historical human annotations, not a fresh manual review of every current candidate.",
        "split_policy": "Existing (reference, index) assignments; seen flags use only eligible train examples.",
        "training_status": "not_started",
        "token_truncation": "not_checked; field clipping is recorded separately",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


ORIGINAL_SPLITS = ("train", "validation", "test")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


def export_assignment(
    examples: list[dict],
    metadata: list[dict],
    audit: list[dict],
    split_of: dict[tuple[str, str], str],
    *,
    output_dir: Path,
    source_manifest: dict,
    details: dict,
) -> dict:
    """Write a trainer-readable export for any task -> split assignment."""
    output_dir.mkdir(parents=True, exist_ok=False)
    new_audit = [{**r, "split": split_of[(r["ref_id"], r["index"])]} for r in audit]
    train_refs = {
        r["ref_id"] for r in new_audit if r["split"] == "train" and r["eligible"]
    }
    train_texts = {
        r["reference_sha256"]
        for r in new_audit
        if r["split"] == "train" and r["eligible"]
    }
    for row in new_audit:
        row["reference_seen_in_train"] = row["ref_id"] in train_refs
        row["citation_text_seen_in_train"] = row["reference_sha256"] in train_texts
    per_split = {s: ([], []) for s in SPLITS}
    for example, meta in zip(examples, metadata):
        split = split_of[(meta["ref_id"], meta["index"])]
        meta = {
            **meta,
            "split": split,
            "reference_seen_in_train": meta["ref_id"] in train_refs,
            "citation_text_seen_in_train": meta["reference_sha256"] in train_texts,
        }
        per_split[split][0].append(example)
        per_split[split][1].append(meta)
    for split, (ex, md) in per_split.items():
        write_rows(output_dir / f"{split}.jsonl", ex)
        write_rows(output_dir / f"{split}.metadata.jsonl", md)
    write_rows(output_dir / "audit.jsonl", new_audit)
    summary = {
        s: {
            "exported": len(per_split[s][0]),
            "references": len({m["ref_id"] for m in per_split[s][1]}),
            "targets": dict(Counter(m["reason"] for m in per_split[s][1])),
            "unseen_reference_tasks": sum(
                not m["reference_seen_in_train"] for m in per_split[s][1]
            ),
        }
        for s in SPLITS
    }
    manifest = {
        "status": "complete",
        "model_id": source_manifest["model_id"],
        "objective": source_manifest["objective"],
        **details,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output_dir.glob("*.jsonl"))
        },
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_source(source_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Checksum-verified (manifest, examples, metadata, audit), sorted by (ref, index)."""
    source_manifest = json.loads((source_dir / "manifest.json").read_text())

    def load(name: str) -> list[dict]:
        data = (source_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != source_manifest["files_sha256"][name]:
            raise ValueError(f"Source checksum mismatch: {name}")
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    examples, metadata = [], []
    for split in ORIGINAL_SPLITS:
        examples += load(f"{split}.jsonl")
        metadata += load(f"{split}.metadata.jsonl")
    audit = load("audit.jsonl")
    order = sorted(
        range(len(metadata)),
        key=lambda i: (metadata[i]["ref_id"], metadata[i]["index"]),
    )
    return (
        source_manifest,
        [examples[i] for i in order],
        [metadata[i] for i in order],
        audit,
    )


def export_full(source_dir: Path, output_dir: Path) -> dict:
    """Every eligible task as train data for the final model; no held-out split remains."""
    source_manifest, examples, metadata, audit = load_source(source_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_meta = [
        {
            **m,
            "split": "train",
            "reference_seen_in_train": True,
            "citation_text_seen_in_train": True,
        }
        for m in metadata
    ]
    write_rows(output_dir / "train.jsonl", examples)
    write_rows(output_dir / "train.metadata.jsonl", train_meta)
    for split in ("validation", "test"):
        write_rows(output_dir / f"{split}.jsonl", [])
        write_rows(output_dir / f"{split}.metadata.jsonl", [])
    write_rows(output_dir / "audit.jsonl", [{**r, "split": "train"} for r in audit])
    manifest = {
        "status": "complete",
        "model_id": source_manifest["model_id"],
        "objective": source_manifest["objective"],
        "split_policy": "all eligible tasks in train; no validation or test (final model)",
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output_dir.glob("*.jsonl"))
        },
        "summary": {
            "train": {
                "exported": len(examples),
                "references": len({m["ref_id"] for m in metadata}),
                "targets": dict(Counter(m["reason"] for m in metadata)),
            }
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_official(source_dir: Path, official_dir: Path, output_dir: Path) -> dict:
    """Export with the published ``reference_disjoint`` split files (parquet)."""
    import pandas as pd

    source_manifest, examples, metadata, audit = load_source(source_dir)
    split_of = {}
    for split in SPLITS:
        frame = pd.read_parquet(
            official_dir / "reference_disjoint" / f"{split}.parquet"
        )
        for ref_id, index in zip(frame.ref_id, frame["index"]):
            if (ref_id, index) in split_of:
                raise ValueError(f"Task in two official splits: {(ref_id, index)}")
            split_of[(ref_id, index)] = split
    if set(split_of) != {(r["ref_id"], r["index"]) for r in audit}:
        raise ValueError("Official split does not cover exactly the audited tasks")
    revision = json.loads((official_dir / "revision.json").read_text())
    return export_assignment(
        examples,
        metadata,
        audit,
        split_of,
        output_dir=output_dir,
        source_manifest=source_manifest,
        details={
            "split_policy": "official reference_disjoint (yurui983/citation_linking)",
            "official_split_revision": revision["revision"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Audited examples from the retrieval pool")
    export.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "logs/citation_link_tryings/four_index/search_top10.jsonl",
        help="Real-retrieval pool; the three-index one is logs/citation_link_tryings/search_top10.jsonl",
    )
    export.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).parents[1] / "data/annotations/corrected" / name
            for name in ("annotations.jsonl", "dasi_ebda.jsonl")
        ],
        help="Corrected annotations written by dataset/build_annotation_dataset.py",
    )
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--seed", type=int, default=42)
    full = sub.add_parser("full", help="All eligible tasks as train, no held-out split")
    full.add_argument("--source-dir", type=Path, required=True)
    full.add_argument("--output-dir", type=Path, required=True)
    official = sub.add_parser("official", help="Published reference_disjoint split")
    official.add_argument("--source-dir", type=Path, required=True)
    official.add_argument(
        "--official-dir",
        type=Path,
        required=True,
        help="Dataset dir with reference_disjoint/*.parquet and revision.json",
    )
    official.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        manifest = export_training_data(
            args.pool, args.annotations, args.output_dir, seed=args.seed
        )
        print(json.dumps(manifest["summary"], indent=2))
    elif args.command == "full":
        print(json.dumps(export_full(args.source_dir, args.output_dir)["summary"]))
    else:
        manifest = export_official(args.source_dir, args.official_dir, args.output_dir)
        print(json.dumps(manifest["summary"]))


if __name__ == "__main__":
    main()
