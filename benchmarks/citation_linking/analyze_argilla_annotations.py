"""Analyse the exported citation-linking annotations: per-index quality and coverage.

Input is the JSONL written by ``export_argilla_annotations.py``. One row there is one
(reference, citation index) decision, carrying the human label plus whether the linker
returned a candidate at all.

Outcome taxonomy (five-way, unambiguous)
----------------------------------------
    correct_link      candidate returned and it is the right work
    wrong_link        candidate returned, wrong work, but the index does hold the work
    spurious_link     candidate returned, and the index holds no record at all
    missed_link       no candidate returned, but the index does hold the work
    correct_abstain   no candidate returned, and the index holds no record

Collapsed to a 2x2 for precision / recall / F1:

    TP = correct_link
    FP = wrong_link + spurious_link
    FN = missed_link + wrong_link
    TN = correct_abstain

``wrong_link`` is deliberately counted twice — a wrong candidate is both a false answer
and a missed true answer — so TP + FP + FN + TN != N. The five-way counts are always
reported next to the metrics so nothing is hidden by that choice.

Coverage is kept separate from quality:

    coverage_actual   = (correct_link + wrong_link + missed_link) / N   index ceiling
    coverage_achieved = correct_link / N                               what we got
    coverage_gap      = actual - achieved                              lost by the linker

Usage
-----
    python analyze_argilla_annotations.py                 # CSVs + summary.json

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from export_argilla_annotations import load_local  # noqa: E402

logger = logging.getLogger("analyze_argilla_annotations")

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "argilla_annotations" / "citation_linking_annotations.jsonl"
DEFAULT_OUT_DIR = HERE / "argilla_annotations" / "analysis"

OUTCOMES = ["correct_link", "wrong_link", "spurious_link", "missed_link", "correct_abstain"]
MULTI_REF_LABEL = "multi_ref_context"
EXCLUDED_LABELS = {"unannotated", "ambiguous", MULTI_REF_LABEL}
Z = 1.959963985  # 95%


# ---------------------------------------------------------------------------
# Small numeric helpers (pure — exercised by --self-check)
# ---------------------------------------------------------------------------
def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def wilson(successes: int, total: int, z: float = Z) -> tuple[Optional[float], Optional[float]]:
    """Wilson score interval — honest on the n=100 per-source cells, unlike normal approx."""
    if not total:
        return None, None
    denominator = total + z**2
    center = (successes + z**2 / 2) / denominator
    half = z * math.sqrt(successes * (total - successes) / total + z**2 / 4) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def outcome(label: str, candidate_missing: bool) -> Optional[str]:
    """Map one annotated row onto the five-way taxonomy. None = not evaluable."""
    if label in EXCLUDED_LABELS:
        return None
    if label == "correct_match":
        return "correct_link"
    if label == "wrong_match":
        return "missed_link" if candidate_missing else "wrong_link"
    if label == "no_record_in_index":
        return "correct_abstain" if candidate_missing else "spurious_link"
    raise ValueError(f"unknown label: {label!r}")


def metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    """Five-way outcome counts → the full metric set for one group."""
    correct = counts.get("correct_link", 0)
    wrong = counts.get("wrong_link", 0)
    spurious = counts.get("spurious_link", 0)
    missed = counts.get("missed_link", 0)
    abstained = counts.get("correct_abstain", 0)
    n = correct + wrong + spurious + missed + abstained

    tp, fp, fn, tn = correct, wrong + spurious, missed + wrong, abstained
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + spurious)
    f1 = (
        safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    exists = correct + wrong + missed

    precision_lo, precision_hi = wilson(tp, tp + fp)
    recall_lo, recall_hi = wilson(tp, tp + fn)
    coverage_lo, coverage_hi = wilson(exists, n)

    return {
        "n_eval": n,
        **{name: counts.get(name, 0) for name in OUTCOMES},
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "precision_lo": precision_lo,
        "precision_hi": precision_hi,
        "recall": recall,
        "recall_lo": recall_lo,
        "recall_hi": recall_hi,
        "f1": f1,
        "accuracy": safe_div(tp + tn, n),
        "specificity": specificity,
        "balanced_accuracy": (
            (recall + specificity) / 2 if recall is not None and specificity is not None else None
        ),
        "mcc": safe_div(tp * tn - fp * fn, mcc_denominator),
        "link_rate": safe_div(correct + wrong + spurious, n),
        "abstain_rate": safe_div(missed + abstained, n),
        "abstain_precision": safe_div(abstained, abstained + missed),
        "coverage_actual": safe_div(exists, n),
        "coverage_actual_lo": coverage_lo,
        "coverage_actual_hi": coverage_hi,
        "coverage_achieved": safe_div(correct, n),
        "coverage_gap": safe_div(exists - correct, n),
    }


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------
def prepare(
    rows: List[Dict[str, Any]],
    exclude_needs_review: bool,
    skip_scope: str = "reference",
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = {"label", "candidate_missing", "index", "source", "ref_id"} - set(df.columns)
    if missing:
        raise RuntimeError(f"input is missing required columns: {sorted(missing)}")

    df["outcome"] = [
        outcome(label, bool(candidate_missing))
        for label, candidate_missing in zip(df["label"], df["candidate_missing"])
    ]
    df["excluded_reason"] = df["label"].where(df["label"].isin(EXCLUDED_LABELS))

    if skip_scope == "reference":
        # A reference that packs several publications is packed for every index, not just the
        # one whose annotator marked it — drop all of its rows so the indexes stay comparable.
        multi_ref_ids = set(df.loc[df["label"] == MULTI_REF_LABEL, "ref_id"])
        spread = df["ref_id"].isin(multi_ref_ids)
        df.loc[spread, "outcome"] = None
        df.loc[spread, "excluded_reason"] = MULTI_REF_LABEL
    elif skip_scope != "row":
        raise ValueError(f"skip_scope must be 'reference' or 'row', got {skip_scope!r}")
    if exclude_needs_review:
        df.loc[df["needs_review"].fillna(False).astype(bool), "outcome"] = None
        df.loc[df["excluded_reason"].isna() & df["outcome"].isna(), "excluded_reason"] = (
            "needs_review"
        )
    df["exists"] = df["outcome"].isin(["correct_link", "wrong_link", "missed_link"])
    df["linked_correct"] = df["outcome"] == "correct_link"
    return df


def group_metrics(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """One metric row per group, with the excluded-row bookkeeping alongside."""
    records = []
    for key, group in df.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        evaluable = group[group["outcome"].notna()]
        record = dict(zip(group_cols, key_tuple))
        record["n_rows"] = len(group)
        record["n_excluded"] = len(group) - len(evaluable)
        # Scoped to evaluable rows: a flagged row inside an excluded reference is already gone.
        record["n_needs_review"] = int(
            evaluable["needs_review"].fillna(False).astype(bool).sum()
        )
        record.update(metrics(evaluable["outcome"].value_counts().to_dict()))
        records.append(record)
    return pd.DataFrame(records).sort_values(list(group_cols)).reset_index(drop=True)


def add_macro_f1(by_index: pd.DataFrame, by_index_source: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 = unweighted mean over corpus sources, so big sources cannot dominate."""
    macro = by_index_source.groupby("index")["f1"].mean().rename("macro_f1_over_sources")
    return by_index.merge(macro, on="index", how="left")


# ---------------------------------------------------------------------------
# Coverage across indexes
# ---------------------------------------------------------------------------
def coverage_pivots(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Complete-case ref x index views: overlap, unique contribution, union ceiling."""
    evaluable = df[df["outcome"].notna()]
    exists = evaluable.pivot_table(
        index="ref_id", columns="index", values="exists", aggfunc="first"
    )
    achieved = evaluable.pivot_table(
        index="ref_id", columns="index", values="linked_correct", aggfunc="first"
    )
    complete = exists.dropna().index
    dropped = sorted(set(exists.index) - set(complete))
    exists = exists.loc[complete].astype(bool)
    achieved = achieved.loc[complete].astype(bool)
    indexes = list(exists.columns)

    n_exists = exists.sum(axis=1)
    histogram = pd.DataFrame(
        {
            "n_indexes_holding_work": range(len(indexes) + 1),
            "n_refs": [int((n_exists == k).sum()) for k in range(len(indexes) + 1)],
        }
    )
    histogram["share"] = histogram["n_refs"] / max(len(exists), 1)

    per_index = pd.DataFrame(
        {
            "index": indexes,
            "n_refs": len(exists),
            "exists": [int(exists[i].sum()) for i in indexes],
            "exists_share": [exists[i].mean() for i in indexes],
            "unique_to_index": [
                int((exists[i] & (n_exists == 1)).sum()) for i in indexes
            ],
            "linked_correct": [int(achieved[i].sum()) for i in indexes],
            "linked_correct_share": [achieved[i].mean() for i in indexes],
        }
    )

    overlap = []
    for left in indexes:
        for right in indexes:
            if left >= right:
                continue
            both = int((exists[left] & exists[right]).sum())
            either = int((exists[left] | exists[right]).sum())
            overlap.append(
                {
                    "index_a": left,
                    "index_b": right,
                    "both": both,
                    "only_a": int((exists[left] & ~exists[right]).sum()),
                    "only_b": int((exists[right] & ~exists[left]).sum()),
                    "neither": int((~exists[left] & ~exists[right]).sum()),
                    "either": either,
                    "jaccard": safe_div(both, either),
                }
            )

    union = pd.DataFrame(
        [
            {
                "scope": "all indexes (union)",
                "n_refs": len(exists),
                "exists_any": int((n_exists > 0).sum()),
                "exists_any_share": safe_div(int((n_exists > 0).sum()), len(exists)),
                "linked_correct_any": int(achieved.any(axis=1).sum()),
                "linked_correct_any_share": safe_div(
                    int(achieved.any(axis=1).sum()), len(exists)
                ),
            }
        ]
    )

    source_by_ref = evaluable.drop_duplicates("ref_id").set_index("ref_id")["source"]
    by_source = []
    for source, refs in source_by_ref.loc[complete].groupby(source_by_ref.loc[complete]):
        subset_exists = exists.loc[refs.index]
        subset_achieved = achieved.loc[refs.index]
        row = {"source": source, "n_refs": len(subset_exists)}
        for index_name in indexes:
            row[f"{index_name}_exists"] = int(subset_exists[index_name].sum())
            row[f"{index_name}_correct"] = int(subset_achieved[index_name].sum())
        row["exists_any"] = int(subset_exists.any(axis=1).sum())
        row["linked_correct_any"] = int(subset_achieved.any(axis=1).sum())
        by_source.append(row)

    return {
        "coverage_dropped_refs": pd.DataFrame({"ref_id": dropped}),
        "coverage_histogram": histogram,
        "coverage_by_index": per_index,
        "coverage_overlap": pd.DataFrame(overlap),
        "coverage_union": union,
        "coverage_by_source": pd.DataFrame(by_source),
    }


# ---------------------------------------------------------------------------
# Similarity heuristic, scored against the gold labels
# ---------------------------------------------------------------------------
def similarity_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Would `is_match_by_similarity` work as an auto-accept gate over returned candidates?"""
    linked = df[df["outcome"].notna() & ~df["candidate_missing"].astype(bool)].copy()
    linked["flag"] = linked["is_match_by_similarity"].astype(str).str.lower() == "true"
    linked["gold"] = linked["outcome"] == "correct_link"

    records = []
    for index_name, group in list(linked.groupby("index")) + [("ALL", linked)]:
        tp = int((group["flag"] & group["gold"]).sum())
        fp = int((group["flag"] & ~group["gold"]).sum())
        fn = int((~group["flag"] & group["gold"]).sum())
        tn = int((~group["flag"] & ~group["gold"]).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        records.append(
            {
                "index": index_name,
                "n_candidates": len(group),
                "flag_rate": safe_div(tp + fp, len(group)),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "f1": (
                    safe_div(2 * precision * recall, precision + recall)
                    if precision and recall
                    else None
                ),
                "accuracy": safe_div(tp + tn, len(group)),
                "baseline_precision_accept_all": safe_div(
                    int(group["gold"].sum()), len(group)
                ),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Run summary (data only — all prose lives in the hand-written report)
# ---------------------------------------------------------------------------
def run_summary(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Bookkeeping the report needs but cannot read off the metric tables."""
    coverage = tables["coverage_by_index"]
    return {
        "n_rows": len(df),
        "n_references": int(df["ref_id"].nunique()),
        "indexes": sorted(df["index"].unique()),
        "sources": sorted(df["source"].unique()),
        "n_evaluable": int(df["outcome"].notna().sum()),
        "n_excluded": int(df["outcome"].isna().sum()),
        "excluded_by_reason": df["excluded_reason"].value_counts().to_dict(),
        "n_multi_ref_references": int(df.loc[df["label"] == MULTI_REF_LABEL, "ref_id"].nunique()),
        "n_manually_fixed_rows": (
            int(df["manual_fix"].notna().sum()) if "manual_fix" in df.columns else 0
        ),
        "n_needs_review": int(df["needs_review"].fillna(False).astype(bool).sum()),
        "n_needs_review_evaluable": int(
            (df["needs_review"].fillna(False).astype(bool) & df["outcome"].notna()).sum()
        ),
        "needs_review_by_reason": (
            df.loc[df["review_reason"].astype(str) != "", "review_reason"].value_counts().to_dict()
            if "review_reason" in df.columns
            else {}
        ),
        "needs_review_by_index": (
            df.loc[df["needs_review"].fillna(False).astype(bool), "index"].value_counts().to_dict()
        ),
        "n_eval_per_index": df[df["outcome"].notna()]["index"].value_counts().to_dict(),
        "n_eval_equal_across_indexes": df[df["outcome"].notna()]["index"]
        .value_counts()
        .nunique()
        == 1,
        "n_complete_case_refs": int(coverage["n_refs"].iloc[0]) if len(coverage) else 0,
        "n_dropped_refs": len(tables["coverage_dropped_refs"]),
        "dropped_refs": list(tables["coverage_dropped_refs"]["ref_id"]),
    }


def analyse(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_index_source = group_metrics(df, ["index", "source"])
    tables: Dict[str, pd.DataFrame] = {
        "outcome_counts": (
            df[df["outcome"].notna()]
            .pivot_table(index="index", columns="outcome", values="ref_id", aggfunc="count")
            .reindex(columns=OUTCOMES)
            .fillna(0)
            .astype(int)
            .reset_index()
        ),
        "metrics_by_index": add_macro_f1(group_metrics(df, ["index"]), by_index_source),
        "metrics_by_index_source": by_index_source,
        "metrics_by_source": group_metrics(df, ["source"]),
        "similarity_heuristic": similarity_heuristic(df),
    }
    tables.update(coverage_pivots(df))
    tables["overall"] = group_metrics(df.assign(all="all"), ["all"])
    return tables



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--exclude-needs-review",
        action="store_true",
        help="drop rows with contradictory annotations instead of keeping them",
    )
    parser.add_argument(
        "--skip-scope",
        choices=["reference", "row"],
        default="reference",
        help="'[SKIP]' rows exclude the whole reference across indexes (default) or only that row",
    )
    # parser.add_argument("--self-check", action="store_true", help="run asserts and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


    if not args.input.exists():
        raise RuntimeError(f"{args.input} not found — run export_argilla_annotations.py first")

    df = prepare(load_local(args.input), args.exclude_needs_review, args.skip_scope)
    tables = analyse(df)
    summary = run_summary(df, tables)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
        logger.info("%-28s %3d rows", f"{name}.csv", len(table))
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("summary.json %s", json.dumps(summary, default=str))
    logger.info("wrote %d CSVs + summary.json to %s", len(tables), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
