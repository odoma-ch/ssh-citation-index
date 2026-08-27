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
    python analyze_argilla_annotations.py                 # tables to stdout + CSVs
    python analyze_argilla_annotations.py --self-check     # asserts, no data needed
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
EXCLUDED_LABELS = {"unannotated", "ambiguous"}
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
def prepare(rows: List[Dict[str, Any]], exclude_needs_review: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = {"label", "candidate_missing", "index", "source", "ref_id"} - set(df.columns)
    if missing:
        raise RuntimeError(f"input is missing required columns: {sorted(missing)}")

    df["outcome"] = [
        outcome(label, bool(candidate_missing))
        for label, candidate_missing in zip(df["label"], df["candidate_missing"])
    ]
    df["excluded_reason"] = df["label"].where(df["label"].isin(EXCLUDED_LABELS))
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
        record["n_needs_review"] = int(group["needs_review"].fillna(False).astype(bool).sum())
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
# Rendering
# ---------------------------------------------------------------------------
def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny markdown renderer — avoids a tabulate dependency for pandas.to_markdown."""
    if df.empty:
        return "_(empty)_\n"
    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = [
            "" if value is None or (isinstance(value, float) and math.isnan(value))
            else floatfmt.format(value) if isinstance(value, float)
            else str(value)
            for value in formatted[column]
        ]
    header = "| " + " | ".join(formatted.columns) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).values]
    return "\n".join([header, divider, *body]) + "\n"


# Markdown keeps the readable subset; the CSVs keep every column.
HEADLINE_COLUMNS = [
    "index",
    "source",
    "n_eval",
    "n_excluded",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "abstain_precision",
    "coverage_actual",
    "coverage_achieved",
    "coverage_gap",
    "macro_f1_over_sources",
]


def headline(table: pd.DataFrame) -> pd.DataFrame:
    """Narrow the metric tables only — count/coverage tables are already readable."""
    if "precision" not in table.columns:
        return table
    return table[[column for column in HEADLINE_COLUMNS if column in table.columns]]


def build_report(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> str:
    excluded = df["excluded_reason"].value_counts().to_dict()
    lines = [
        "# Citation linking — annotation analysis",
        "",
        f"- rows: {len(df)}  ({df['ref_id'].nunique()} references x {df['index'].nunique()} indexes)",
        f"- evaluable rows: {int(df['outcome'].notna().sum())}",
        f"- excluded rows: {excluded or 'none'}",
        f"- needs_review rows: {int(df['needs_review'].fillna(False).astype(bool).sum())}",
        "",
        "TP = correct_link, FP = wrong_link + spurious_link, FN = missed_link + wrong_link,",
        "TN = correct_abstain. wrong_link counts in both FP and FN, so the four do not sum to N.",
        "A blank f1 means no true positives and zero recall, not a failed computation.",
        "",
    ]
    notes = {
        "metrics_by_index": (
            "coverage_* here use each index's own evaluable rows (openalex has 97 rows still "
            "pending, so its denominator is 403 vs 500). For an apples-to-apples cross-index "
            "comparison use 'Coverage per index' below — complete-case, same 403 refs for all."
        ),
        "metrics_by_source": (
            "pooled over index decisions — each reference contributes 3 rows (one per index), "
            "so n_eval counts decisions, not references."
        ),
        "coverage_by_index": (
            "complete-case: only refs annotated in every index. Dropped refs are listed in "
            "coverage_dropped_refs.csv."
        ),
    }
    titles = {
        "overall": "Overall (all indexes pooled)",
        "outcome_counts": "Outcome counts (five-way, per index)",
        "metrics_by_index": "Metrics per citation index",
        "metrics_by_index_source": "Metrics per index x corpus source",
        "metrics_by_source": "Metrics per corpus source (all indexes pooled)",
        "coverage_by_index": "Coverage per index (complete-case refs)",
        "coverage_histogram": "How many indexes hold each work",
        "coverage_overlap": "Pairwise coverage overlap",
        "coverage_union": "Union ceiling across indexes",
        "coverage_by_source": "Coverage per corpus source x index",
        "similarity_heuristic": "is_match_by_similarity scored as an auto-accept gate",
    }
    lines += ["Tables below show headline columns only — the CSVs carry every metric.", ""]
    for name, title in titles.items():
        if name in tables:
            lines += [f"## {title}", ""]
            if name in notes:
                lines += [f"_{notes[name]}_", ""]
            lines += [to_markdown(headline(tables[name])), ""]
    return "\n".join(lines)


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
# Self-check
# ---------------------------------------------------------------------------
def self_check() -> None:
    assert outcome("correct_match", False) == "correct_link"
    assert outcome("wrong_match", True) == "missed_link"
    assert outcome("wrong_match", False) == "wrong_link"
    assert outcome("no_record_in_index", True) == "correct_abstain"
    assert outcome("no_record_in_index", False) == "spurious_link"
    assert outcome("unannotated", False) is None

    counts = {
        "correct_link": 2,
        "wrong_link": 1,
        "spurious_link": 1,
        "missed_link": 1,
        "correct_abstain": 5,
    }
    m = metrics(counts)
    assert (m["TP"], m["FP"], m["FN"], m["TN"], m["n_eval"]) == (2, 2, 2, 5, 10)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["accuracy"] == 0.7
    assert abs(m["specificity"] - 5 / 6) < 1e-12
    assert abs(m["mcc"] - 6 / 28) < 1e-12  # (TP*TN-FP*FN)/sqrt(4*4*7*7)
    assert m["link_rate"] == 0.4 and m["abstain_rate"] == 0.6
    assert abs(m["abstain_precision"] - 5 / 6) < 1e-12
    assert m["coverage_actual"] == 0.4 and m["coverage_achieved"] == 0.2
    assert abs(m["coverage_gap"] - 0.2) < 1e-12
    assert m["precision_lo"] < 0.5 < m["precision_hi"]

    empty = metrics({})
    assert empty["n_eval"] == 0 and empty["precision"] is None and empty["f1"] is None

    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.4, (lo, hi)
    lo, hi = wilson(286, 614)  # matches the real openalex+matilda+wikidata pooled precision
    assert abs(lo - 0.4267) < 0.002 and abs(hi - 0.5054) < 0.002, (lo, hi)

    # Two refs x two indexes: ref A only in openalex, ref B in both.
    rows = [
        {"ref_id": "A", "index": "openalex", "source": "cex", "label": "correct_match",
         "candidate_missing": False, "needs_review": False, "is_match_by_similarity": "true"},
        {"ref_id": "A", "index": "wikidata", "source": "cex", "label": "no_record_in_index",
         "candidate_missing": True, "needs_review": False, "is_match_by_similarity": "false"},
        {"ref_id": "B", "index": "openalex", "source": "cex", "label": "wrong_match",
         "candidate_missing": True, "needs_review": False, "is_match_by_similarity": "false"},
        {"ref_id": "B", "index": "wikidata", "source": "cex", "label": "correct_match",
         "candidate_missing": False, "needs_review": False, "is_match_by_similarity": "false"},
    ]
    df = prepare(rows, exclude_needs_review=False)
    tables = analyse(df)
    coverage = tables["coverage_by_index"].set_index("index")
    assert coverage.loc["openalex", "exists"] == 2  # correct_link + missed_link
    assert coverage.loc["wikidata", "exists"] == 1
    assert coverage.loc["openalex", "unique_to_index"] == 1  # ref A
    assert coverage.loc["openalex", "linked_correct"] == 1
    overlap = tables["coverage_overlap"].iloc[0]
    assert (overlap["both"], overlap["only_a"], overlap["only_b"]) == (1, 1, 0)
    assert tables["coverage_union"].iloc[0]["exists_any"] == 2
    assert tables["coverage_union"].iloc[0]["linked_correct_any"] == 2
    heuristic = tables["similarity_heuristic"].set_index("index").loc["ALL"]
    assert heuristic["n_candidates"] == 2 and heuristic["TP"] == 1 and heuristic["FN"] == 1

    # Ref C annotated in wikidata but still pending in openalex → excluded from the pivot,
    # which is exactly the shape of the 97 pending openalex rows in the real export.
    partial = prepare(
        rows
        + [
            {"ref_id": "C", "index": "openalex", "source": "cex", "label": "unannotated",
             "candidate_missing": False, "needs_review": False, "is_match_by_similarity": "false"},
            {"ref_id": "C", "index": "wikidata", "source": "cex", "label": "correct_match",
             "candidate_missing": False, "needs_review": False, "is_match_by_similarity": "true"},
        ],
        exclude_needs_review=False,
    )
    partial_pivots = coverage_pivots(partial)
    assert list(partial_pivots["coverage_dropped_refs"]["ref_id"]) == ["C"]
    assert partial_pivots["coverage_by_index"]["n_refs"].eq(2).all()  # A, B only

    assert to_markdown(pd.DataFrame()).startswith("_(empty)_")
    print("✓ self-check passed")


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
    parser.add_argument("--self-check", action="store_true", help="run asserts and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.self_check:
        self_check()
        return 0

    if not args.input.exists():
        raise RuntimeError(f"{args.input} not found — run export_argilla_annotations.py first")

    df = prepare(load_local(args.input), args.exclude_needs_review)
    tables = analyse(df)
    report = build_report(df, tables)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("wrote %d CSVs + report.md to %s", len(tables), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
