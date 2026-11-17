"""Argilla annotations -> local raw export -> Hugging Face ``citation_linking_annotations``.

First of the two dataset scripts. It publishes the annotators' answers exactly as
submitted; ``build_annotation_dataset.py`` turns them into the reviewed gold dataset.

    export ARGILLA_API_TOKEN=... HF_TOKEN=...

    python export_argilla_annotations.py            # fetch, write data/annotations/raw/
    python export_argilla_annotations.py --push     # ... and publish to Hugging Face
    python export_argilla_annotations.py --no-fetch --push   # publish the files on disk

Two splits: ``annotations`` (the benchmark references) and ``dasi_ebda``. OpenAlex,
Matilda and Wikidata have one Argilla dataset per split; OpenCitations was annotated in
a single dataset that holds both and is divided here by ``source``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("export_argilla_annotations")

ARGILLA_API_URL = os.environ.get("ARGILLA_API_URL") or "https://argilla.graphia-ssh.eu/"
ARGILLA_API_TOKEN = os.environ.get("ARGILLA_API_TOKEN") or "argilla.apikey"

INDEXES = ["openalex", "matilda", "wikidata"]
ALL_INDEXES = [*INDEXES, "opencitations"]
# split -> Argilla dataset name per index
SPLIT_DATASETS = {
    "annotations": "citation_linking_{index}",
    "dasi_ebda": "citation_linking_dasi_ebda_{index}",
}
OPENCITATIONS_DATASET = "citation_linking_opencitations"  # both splits in one dataset
DASI_EBDA_SOURCES = {"dasi", "ebda"}
QUESTIONS = ["is_match_correct", "correct_id", "no_match"]

HF_REPO = "yurui983/citation_linking_annotations"
ANNOTATIONS = Path(__file__).resolve().parents[1] / "data/annotations"
RAW = ANNOTATIONS / "raw"
# Analysis tables published next to each split, when analyze_argilla_annotations.py
# has written them. They are computed after the reviewed corrections.
ANALYSIS_DIRS = {
    "annotations": ANNOTATIONS / "analysis/annotations",
    "dasi_ebda": ANNOTATIONS / "analysis/dasi_ebda",
}
ANALYSIS_FILES = [
    "overall",
    "metrics_by_index",
    "metrics_by_source",
    "metrics_by_index_source",
    "outcome_counts",
    "coverage_by_index",
    "coverage_overlap",
    "coverage_union",
    "opencitations_cross_index_review",
]
README = """---
configs:
- config_name: default
  data_files:
  - split: annotations
    path: data/annotations-*
  - split: dasi_ebda
    path: data/dasi_ebda-*
---

# Citation linking: raw annotations

Raw Argilla annotations for citation linking: one row per (reference, index) decision,
exactly as the annotators submitted them. Nothing is corrected here; `needs_review`
marks contradictory answers.

The reviewed gold dataset is `yurui983/citation_linking`. Every correction applied on
the way there (fixed answers, replaced or cleared targets, DOI rulings) is listed with
its reason in that dataset's `CORRECTIONS.md` and `build_report.json`.

## Splits

{splits}

## Columns

| Column | Meaning |
|---|---|
| `ref_id`, `source`, `original_ref_string` | The reference and the corpus it comes from. |
| `index` | `openalex`, `matilda`, `wikidata` or `opencitations`. |
| `matched_id`, `matched_doi`, `matched_result`, `matched_link` | The candidate shown to the annotator (`Not Found` when the baseline returned nothing). |
| `is_match_correct`, `no_match`, `correct_id` | The three answers. `*.responses`, `*.responses.users` and `*.responses.status` keep every submitted response. |
| `label` | Derived: `correct_match`, `wrong_match`, `no_record_in_index`, `multi_ref_context`, `ambiguous` or `unannotated`. |
| `gold_id` | The annotator's `correct_id`, cleaned of placeholders. |
| `needs_review`, `review_reason` | Set when the answers contradict each other or a wrong match has no corrected ID. |
| `search_method` | OpenCitations only: how the shown candidate was found. |

## Notes

- OpenCitations was annotated last (November 2026), by one annotator. Part of its
  candidates came from an ID lookup that reused DOIs / OpenAlex IDs confirmed in the
  other indexes, so its baseline metrics are not comparable with theirs.
- `analysis/<split>/` holds the metric tables, computed after the reviewed corrections.
  `opencitations_cross_index_review.csv` lists the references where annotators judged
  the same DOI differently across indexes; these were adjudicated in the gold dataset.
"""


# ---------------------------------------------------------------------------
# Row normalisation (pure — exercised by --self-check)
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """Coerce Argilla values (UUIDs, enums, ...) into JSON-serialisable ones."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _pick_response(raw: Dict[str, Any], question: str) -> Optional[Any]:
    """First submitted response for a question, falling back to the first draft."""
    values = raw.get(f"{question}.responses") or []
    statuses = raw.get(f"{question}.responses.status") or []
    for value, status in zip(values, statuses):
        if status == "submitted":
            return value
    return values[0] if values else None


def record_to_raw(record: Any) -> Dict[str, Any]:
    """Flatten one Argilla ``Record`` into a flat dict of primitives.

    Written by hand rather than using ``records.to_list(flatten=True)``: that helper
    merges fields and metadata with ``dict.update(**fields, **metadata)``, which raises
    TypeError because these datasets carry ``ref_id`` / ``source`` in both. It also
    drops per-response status, which we need to tell submitted from draft.
    """
    raw: Dict[str, Any] = {"id": str(record.id) if record.id else None, "status": record.status}
    raw.update(record.metadata.to_dict())
    raw.update(record.fields.to_dict())  # fields win: same values, canonical source

    for response in record.responses:
        key = f"{response.question_name}.responses"
        raw.setdefault(key, []).append(response.value)
        raw.setdefault(f"{key}.users", []).append(str(response.user_id))
        raw.setdefault(f"{key}.status", []).append(
            response.status.value if response.status else None
        )

    for suggestion in record.suggestions:
        key = f"{suggestion.question_name}.suggestion"
        raw[key] = suggestion.value
        raw[f"{key}.score"] = suggestion.score
        raw[f"{key}.agent"] = suggestion.agent

    return raw


NO_GOLD_VALUES = {"", "skip", "n/a", "na", "none", "-", "incorrect", "wrong", "unknown"}
NO_GOLD_PREFIXES = ("[skip]", "skip ", "incorrect")
NOT_FOUND = "Not Found"

# Annotators typed "[SKIP]" (sometimes with a note) when the reference text packs more than
# one publication into a single context, so no single candidate can be right or wrong. Those
# rows are not a judgement about the index and are excluded from the analysis.
MULTI_REF_PREFIX = "[skip]"
MULTI_REF_LABEL = "multi_ref_context"


def clean_gold_id(value: Optional[str]) -> Optional[str]:
    """Annotator free text → a usable ID, or None.

    None covers blanks and the placeholders annotators actually used: "[SKIP]", the same
    with a trailing note ("[SKIP] noisy, several references in this context"), "INCORRECT".
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    lowered = cleaned.lower()
    if lowered in NO_GOLD_VALUES or lowered.startswith(NO_GOLD_PREFIXES):
        return None
    return cleaned or None


def is_multi_ref_context(value: Optional[str]) -> bool:
    """True when the annotator marked the reference as packing several publications."""
    return isinstance(value, str) and value.strip().lower().startswith(MULTI_REF_PREFIX)


def derive_label(row: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse the three questions into one ground-truth label.

    The two label questions are not independent: a "Not Found" candidate answered
    is_match_correct=true means "correctly reported as absent", not "candidate matches".
    Contradictory combinations are labelled anyway and flagged via needs_review.
    """
    is_correct = row.get("is_match_correct")
    no_match = row.get("no_match")
    gold_id = clean_gold_id(row.get("correct_id"))
    candidate_missing = (row.get("matched_id") or NOT_FOUND) == NOT_FOUND

    # A gold ID identical to the candidate is the annotator confirming, not correcting.
    gold_confirms_candidate = bool(gold_id) and gold_id == row.get("matched_id")
    gold_correction = gold_id if not gold_confirms_candidate else None

    if is_multi_ref_context(row.get("correct_id")):
        label, reason = MULTI_REF_LABEL, ""
    elif is_correct is None and no_match is None:
        label, reason = "unannotated", ""
    elif no_match == "true":
        # No record for this reference in this index, whatever the candidate was.
        label = "no_record_in_index"
        reason = (
            "contradiction"
            if (is_correct == "true" and not candidate_missing) or gold_correction
            else ""
        )
    elif is_correct == "true":
        # A record exists and the candidate is it — unless there was no candidate.
        label = "correct_match" if not candidate_missing else "ambiguous"
        reason = "contradiction" if candidate_missing or gold_correction else ""
    else:
        label = "wrong_match"
        reason = "" if gold_id else "gold_missing"  # wrong, but no correct ID given

    return {
        "label": label,
        "gold_id": gold_id,
        "gold_confirms_candidate": gold_confirms_candidate,
        "candidate_missing": candidate_missing,
        "review_reason": reason,
        "needs_review": bool(reason),
    }


def flatten_to_row(raw: Dict[str, Any], index: str) -> Dict[str, Any]:
    """Turn one ``record_to_raw`` dict into an export row.

    Keeps every raw key (fields, metadata, per-user response lists) and adds the
    convenience columns a consumer actually wants: one resolved answer per question.
    """
    row: Dict[str, Any] = {key: _jsonable(value) for key, value in raw.items()}
    row["index"] = index
    row["record_id"] = row.get("id")
    for question in QUESTIONS:
        row[question] = _jsonable(_pick_response(raw, question))
    row["n_responses"] = max((len(raw.get(f"{q}.responses") or []) for q in QUESTIONS), default=0)
    row["annotators"] = sorted(
        {str(user) for q in QUESTIONS for user in (raw.get(f"{q}.responses.users") or [])}
    )
    row.update(derive_label(row))
    return row


def align_schema(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give every row the same keys — one index may lack a question's response column."""
    columns = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in columns} for row in rows]


# ---------------------------------------------------------------------------
# Argilla → rows
# ---------------------------------------------------------------------------
def fetch_dataset(client: Any, name: str, index: str) -> List[Dict[str, Any]]:
    dataset = client.datasets(name=name)
    if dataset is None:
        raise RuntimeError(f"Argilla dataset '{name}' not found")
    rows = [
        flatten_to_row(record_to_raw(record), index)
        for record in dataset.records(with_responses=True, with_suggestions=True)
    ]
    if not rows:
        raise RuntimeError(f"Argilla dataset '{name}' has no records - nothing to export")
    logger.info(
        "%s: %d records, %d with responses",
        name,
        len(rows),
        sum(1 for row in rows if row["n_responses"]),
    )
    return rows


def fetch_splits(api_url: str, api_token: str) -> Dict[str, List[Dict[str, Any]]]:
    """Both splits, every index, in the published row order."""
    import argilla as rg

    client = rg.Argilla(api_key=api_token, api_url=api_url)
    splits = {
        split: [
            row
            for index in INDEXES
            for row in fetch_dataset(client, template.format(index=index), index)
        ]
        for split, template in SPLIT_DATASETS.items()
    }
    for row in fetch_dataset(client, OPENCITATIONS_DATASET, "opencitations"):
        split = "dasi_ebda" if row["source"] in DASI_EBDA_SOURCES else "annotations"
        splits[split].append(row)
    return {split: align_schema(rows) for split, rows in splits.items()}


def load_local(path: Path) -> List[Dict[str, Any]]:
    """Re-read a previous export and re-derive labels — no Argilla, no credentials."""
    raw_rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not raw_rows:
        raise RuntimeError(f"{path} is empty")
    return align_schema([flatten_to_row(row, row.get("index") or "unknown") for row in raw_rows])


def log_summary(rows: List[Dict[str, Any]]) -> None:
    """Loud per-index breakdown — nulls in this export are meaningful, not silent loss."""
    from collections import Counter

    for index in sorted({row["index"] for row in rows}):
        subset = [row for row in rows if row["index"] == index]
        labels = Counter(row["label"] for row in subset)
        logger.info(
            "%-14s %4d rows | %s | needs_review=%d | gold_id=%d",
            index,
            len(subset),
            " ".join(f"{name}={count}" for name, count in sorted(labels.items())),
            sum(1 for row in subset if row["needs_review"]),
            sum(1 for row in subset if row["gold_id"]),
        )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_raw(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = RAW / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        logger.info("wrote %s (%d rows)", path, len(rows))


def read_raw(split: str) -> List[Dict[str, Any]]:
    text = (RAW / f"{split}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def annotation_table(rows: List[Dict[str, Any]], published: Any) -> Any:
    """Raw rows as parquet with the published column types."""
    import pyarrow as pa

    schema = published.remove_metadata()
    if "manual_fix" in schema.names:  # corrections belong to the gold dataset
        schema = schema.remove(schema.get_field_index("manual_fix"))
    if "search_method" not in schema.names:
        schema = schema.append(pa.field("search_method", pa.string()))
    unknown = {key for row in rows for key in row} - set(schema.names)
    if unknown:
        raise ValueError(f"Columns missing from the published schema: {sorted(unknown)}")
    return pa.Table.from_pylist(
        [{name: row.get(name) for name in schema.names} for row in rows], schema=schema
    )


def publish(repo: str, push: bool) -> None:
    """Raw rows of both splits, their analysis tables and the README, in one commit."""
    import pyarrow.parquet as pq
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    operations, lines = [], []
    for split in SPLIT_DATASETS:
        path_in_repo = f"data/{split}-00000-of-00001.parquet"
        old = pq.read_table(hf_hub_download(repo, path_in_repo, repo_type="dataset"))
        table = annotation_table(read_raw(split), old.schema)
        # Same tasks in the same order: published rows may only be extended.
        keys = list(zip(table["ref_id"].to_pylist(), table["index"].to_pylist()))
        old_keys = list(zip(old["ref_id"].to_pylist(), old["index"].to_pylist()))
        if keys[: len(old_keys)] != old_keys:
            raise ValueError(f"{split}: raw export does not extend the published rows")
        parquet = io.BytesIO()
        pq.write_table(table, parquet)
        counts: Dict[str, int] = {}
        for index in table["index"].to_pylist():
            counts[index] = counts.get(index, 0) + 1
        lines.append(
            f"- `{split}`: {table.num_rows} rows, "
            f"{len(set(table['ref_id'].to_pylist()))} references; per index {counts}."
        )
        operations.append(CommitOperationAdd(path_in_repo, parquet.getvalue()))
        for name in ANALYSIS_FILES:
            csv = ANALYSIS_DIRS[split] / f"{name}.csv"
            if csv.exists():
                operations.append(CommitOperationAdd(f"analysis/{split}/{name}.csv", str(csv)))
        logger.info("%s: %d -> %d rows %s", split, old.num_rows, table.num_rows, counts)
    readme = README.format(splits="\n".join(lines))
    operations.append(CommitOperationAdd("README.md", readme.encode()))
    logger.info("%s: %d files", repo, len(operations))
    if not push:
        logger.info("dry run; pass --push to publish")
        return
    commit = HfApi(token=os.environ["HF_TOKEN"]).create_commit(
        repo,
        operations,
        repo_type="dataset",
        commit_message="publish raw Argilla annotations",
    )
    logger.info("pushed %s", commit.oid)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-fetch", action="store_true", help="reuse the files in raw/")
    parser.add_argument("--push", action="store_true", help="publish to Hugging Face")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--argilla-url", default=ARGILLA_API_URL)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.no_fetch:
        splits = fetch_splits(args.argilla_url, ARGILLA_API_TOKEN)
        for rows in splits.values():
            log_summary(rows)
        write_raw(splits)
    publish(args.hf_repo, args.push)
    return 0


if __name__ == "__main__":
    sys.exit(main())
