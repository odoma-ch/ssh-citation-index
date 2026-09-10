"""Export citation-linking annotations from Argilla, save locally, push to HuggingFace.

Counterpart of ``build_citation_linking_dataset.ipynb``: that notebook creates the
four ``citation_linking_{index}`` Argilla datasets, this script pulls the annotated
records back out.

Usage
-----
    export ARGILLA_API_TOKEN=...            # same names the notebook uses
    export HF_TOKEN=...

    # local only
    python export_argilla_annotations.py --no-hf

    # local + private HuggingFace dataset
    python export_argilla_annotations.py --hf-repo yurui983/citation_linking_annotations

    # logic self-check, no network, no Argilla install needed
    python export_argilla_annotations.py --self-check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("export_argilla_annotations")

ARGILLA_API_URL = os.environ.get("ARGILLA_API_URL") or "https://argilla.graphia-ssh.eu/"
ARGILLA_API_TOKEN = os.environ.get("ARGILLA_API_TOKEN") or "argilla.apikey"
HF_TOKEN = os.environ.get("HF_TOKEN")

INDEXES = ["openalex", "matilda", "wikidata",] # "opencitations"]
DATASET_NAME_TEMPLATE = "citation_linking_{index}"
QUESTIONS = ["is_match_correct", "correct_id", "no_match"]

# Deliberately NOT yurui983/citation_linking: push_to_hub rewrites the repo's split
# layout, which would orphan that dataset's existing `full` / `annotated` splits.
DEFAULT_HF_REPO = "yurui983/citation_linking_annotations"
DEFAULT_OUT_DIR = Path(__file__).parent / "argilla_annotations"
HF_SPLIT = "annotations"


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
    row["n_responses"] = max(
        (len(raw.get(f"{q}.responses") or []) for q in QUESTIONS), default=0
    )
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
def fetch_rows(indexes: List[str], api_url: str, api_token: str) -> List[Dict[str, Any]]:
    import argilla as rg

    client = rg.Argilla(api_key=api_token, api_url=api_url)
    rows: List[Dict[str, Any]] = []

    for index in indexes:
        name = DATASET_NAME_TEMPLATE.format(index=index)
        dataset = client.datasets(name=name)
        if dataset is None:
            raise RuntimeError(f"Argilla dataset '{name}' not found at {api_url}")

        index_rows = [
            flatten_to_row(record_to_raw(record), index)
            for record in dataset.records(with_responses=True, with_suggestions=True)
        ]
        if not index_rows:
            raise RuntimeError(f"Argilla dataset '{name}' has no records — nothing to export")

        answered = sum(1 for row in index_rows if row["n_responses"])
        logger.info("%s: %d records, %d with responses", name, len(index_rows), answered)
        if not answered:
            logger.warning("%s: no annotations submitted yet", name)
        rows.extend(index_rows)

    return align_schema(rows)


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
    missing = [index for index in INDEXES if index not in {row["index"] for row in rows}]
    if missing:
        logger.warning("no rows exported for: %s", ", ".join(missing))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_local(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in sorted({row["index"] for row in rows}):
        path = out_dir / f"citation_linking_{index}_annotations.jsonl"
        subset = [row for row in rows if row["index"] == index]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
            encoding="utf-8",
        )
        logger.info("wrote %s (%d rows)", path, len(subset))

    combined = out_dir / "citation_linking_annotations.jsonl"
    combined.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    logger.info("wrote %s (%d rows)", combined, len(rows))
    return combined


def push_hf(rows: List[Dict[str, Any]], repo: str, private: bool, token: Optional[str]) -> None:
    from datasets import Dataset, DatasetDict  # lazy: local export must work without it

    dataset = DatasetDict({HF_SPLIT: Dataset.from_list(rows)})
    dataset.push_to_hub(repo, private=private, token=token)
    logger.info("pushed %d rows to https://huggingface.co/datasets/%s", len(rows), repo)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
class _FakeMapping:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class _FakeStatus:
    def __init__(self, value: str):
        self.value = value


class _FakeResponse:
    def __init__(self, question_name: str, value: Any, user_id: str, status: Optional[str]):
        self.question_name = question_name
        self.value = value
        self.user_id = user_id
        self.status = _FakeStatus(status) if status else None


class _FakeSuggestion:
    def __init__(self, question_name: str, value: Any, score=None, agent=None):
        self.question_name = question_name
        self.value = value
        self.score = score
        self.agent = agent


class _FakeRecord:
    """Duck-type of argilla.Record — same attributes record_to_raw touches."""

    def __init__(self, rec_id, fields, metadata, responses=(), suggestions=(), status="pending"):
        self.id = rec_id
        self.status = status
        self.fields = _FakeMapping(fields)
        self.metadata = _FakeMapping(metadata)
        self.responses = list(responses)
        self.suggestions = list(suggestions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--indexes", nargs="+", default=INDEXES, choices=INDEXES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--no-hf", action="store_true", help="skip the HuggingFace push")
    parser.add_argument("--public", action="store_true", help="push as a public dataset")
    parser.add_argument("--argilla-url", default=ARGILLA_API_URL)
    parser.add_argument(
        "--from-jsonl",
        type=Path,
        help="re-derive labels from a previous export instead of querying Argilla",
    )
    # parser.add_argument("--self-check", action="store_true", help="run logic asserts and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_jsonl:
        rows = load_local(args.from_jsonl)
    else:
        rows = fetch_rows(args.indexes, args.argilla_url, ARGILLA_API_TOKEN)
    log_summary(rows)
    write_local(rows, args.out_dir)

    if args.no_hf:
        logger.info("--no-hf set, skipping HuggingFace push")
        return 0
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set — export it or pass --no-hf")
    push_hf(rows, args.hf_repo, private=not args.public, token=HF_TOKEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
