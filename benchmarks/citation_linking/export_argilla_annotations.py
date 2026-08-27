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


NO_GOLD_VALUES = {"", "[skip]", "skip", "n/a", "na", "none", "-"}
NOT_FOUND = "Not Found"


def clean_gold_id(value: Optional[str]) -> Optional[str]:
    """Annotator free text → a usable ID, or None (blank / '[SKIP]' / placeholder)."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None if cleaned.lower() not in NO_GOLD_VALUES else None


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

    if is_correct is None and no_match is None:
        label, needs_review = "unannotated", False
    elif no_match == "true":
        # No record for this reference in this index, whatever the candidate was.
        label = "no_record_in_index"
        needs_review = bool(gold_id) or (is_correct == "true" and not candidate_missing)
    elif is_correct == "true":
        # A record exists and the candidate is it — unless there was no candidate.
        label = "correct_match" if not candidate_missing else "ambiguous"
        needs_review = candidate_missing or bool(gold_id)
    else:
        label = "wrong_match"
        needs_review = not gold_id  # wrong, but no correct ID given

    return {
        "label": label,
        "gold_id": gold_id,
        "candidate_missing": candidate_missing,
        "needs_review": needs_review,
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


def self_check() -> None:
    # ref_id / source live in BOTH fields and metadata (see notebook cell 14) — the
    # collision that makes argilla's own flatten=True raise TypeError.
    record = _FakeRecord(
        rec_id="rec-1",
        fields={
            "ref_id": "brill_1",
            "source": "brill",
            "original_ref_string": "Foo (1999). Bar.",
            "matched_id": "W999",
        },
        metadata={"ref_id": "brill_1", "source": "brill", "index": "openalex"},
        responses=[
            _FakeResponse("is_match_correct", "false", "u-draft", "draft"),
            _FakeResponse("is_match_correct", "true", "u-sub", "submitted"),
            _FakeResponse("correct_id", "W123", "u-sub", "submitted"),
            _FakeResponse("no_match", "false", "u-sub", "submitted"),
        ],
        suggestions=[_FakeSuggestion("is_match_correct", "true", score=0.9, agent="similarity")],
        status="completed",
    )
    raw = record_to_raw(record)
    assert raw["ref_id"] == "brill_1" and raw["source"] == "brill"
    assert raw["is_match_correct.responses"] == ["false", "true"]
    assert raw["is_match_correct.responses.status"] == ["draft", "submitted"]
    assert raw["is_match_correct.suggestion"] == "true"

    row = flatten_to_row(raw, "openalex")
    assert row["is_match_correct"] == "true", row["is_match_correct"]  # submitted beats draft
    assert row["correct_id"] == "W123"
    assert row["no_match"] == "false"
    assert row["index"] == "openalex" and row["record_id"] == "rec-1"
    assert row["n_responses"] == 2  # max responses on any single question
    assert row["annotators"] == ["u-draft", "u-sub"]

    draft_only = flatten_to_row(
        record_to_raw(
            _FakeRecord(
                "rec-2",
                {"ref_id": "cex_1"},
                {"index": "wikidata"},
                responses=[_FakeResponse("is_match_correct", "false", "u-1", "draft")],
            )
        ),
        "wikidata",
    )
    assert draft_only["is_match_correct"] == "false"  # draft used when nothing submitted

    empty = flatten_to_row(record_to_raw(_FakeRecord("rec-3", {"ref_id": "cex_9"}, {})), "matilda")
    assert empty["n_responses"] == 0 and empty["annotators"] == []
    assert all(empty[q] is None for q in QUESTIONS)

    # Derived labels: the three questions are not independent (see derive_label).
    assert row["label"] == "correct_match" and row["gold_id"] == "W123"
    assert row["needs_review"] is True  # says correct yet supplied a correct_id

    def _label(matched_id, is_correct, no_match, correct_id=None):
        raw = record_to_raw(
            _FakeRecord(
                "r",
                {"matched_id": matched_id},
                {},
                responses=[
                    _FakeResponse(name, value, "u", "submitted")
                    for name, value in (
                        ("is_match_correct", is_correct),
                        ("no_match", no_match),
                        ("correct_id", correct_id),
                    )
                    if value is not None
                ],
            )
        )
        return flatten_to_row(raw, "openalex")

    absent = _label("Not Found", "true", "true")
    assert absent["label"] == "no_record_in_index" and absent["needs_review"] is False
    good = _label("W1", "true", "false")
    assert good["label"] == "correct_match" and good["needs_review"] is False
    fp = _label("W1", "false", "true")
    assert fp["label"] == "no_record_in_index" and fp["needs_review"] is False
    missed = _label("Not Found", "false", "false", "W9")
    assert missed["label"] == "wrong_match" and missed["gold_id"] == "W9"
    unsourced = _label("W1", "false", "false")
    assert unsourced["label"] == "wrong_match" and unsourced["needs_review"] is True
    contradiction = _label("W1", "true", "true")
    assert contradiction["label"] == "no_record_in_index" and contradiction["needs_review"] is True
    assert _label("Not Found", "true", "false")["label"] == "ambiguous"
    assert clean_gold_id("  [SKIP] ") is None and clean_gold_id(" W7 ") == "W7"
    assert empty["label"] == "unannotated"

    aligned = align_schema([row, draft_only, empty])
    assert len({tuple(sorted(r)) for r in aligned}) == 1, "schema not aligned"
    assert json.dumps(aligned)  # every value JSON-serialisable

    import uuid

    assert isinstance(_jsonable(uuid.uuid4()), str)
    print("✓ self-check passed")


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
    parser.add_argument("--self-check", action="store_true", help="run logic asserts and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.self_check:
        self_check()
        return 0

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
