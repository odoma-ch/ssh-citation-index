"""Label checks of the citation-linking release, for a person to review.

Helper of ``build_annotation_dataset.py``. It only flags tasks; it never changes or
skips one. Three checks, each from the frozen candidate fields the model reads:

- ``possible_review_gold``: a book is cited and the gold is a later article or review
  with the same title, or the gold's own title or type says "review".
- ``none_with_strong_match``: the gold is none, yet a candidate has the same title,
  first author and year (within one).
- ``cross_index_doi_disagreement``: the gold records of one reference in different
  indexes carry different DOIs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from citation_index.core.models import Reference
from citation_index.pipelines.citation_linking import _candidate_doi
from citation_index.utils.reference_matching import (
    custom_match,
    reference_match_fields,
)

REVIEW_WORDS = re.compile(
    r"\b(review|reviews|rezension|compte[- ]rendu|recensione|reseña)\b|\(review\)", re.I
)
ARTICLE_TYPES = ("article", "review")


def _reference(row: dict[str, Any]) -> tuple[Reference, dict[str, Any]]:
    parsed = row.get("parsed_reference") or {}
    reference = Reference.model_validate(parsed)
    if not (reference.full_title or "").strip():
        reference.full_title = row["reference"].strip()
    return reference, parsed


def _text(value: Any) -> str | None:
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value if item)
    return str(value) if value not in (None, "") else None


def _record(index: str, metadata: dict[str, Any]) -> Reference:
    """The candidate as the model reads it (frozen fields, no live label lookups)."""
    from benchmarks.citation_linking.experiments.linking_eval import candidate_fields

    fields = candidate_fields(index, metadata)
    authors = fields.get("authors") or []
    year = fields.get("publication_year")
    return Reference(
        full_title=_text(fields.get("full_title")),
        authors=[
            str(a) for a in ([authors] if isinstance(authors, str) else authors) if a
        ]
        or None,
        publication_year=year if isinstance(year, int) else None,
        publication_date_raw=_text(fields.get("publication_date_raw")),
        journal_title=_text(fields.get("journal_title")),
        ref_type=_text(fields.get("ref_type")),
    )


def _doi(value: str | None) -> str:
    return (value or "").lower().replace("https://doi.org/", "").strip()


def flag(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_dois: dict[str, dict[str, str]] = defaultdict(dict)
    for row, found in zip(rows, candidates):
        index, gold = row["index"], row["gold_id"]
        reference, parsed = _reference(row)
        for candidate in found["candidates"]:
            metadata = candidate["metadata"]
            try:
                record = _record(index, metadata)
            except Exception:  # noqa: BLE001 - a malformed record is itself a flag
                report["unreadable_candidate"].append(
                    {
                        "ref_id": row["ref_id"],
                        "index": index,
                        "candidate": candidate["candidate_id"],
                    }
                )
                continue
            fields = reference_match_fields(record)
            _, detail = custom_match(reference, fields)
            entry = {
                "ref_id": row["ref_id"],
                "index": index,
                "candidate": candidate["candidate_id"],
                "reference": row["reference"][:200],
                "candidate_record": f"{fields['title']} | {fields['first_author']} | {fields['year']}"[
                    :200
                ],
            }
            if candidate["candidate_id"] == gold:
                doi = _doi(_candidate_doi(index, metadata))
                if doi:
                    gold_dois[row["ref_id"]][index] = doi
                kind = str(record.ref_type or metadata.get("type") or "").lower()
                book = not parsed.get("journal_title") and (
                    parsed.get("publisher") or parsed.get("publication_place")
                )
                later = (
                    fields["year"]
                    and reference.publication_year
                    and fields["year"] > reference.publication_year
                )
                if REVIEW_WORDS.search(f"{fields['title']} {kind}") or (
                    book
                    and later
                    and any(t in kind for t in ARTICLE_TYPES)
                    and detail["title_similarity"] >= 85
                ):
                    report["possible_review_gold"].append({**entry, "type": kind})
            elif (
                gold is None
                and detail["title_similarity"] >= 95
                and detail["author_similarity"] >= 80
                and detail["year_diff"] is not None
                and detail["year_diff"] <= 1
            ):
                report["none_with_strong_match"].append(entry)
    for ref_id, dois in sorted(gold_dois.items()):
        if len(set(dois.values())) > 1:
            report["cross_index_doi_disagreement"].append(
                {"ref_id": ref_id, "dois": dois}
            )
    return {
        key: report.get(key, [])
        for key in (
            "possible_review_gold",
            "none_with_strong_match",
            "cross_index_doi_disagreement",
            "unreadable_candidate",
        )
    }
