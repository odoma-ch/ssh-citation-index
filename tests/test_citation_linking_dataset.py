"""Label and candidate cleaning of the citation-linking release build."""

from benchmarks.citation_linking.dataset.build_annotation_dataset import (
    doi_evidence,
    final_target,
)
from benchmarks.citation_linking.dataset.candidate_pool import deduplicate


def openalex(identifier: str, title: str) -> dict:
    return {
        "candidate_id": identifier,
        "metadata": {"title": title, "publication_year": 1990, "type": "book"},
    }


def test_deduplicate_keeps_gold_and_renumbers_ranks():
    found = {
        "ref_id": "r",
        "index": "openalex",
        "candidates": [
            {**openalex("W1", "Same book"), "rank": 1},
            {**openalex("W2", "Other book"), "rank": 2},
            {**openalex("W3", "Same book"), "rank": 3},
        ],
    }
    row = {"ref_id": "r", "index": "openalex", "gold_id": "W3"}
    (cleaned,), removed = deduplicate([row], [found])
    assert [(c["candidate_id"], c["rank"]) for c in cleaned["candidates"]] == [
        ("W3", 1),
        ("W2", 2),
    ]
    assert removed == [
        {"ref_id": "r", "index": "openalex", "removed": "W1", "duplicate_of": "W3"}
    ]
    assert len(found["candidates"]) == 3  # input untouched


def test_deduplicate_without_gold_keeps_higher_rank():
    found = {
        "candidates": [
            {**openalex("W9", "Same"), "rank": 2},
            {**openalex("W8", "Same"), "rank": 1},
        ]
    }
    row = {"ref_id": "r", "index": "openalex", "gold_id": None}
    (cleaned,), removed = deduplicate([row], [found])
    assert [c["candidate_id"] for c in cleaned["candidates"]] == ["W8"]
    assert removed[0]["removed"] == "W9"


def annotation(index: str, matched: str, doi: str) -> dict:
    return {
        "index": index,
        "label": "correct_match",
        "matched_id": matched,
        "matched_doi": doi,
    }


def test_overridden_match_no_longer_vouches_for_its_doi():
    group = [
        annotation("openalex", "W1", "10.5860/choice.29-0573"),  # a review
        annotation(
            "matilda", "d6313d68-4fc7-4307-9b92-48b25363cdda", "10.1515/9780691215983"
        ),
    ]
    fix = {"gold_id": {"openalex": None}}
    assert final_target(group[0], fix) is None
    assert doi_evidence(group, fix) == [
        {"index": "matilda", "doi": "10.1515/9780691215983"}
    ]
    assert len(doi_evidence(group, {})) == 2
