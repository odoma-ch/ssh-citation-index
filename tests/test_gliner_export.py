"""GLiNER training export: tasks that must not become ``none`` targets."""

from benchmarks.citation_linking.experiments.gliner_data import gold_doi_in_pool


def row(gold_doi, candidate_doi):
    return {
        "index": "openalex",
        "gold_doi": gold_doi,
        "candidates": [
            {"metadata": {"id": "W1", "doi": None}},
            {"metadata": {"id": "W2", "doi": candidate_doi}},
        ],
    }


def test_flags_pool_holding_the_gold_doi():
    assert gold_doi_in_pool(
        row("10.1037/12819-002", "https://doi.org/10.1037/12819-002")
    )
    assert gold_doi_in_pool(row("10.1037/ABC", "https://doi.org/10.1037/abc"))


def test_keeps_none_without_gold_doi_match():
    assert not gold_doi_in_pool(row("10.1037/12819-002", "https://doi.org/10.1/other"))
    assert not gold_doi_in_pool(row(None, "https://doi.org/10.1/other"))
    assert not gold_doi_in_pool(row(float("nan"), None))  # pandas missing value
