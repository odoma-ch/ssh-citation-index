"""Live regression test for reference extraction and parsing against the real LLM.

These tests call the configured vLLM endpoint directly through the pipeline functions the
RQ workers use — no API, no queue. They exist because the damaging regressions in this
pipeline are field-level and invisible to mocked tests: a run can return one record per
input string with every title null (observed: 0/63 titles on a footnote-style PDF, 4/7 on
the API guide's fixture) while every count-based assertion passes.

Opt in explicitly, they take minutes and cost real inference:

    CITATION_INDEX_LIVE_TESTS=1 pytest tests/test_reference_pipeline_live.py -v -s

Thresholds are floors well below observed values, not targets — reference extraction runs
at temperature 0.3 and returns a slightly different set of strings each run. Tighten a
floor only after checking a few runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from citation_index.config import settings
from citation_index.core.models import Reference
from citation_index.llm.client import LLMClient
from citation_index.pipelines.reference_extraction import extract_text_references
from citation_index.pipelines.reference_parsing import parse_reference_strings


DATA_DIR = Path(__file__).parent / "data"
NOTEBOOK_PDF = Path(__file__).parent.parent / "benchmarks" / "excite" / "all_pdfs" / "44404.pdf"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("CITATION_INDEX_LIVE_TESTS") != "1",
        reason="set CITATION_INDEX_LIVE_TESTS=1 to run tests that call the real LLM",
    ),
]


@dataclass(frozen=True)
class Document:
    """A fixture document and the floors its results must clear."""

    name: str
    path: Path
    min_reference_strings: int
    min_records: int
    min_titled_share: float
    min_people_share: float
    expected_titles: Tuple[str, ...] = field(default=())

    @property
    def is_pdf(self) -> bool:
        return self.path.suffix.lower() == ".pdf"


# Floors are deliberately loose. Run-to-run spread on the same document is wide — the reg
# PDF has produced both 64 records with 64/64 titled and 40 records with 29/40 titled — so
# these guard against the pipeline being *broken* (0 titles, 0 authors), not against
# quality drift. Raise one only after several runs show headroom.
DOCUMENTS = [
    Document(
        name="reg_footnote_pdf",
        path=DATA_DIR / "reg_0035-2039_1989_num_102_485_2445.pdf",
        min_reference_strings=30,
        min_records=30,
        min_titled_share=0.6,
        min_people_share=0.4,
        expected_titles=("Sophoclean Tragedy",),
    ),
    Document(
        name="holanda_markdown",
        path=DATA_DIR / "Holanda-Brasliadailyinvention-1989.md",
        min_reference_strings=5,
        min_records=4,
        min_titled_share=0.8,
        min_people_share=0.4,
        expected_titles=("The Social Logic of Space",),
    ),
    Document(
        # Four of this document's references are bare periodical names with no author
        # ("Caretas", "El Comercio"), hence the low people floor.
        name="notebook_pdf",
        path=NOTEBOOK_PDF,
        min_reference_strings=4,
        min_records=4,
        min_titled_share=0.8,
        min_people_share=0.25,
        expected_titles=("Die Koka- und Kokainwirtschaft Perus",),
    ),
]

# Extraction and parsing are expensive; run each document once per session.
_CACHE: Dict[str, Tuple[List[str], List[Reference]]] = {}


def _client(timeout: float, first_token_timeout: float) -> LLMClient:
    return LLMClient(
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=timeout,
        max_retries=settings.llm_max_retries,
        first_token_timeout=first_token_timeout,
        enable_thinking=settings.llm_enable_thinking,
    )


def _document_text(document: Document) -> str:
    if not document.is_pdf:
        return document.path.read_text(encoding="utf-8")

    pytest.importorskip("pymupdf", reason="PyMuPDF is required to extract text from PDF fixtures")
    from citation_index.pipelines.text_extraction import extract_text

    result = extract_text(str(document.path), extractor="pymupdf", markdown=True)
    assert len(result.text) > 1_000, f"{document.name}: extractor returned {len(result.text)} chars"
    return result.text


def _run_pipeline(document: Document) -> Tuple[List[str], List[Reference]]:
    """Extract reference strings, then parse them into Reference records."""
    if document.name in _CACHE:
        return _CACHE[document.name]

    assert document.path.is_file(), f"missing fixture: {document.path}"

    reference_strings = extract_text_references(
        text=_document_text(document),
        llm_client=_client(
            settings.llm_timeout, settings.llm_first_token_timeout_reference_extraction
        ),
        temperature=0.3,
    )
    records = list(
        parse_reference_strings(
            reference_lines=reference_strings,
            llm_client=_client(
                settings.llm_timeout_reference_parsing,
                settings.llm_first_token_timeout_reference_parsing,
            ),
            temperature=0.0,
        )
    )

    titled = sum(1 for r in records if r.full_title or r.journal_title)
    with_people = sum(1 for r in records if r.authors or r.editors)
    print(
        f"\n[{document.name}] {len(reference_strings)} reference strings -> "
        f"{len(records)} records | titled {titled}/{len(records)} "
        f"(floor {document.min_titled_share:.0%}) | with author/editor "
        f"{with_people}/{len(records)} (floor {document.min_people_share:.0%})"
    )
    _CACHE[document.name] = (reference_strings, records)
    return _CACHE[document.name]


def _titles(records: List[Reference]) -> List[str]:
    return [r.full_title or r.journal_title or "" for r in records]


@pytest.fixture(scope="module", autouse=True)
def require_llm_endpoint():
    if not settings.llm_endpoint or not settings.llm_api_key:
        pytest.skip("LLM_ENDPOINT and LLM_API_KEY must be configured for live tests")


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d.name)
def test_reference_extraction_returns_citation_strings(document: Document):
    reference_strings, _ = _run_pipeline(document)

    assert len(reference_strings) >= document.min_reference_strings, (
        f"{document.name}: extracted {len(reference_strings)} reference strings, "
        f"expected at least {document.min_reference_strings}"
    )
    assert all(isinstance(s, str) and s.strip() for s in reference_strings)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d.name)
def test_parsed_references_carry_titles(document: Document):
    """The regression this guards: records returned with every title null."""
    _, records = _run_pipeline(document)

    assert len(records) >= document.min_records, (
        f"{document.name}: parsed {len(records)} records, "
        f"expected at least {document.min_records}"
    )

    titles = _titles(records)
    titled_share = sum(1 for title in titles if title) / len(records)
    assert titled_share >= document.min_titled_share, (
        f"{document.name}: only {titled_share:.0%} of records carry a title, expected "
        f"at least {document.min_titled_share:.0%}"
    )


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d.name)
def test_parsed_references_match_known_titles(document: Document):
    """Spot-check real titles from the source documents, not just non-null strings."""
    _, records = _run_pipeline(document)
    titles = _titles(records)

    for expected in document.expected_titles:
        assert any(expected in title for title in titles), (
            f"{document.name}: expected a title containing {expected!r}; got {titles}"
        )


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d.name)
def test_parsed_references_carry_authors_or_editors(document: Document):
    """Requiring only the titles in the schema made the model drop every author."""
    _, records = _run_pipeline(document)

    with_people = sum(1 for r in records if r.authors or r.editors)
    people_share = with_people / len(records)
    assert people_share >= document.min_people_share, (
        f"{document.name}: only {with_people}/{len(records)} records name an author or "
        f"editor ({people_share:.0%}), expected at least {document.min_people_share:.0%}"
    )
