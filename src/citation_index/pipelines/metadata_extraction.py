"""PDF -> bibliographic metadata of the document itself (the citing work).

GROBID reads it from the fulltext TEI header (``sourceDesc/biblStruct``), sharing
the cached TEI with the reference endpoints. The LLM reads the first pages.
Both return one ``Reference`` dump, the same shape as a parsed reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pymupdf
from lxml import etree

from citation_index.core.models import Reference
from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.llm.client import LLMClient
from citation_index.llm.grobid_client import GrobidClient
from citation_index.pipelines.grobid_references import fetch_fulltext_tei
from citation_index.pipelines.reference_parsing import parse_reference_strings

logger = logging.getLogger(__name__)

_NS = {"t": "http://www.tei-c.org/ns/1.0"}
METADATA_PROMPT = "prompts/prompts.yaml:metadata.default"
# Title page + one more covers title, authors, journal/book and DOI in practice
LLM_MAX_PAGES = 2
LLM_MAX_CHARS = 12000


class MetadataExtractionError(RuntimeError):
    """Raised when no document metadata could be extracted."""


def _drop_grobid_hash(ref: Reference) -> Reference:
    # GROBID's md5 idno is a hash of the PDF, not a bibliographic identifier
    ids = [i for i in ref.identifiers or [] if i.scheme != "md5"]
    return ref.model_copy(update={"identifiers": ids})


def metadata_from_tei(tei_xml: str) -> Reference:
    """Parse the TEI header biblStruct into a Reference. Raises if it is empty."""
    root = etree.fromstring(tei_xml.encode("utf-8"))
    header = root.xpath("//t:teiHeader//t:sourceDesc/t:biblStruct", namespaces=_NS)
    parsed = (
        TeiBiblParser().to_references(header[0], raise_empty_error=False)
        if header
        else []
    )
    if not parsed:
        raise MetadataExtractionError("GROBID TEI header has no document metadata")
    return _drop_grobid_hash(parsed[0])


def extract_metadata_grobid(
    pdf_path: str | Path, client: GrobidClient
) -> Dict[str, Any]:
    reference = metadata_from_tei(fetch_fulltext_tei(pdf_path, client))
    return {"metadata": reference.model_dump(), "method": "grobid"}


def first_pages_text(pdf_path: str | Path, max_pages: int = LLM_MAX_PAGES) -> str:
    with pymupdf.open(pdf_path) as doc:
        text = "".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
    return text[:LLM_MAX_CHARS]


def extract_metadata_llm(pdf_path: str | Path, llm_client: LLMClient) -> Dict[str, Any]:
    text = first_pages_text(pdf_path)
    if not text.strip():
        raise MetadataExtractionError("PDF has no text layer on its first pages")
    candidates = list(
        parse_reference_strings([text], llm_client, prompt_name=METADATA_PROMPT)
    )
    # The prompt asks for exactly one object; parse_reference_strings raises on none
    if len(candidates) > 1:
        logger.warning(
            "LLM returned %d metadata objects for %s; using the first",
            len(candidates),
            pdf_path,
        )
    reference = candidates[0]
    return {"metadata": reference.model_dump(), "method": "llm"}
