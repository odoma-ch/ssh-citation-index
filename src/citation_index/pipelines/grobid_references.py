"""PDF -> references (raw, parsed, and citing contexts) using GROBID fulltext TEI.

Contexts are the sentences that cite each reference: in GROBID TEI an in-text
marker is ``<ref type="bibr" target="#bN">`` and ``N`` is the ``xml:id`` of a
``<biblStruct>`` in the back-matter ``<listBibl>``. Each context is shaped as
``{"section", "citation"}`` so it can be posted as-is to ``/classify/citations``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from lxml import etree

from citation_index.core.parsers.tei_bibl_parser import TeiBiblParser
from citation_index.llm.grobid_client import GrobidClient

logger = logging.getLogger(__name__)

_NS = {"t": "http://www.tei-c.org/ns/1.0"}
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _text(element: etree._Element) -> str:
    # GROBID sets xml:space="preserve", so whitespace has to be normalised here
    return " ".join("".join(element.itertext()).split())


def _citing_contexts(root: etree._Element) -> Dict[str, List[Dict[str, str]]]:
    """Map bibliography xml:id -> citing sentences (deduplicated, in document order)."""
    contexts: Dict[str, List[Dict[str, str]]] = {}
    markers = root.xpath(
        "//t:text//t:ref[@type='bibr'][not(ancestor::t:listBibl)]", namespaces=_NS
    )
    for marker in markers:
        # Union comes back in document order (outermost first), so [-1] is the <s>
        # when sentences were segmented, else the enclosing <p>
        container = marker.xpath("ancestor::t:s[1] | ancestor::t:p[1]", namespaces=_NS)
        if not container:
            continue
        sentence = _text(container[-1])
        if not sentence:
            continue
        head = marker.xpath("ancestor::t:div[1]/t:head[1]", namespaces=_NS)
        section = _text(head[0]) if head else ""

        # Untargeted markers cannot be tied to a reference; multi-target ones list ids
        for ref_id in (marker.get("target") or "").replace("#", "").split():
            entries = contexts.setdefault(ref_id, [])
            if any(e["citation"] == sentence for e in entries):
                continue
            entries.append(
                {
                    "section": section,
                    "citation": sentence,
                    "marker": _text(marker),
                    "ref_id": ref_id,
                }
            )
    return contexts


def parse_grobid_fulltext(
    tei_xml: str, include_parsed: bool = False, include_context: bool = False
) -> Dict[str, Any]:
    """Turn a processFulltextDocument TEI string into the job output.

    ``references`` holds raw strings; ``parsed_references`` and ``contexts`` are
    added on request and stay index-aligned with it. Raises on invalid TEI.
    """
    root = etree.fromstring(tei_xml.encode("utf-8"))
    # back//listBibl only: the header's biblStruct describes the paper itself
    bibl_structs = root.xpath("//t:back//t:listBibl/t:biblStruct", namespaces=_NS)
    contexts = _citing_contexts(root) if include_context else {}
    tei_parser = TeiBiblParser()

    raw_refs: List[str] = []
    parsed_refs: List[Any] = []
    ref_contexts: List[List[Dict[str, str]]] = []
    for bibl in bibl_structs:
        note = bibl.xpath("t:note[@type='raw_reference']", namespaces=_NS)
        raw = _text(note[0]) if note else _text(bibl)
        parsed = tei_parser.to_references(bibl, raise_empty_error=False)
        if not raw and not parsed:
            continue  # empty entry; skipped here so ids cannot shift the alignment
        raw_refs.append(raw)
        parsed_refs.append(parsed[0].model_dump() if parsed else None)
        ref_contexts.append(contexts.get(bibl.get(_XML_ID, ""), []))

    output: Dict[str, Any] = {
        "references": raw_refs,
        "method": "grobid",
        "count": len(raw_refs),
    }
    if include_parsed:
        output["parsed_references"] = parsed_refs
    if include_context:
        output["contexts"] = ref_contexts
    return output


def extract_references_grobid(
    pdf_path: str | Path,
    client: GrobidClient,
    include_parsed: bool = False,
    include_context: bool = False,
) -> Dict[str, Any]:
    """Run GROBID fulltext on a PDF and extract references (+ parsed, + contexts)."""
    tei_xml = client.process_full_text(
        pdf_path,
        # consolidation calls CrossRef/Glutton per reference; keep it off
        consolidate_references=False,
        include_raw_references=True,
        segment_sentences=include_context,
    )
    return parse_grobid_fulltext(tei_xml, include_parsed, include_context)
