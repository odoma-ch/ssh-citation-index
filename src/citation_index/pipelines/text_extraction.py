"""Utilities for text extraction and page splitting from PDFs or raw text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from citation_index.core.extractors import ExtractorFactory
from citation_index.core.extractors.base import ExtractResult


def extract_text(
    pdf_path: str | Path,
    extractor: str = "pymupdf",
    markdown: bool = True,
    save_dir: str | None = None,
) -> ExtractResult:
    """Extract text from a PDF using the configured extractor."""
    extractor_impl = ExtractorFactory.create(extractor)
    return extractor_impl.extract(str(pdf_path), save_dir=save_dir, markdown=markdown)


def split_pages(text: str, extractor_type: str | None = None) -> List[str]:
    """Split extracted text into per-page chunks based on extractor-specific markers.

    Rules provided:
      - mineru: cannot split → return single chunk
      - marker: each page is preceded by a line like "{n}------------------------------------------------"
      - pymupdf: pages separated by "\n-----\n"

    If extractor_type is None, try to infer by searching for known separators.
    """
    if not text:
        return []

    # Explicit rules by extractor
    if extractor_type is not None:
        et = extractor_type.lower()
        if et == "mineru":
            return [text]
        if et == "pymupdf":
            if "\n-----\n" in text:
                return text.split("\n-----\n")
            # fallback to generic
        if et == "marker":
            return _split_by_marker_header(text)

    # Heuristic fallback when type is unknown
    if "\n-----\n" in text:  # likely pymupdf
        return text.split("\n-----\n")
    if re.search(r"^\s*\d+\s*-{5,}\s*$", text, flags=re.MULTILINE):  # likely marker
        return _split_by_marker_header(text)

    # No separators detected
    return [text]


def _split_by_marker_header(text: str) -> List[str]:
    """Split text where lines look like `{n}-----...` as page headers (marker output)."""
    lines = text.splitlines()
    pages: List[str] = []
    current: List[str] = []
    header_re = re.compile(r"^\s*\d+\s*-{5,}\s*$")
    for line in lines:
        if header_re.match(line):
            if current:
                pages.append("\n".join(current).strip())
                current = []
            # Do not include header line in content
            continue
        current.append(line)
    if current:
        pages.append("\n".join(current).strip())
    # Filter empties
    return [p for p in pages if p]


def split_pages_from_pdf(
    pdf_path: str | Path,
    extractor: str = "pymupdf",
    markdown: bool = True,
) -> List[str]:
    """Extract text then split into pages based on extractor rules."""
    res = extract_text(pdf_path, extractor=extractor, markdown=markdown)
    return split_pages(res.text, extractor_type=extractor)


