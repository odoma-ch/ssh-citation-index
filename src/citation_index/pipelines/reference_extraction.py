"""Pipelines for extracting reference strings from text or PDFs using an LLM."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from citation_index.core.extractors import ExtractorFactory
from citation_index.core.segmenters.references_locator import extract_all_reference_sections
from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceExtractionPrompt
from .text_extraction import split_pages, extract_text


def extract_text_references(
    text: str,
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.0,
) -> List[str]:
    """Extract reference strings (one per line) from raw text via LLM.

    Returns a list of non-empty trimmed lines as reference candidates.
    """
    prompt = ReferenceExtractionPrompt(prompt=prompt_name, input_text=text).prompt
    response = llm_client.call(prompt, json_output=False, temperature=temperature)
    # remove <start>/<start> and <end>/</end> tags
    response = re.sub(r"<\/?\s*start\s*>", "", response, flags=re.IGNORECASE)
    response = re.sub(r"<\/?\s*end\s*>", "", response, flags=re.IGNORECASE)
    if not response.strip():
        return []
    lines = [ln.strip() for ln in response.splitlines()]
    return [ln for ln in lines if ln]


def extract_pdf_references(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: str = "pymupdf",
    section_locator=extract_all_reference_sections,
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.0,
) -> List[str]:
    """Method 1: Extract reference strings from a PDF using an LLM."""
    if isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)
    return extract_text_references_section_detect(input_text, extractor, section_locator)


def extract_text_references_section_detect(
    text_or_pdf: str | Path,
    extractor: Optional[str] = None,
    section_locator=extract_all_reference_sections,
) -> List[str]:
    """Method 2: Extract reference strings from text by detecting the reference section without using an LLM.
    
    - If `extractor` is provided and `text_or_pdf` is a valid file path, extract text first.
    - Otherwise, treat `text_or_pdf` as raw text.
    - Uses section detection to locate reference sections and returns raw text lines.
    """
    if extractor is not None and isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    sections = section_locator(input_text, prefer_tokens=False)
    section_texts = [section["text"] for section in sections]
    
    # Combine all section texts and split into lines
    all_text = "\n".join(section_texts)
    lines = [ln.strip() for ln in all_text.splitlines()]
    return [ln for ln in lines if ln]


def extract_text_references_by_page(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.0,
    max_workers: int = 8,
) -> List[str]:
    """Method 3: Page-level reference extraction.

    - If `extractor` is provided and `text_or_pdf` is a path, extract text first and split by extractor rules.
    - If `extractor` is None, treat `text_or_pdf` as raw text and split heuristically.
    - Calls the LLM per page in parallel batches (max_workers).
    """
    # Prepare page texts
    if isinstance(text_or_pdf, (str, Path)) and extractor is not None and Path(text_or_pdf).exists():
        pages = split_pages(extract_text(text_or_pdf, extractor=extractor).text, extractor_type=extractor)
    else:
        # Treat as raw text
        txt = str(text_or_pdf)
        pages = split_pages(txt, extractor_type=extractor)

    def _worker(page_text: str) -> List[str]:
        refs = extract_text_references(
            page_text, llm_client=llm_client, prompt_name=prompt_name, temperature=temperature
        )
        return refs

    results: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, p) for p in pages if p and p.strip()]
        for f in as_completed(futures):
            try:
                res = f.result()
                if isinstance(res, list):
                    results.extend([r for r in res if r])
            except Exception:
                continue
    return results


