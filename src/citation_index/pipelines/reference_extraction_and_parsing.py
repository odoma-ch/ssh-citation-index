"""Pipelines for combined reference extraction and parsing (text or PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from citation_index.core.segmenters.references_locator import extract_all_reference_sections
from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceExtractionAndParsingPrompt
from citation_index.utils.json_helper import safe_json_parse
from citation_index.core.models import References
from .reference_parsing import parse_reference_strings
from .reference_extraction import extract_text_references, extract_text_references_by_page
from .text_extraction import split_pages, extract_text


def _parse_json_to_references(response: str) -> References:
    parsed = safe_json_parse(response)
    if isinstance(parsed, list):
        data = parsed
    elif isinstance(parsed, dict):
        data = (
            parsed.get("references")
            or parsed.get("parsed_references")
            or parsed.get("refs")
        )
        if data is None:
            data = [parsed]
    else:
        data = []
    return References.from_dict(data) if data else References(references=[])


def run_pdf_one_step(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    prompt_name: str = "prompts/reference_extraction_and_parsing.md",
    temperature: float = 0.3,
    include_schema: bool = True,
) -> References:
    """Method 1: One-step extraction+parsing on full text using a single prompt.

    - If `extractor` is provided and `text_or_pdf` is a valid file path, extract text first.
    - Otherwise, treat `text_or_pdf` as raw text.
    """
    if extractor is not None and isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    prompt = ReferenceExtractionAndParsingPrompt(
        prompt=prompt_name, input_text=input_text, include_json_schema=include_schema
    )
    response = llm_client.call(prompt.prompt, json_output=True, temperature=temperature, json_schema=prompt.json_schema)
    return _parse_json_to_references(response)


def run_pdf_two_step(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    temperature: float = 0.3,
    include_schema: bool = True,
) -> References:
    """Method 2: Two-step – extract reference strings, then parse to structured refs."""
    if extractor is not None and isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    lines = extract_text_references(input_text, llm_client=llm_client, temperature=temperature)
    return parse_reference_strings(lines, llm_client=llm_client, temperature=temperature, include_schema=include_schema)


def run_pdf_section_detect_and_parse(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    section_locator=extract_all_reference_sections,
    include_schema: bool = False,
    temperature: float = 0.3,
) -> References:
    """Method 3: Detect references section, then parse lines to structured refs (current)."""
    if extractor is not None and isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    sections = section_locator(input_text)
    section_texts = [section["text"] for section in sections]
    return parse_reference_strings(
        section_texts,
        llm_client=llm_client,
        include_schema=include_schema,
        temperature=temperature,
    )


def run_pdf_one_step_by_page(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    prompt_name: str = "prompts/reference_extraction_and_parsing.md",
    temperature: float = 0.3,
    include_schema: bool = False,
    max_workers: int = 8,
) -> References:
    """Method 4: Page-wise one-step extraction+parsing, then aggregate (batched)."""
    # Split into pages based on extractor rules if provided, or heuristics otherwise
    if extractor is not None and isinstance(text_or_pdf, (str, Path)) and Path(text_or_pdf).exists():
        pages = split_pages(extract_text(text_or_pdf, extractor=extractor).text, extractor_type=extractor)
    else:
        pages = split_pages(str(text_or_pdf), extractor_type=extractor)

    def _worker(page_text: str) -> Optional[List[dict]]:
        prompt = ReferenceExtractionAndParsingPrompt(
            prompt=prompt_name, input_text=page_text, include_json_schema=include_schema
        )
        response = llm_client.call(prompt.prompt, json_output=True, temperature=temperature, json_schema=prompt.json_schema)
        refs = _parse_json_to_references(response)
        return [r.model_dump() for r in refs]

    all_refs: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, p) for p in pages if p and p.strip()]
        for f in as_completed(futures):
            try:
                res = f.result()
                if isinstance(res, list):
                    all_refs.extend(res)
            except Exception:
                continue
    return References.from_dict(all_refs) if all_refs else References(references=[])


def run_pdf_two_step_by_page(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    temperature: float = 0.3,
    include_schema: bool = False,
    max_workers: int = 8,
) -> References:
    """Method 5: Page-wise extraction of strings, concatenate, then parse once (batched)."""
    all_lines: List[str] = extract_text_references_by_page(
        text_or_pdf,
        llm_client=llm_client,
        extractor=extractor,
        temperature=temperature,
        max_workers=max_workers,
    )
    return parse_reference_strings(all_lines, llm_client=llm_client, temperature=temperature, include_schema=include_schema)


# Backwards-compatible alias (previous default behavior matched method 1)
def run_pdf_extract_and_parse(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    section_locator=extract_all_reference_sections,
    include_schema: bool = True,
    temperature: float = 0.3,
    prompt_name: str = "prompts/reference_extraction_and_parsing.md",
) -> References:
    return run_pdf_one_step(
        text_or_pdf,
        llm_client=llm_client,
        extractor=extractor,
        prompt_name=prompt_name,
        temperature=temperature,
        include_schema=include_schema,
    )


