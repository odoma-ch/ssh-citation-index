"""Pipelines for extracting reference strings from text or PDFs using an LLM."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from citation_index.core.extractors import ExtractorFactory
from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceExtractionPrompt
from citation_index.core.segmenters.semantic_reference_locator import locate_reference_sections_semantic
from .text_extraction import split_pages, extract_text


def extract_text_references(
    text: str,
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.3,
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


def extract_text_references_semantic_sections(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    chunker=None,
    chunks=None,
    extractor: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    embedding_endpoint: str = "http://0.0.0.0:7997/embeddings",
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.3,
    fast_path: bool = False,
) -> List[str]:
    """Method 2: Semantic reference section detection followed by LLM extraction.
    
    Uses embedding-based semantic search to locate reference sections, then
    applies LLM-based extraction to those sections only.
    
    Args:
        text_or_pdf: Input text or PDF path
        llm_client: LLM client for reference extraction
        chunker: Text chunker object with chunk() method. Ignored if chunks parameter is provided.
        chunks: Pre-computed chunks from the text. If provided, chunker is ignored.
        extractor: Text extractor type (if PDF input)
        embedding_model: Model for semantic embeddings
        embedding_endpoint: API endpoint for embedding service
        prompt_name: Prompt template for reference extraction
        temperature: LLM temperature
        top_k: Minimum chunks to consider for reference sections
        top_percentile: Percentile threshold for chunk selection
        fast_path: Try regex matching first
        gap_size_threshold: Minimum gap size to trigger gap-based candidate selection
        drop_tolerance: Maximum score drop allowed during contiguous expansion
        
    Returns:
        List of extracted reference strings
    """
    # Extract text if PDF input
    if isinstance(text_or_pdf, (str, Path)) and extractor is not None and Path(text_or_pdf).exists():
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)
    
    # Locate reference sections using semantic search
    reference_sections = locate_reference_sections_semantic(
        input_text,
        chunker=chunker,
        chunks=chunks,
        embedding_model=embedding_model,
        embedding_endpoint=embedding_endpoint,
        fast_path=fast_path
    )
    
    if not reference_sections.strip():
        reference_sections = input_text
    
    # Extract references from the located sections
    references = extract_text_references(
        reference_sections,
        llm_client=llm_client,
        prompt_name=prompt_name,
        temperature=temperature
    )
    # if references is empty, use method 1 as fallback
    if not references:
        return extract_text_references(
            input_text,
            llm_client=llm_client,
            prompt_name=prompt_name,
            temperature=temperature
        )
    return references


