"""Pipelines for extracting reference strings from text or PDFs using an LLM."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceExtractionPrompt
from citation_index.core.segmenters.semantic_reference_locator import (
    locate_reference_sections_semantic,
)
from .text_extraction import split_pages, extract_text

logger = logging.getLogger(__name__)


def extract_text_references(
    text: str,
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_extraction.md",
    temperature: float = 0.3,
) -> List[str]:
    """Extract reference strings (one per line) from raw text via LLM.

    Returns a list of non-empty trimmed lines as reference candidates.

    Args:
        text: Input text containing references
        llm_client: LLM client for API calls
        prompt_name: Use "file.md" for legacy or "file.yaml:namespace.key" for YAML
        temperature: LLM temperature parameter

    Examples:
        # Old way: extract_text_references(text, client, "prompts/file.md")
        # New way: extract_text_references(text, client, "prompts/prompts.yaml:extraction.default")
    """
    # Auto-detect format: YAML uses "path.yaml:namespace.key" syntax
    if ":" in prompt_name and prompt_name.split(":")[0].endswith((".yaml", ".yml")):
        # NEW WAY: YAML format with system/user separation
        yaml_path, prompt_key = prompt_name.split(":", 1)
        prompt_obj = ReferenceExtractionPrompt(
            prompt=yaml_path, prompt_key=prompt_key, input_text=text
        )
        response = llm_client.call(
            messages=prompt_obj.messages, json_output=False, temperature=temperature
        )
    else:
        # OLD WAY: Legacy .md format
        prompt_obj = ReferenceExtractionPrompt(prompt=prompt_name, input_text=text)
        response = llm_client.call(
            prompt_obj.prompt, json_output=False, temperature=temperature
        )

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
    if (
        isinstance(text_or_pdf, (str, Path))
        and extractor is not None
        and Path(text_or_pdf).exists()
    ):
        pages = split_pages(
            extract_text(text_or_pdf, extractor=extractor).text,
            extractor_type=extractor,
        )
    else:
        # Treat as raw text
        txt = str(text_or_pdf)
        pages = split_pages(txt, extractor_type=extractor)

    def _worker(page_text: str) -> List[str]:
        refs = extract_text_references(
            page_text,
            llm_client=llm_client,
            prompt_name=prompt_name,
            temperature=temperature,
        )
        return refs

    results: List[str] = []
    completed_pages = 0
    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, p) for p in pages if p and p.strip()]
        for f in as_completed(futures):
            try:
                res = f.result()
                completed_pages += 1
                if isinstance(res, list):
                    results.extend([r for r in res if r])
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.exception("LLM reference extraction failed for one page")

    if first_error is not None and completed_pages == 0:
        raise RuntimeError(
            "LLM reference extraction failed for every page"
        ) from first_error
    if first_error is not None:
        logger.warning(
            "Page-level reference extraction completed partially: "
            "successful_pages=%d, total_pages=%d",
            completed_pages,
            len(futures),
        )
    return results


def extract_text_references_semantic_sections(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    embed_client,
    chunker=None,
    chunks=None,
    extractor: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
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
        embed_client: EmbedClient for getting embeddings
        chunker: Text chunker object with chunk() method. Ignored if chunks parameter is provided.
        chunks: Pre-computed chunks from the text. If provided, chunker is ignored.
        extractor: Text extractor type (if PDF input)
        embedding_model: Model for semantic embeddings
        prompt_name: Prompt template for reference extraction
        temperature: LLM temperature
        fast_path: Try regex matching first

    Returns:
        List of extracted reference strings
    """
    # Extract text if PDF input
    if (
        isinstance(text_or_pdf, (str, Path))
        and extractor is not None
        and Path(text_or_pdf).exists()
    ):
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    # Locate reference sections using semantic search
    reference_sections = locate_reference_sections_semantic(
        input_text,
        embed_client=embed_client,
        embedding_model=embedding_model,
        chunker=chunker,
        chunks=chunks,
        fast_path=fast_path,
    )

    if not reference_sections.strip():
        reference_sections = input_text

    # Extract references from the located sections
    references = extract_text_references(
        reference_sections,
        llm_client=llm_client,
        prompt_name=prompt_name,
        temperature=temperature,
    )
    # if references is empty, use method 1 as fallback
    if not references:
        return extract_text_references(
            input_text,
            llm_client=llm_client,
            prompt_name=prompt_name,
            temperature=temperature,
        )
    return references


if __name__ == "__main__":
    """Simple test demonstrating both old and new prompt formats."""
    print("=" * 80)
    print("REFERENCE EXTRACTION PIPELINE - Test Both Prompt Formats")
    print("=" * 80)

    # ============================================================
    # CONFIGURATION - Replace with your actual values
    # ============================================================
    API_KEY = "your-api-key-here"  # TODO: Replace with actual API key
    ENDPOINT = "https://api.openai.com/v1"  # TODO: Replace with your LLM endpoint
    MODEL = "gpt-4"  # TODO: Replace with your model name

    print("\n⚠️  Configuration (update before running):")
    print(f"  API_KEY: {API_KEY}")
    print(f"  ENDPOINT: {ENDPOINT}")
    print(f"  MODEL: {MODEL}")

    # Test data - sample text with references
    test_text = """
    This paper discusses recent advances in machine learning and natural language processing.
    
    References
    
    1. Smith, J., & Brown, A. (2020). Deep Learning in NLP. AI Journal, 15(3), 100-120.
    2. Jones, M. (2019). Machine Translation Systems. MIT Press.
    3. Chen, L., Wang, X., & Li, Y. (2021). Transformer Models. Nature, 567, 45-50.
    4. Davis, R. (2018). Neural Networks for Language Understanding. Springer.
    """

    print("\n📄 Test text (excerpt):")
    print(f"  {test_text[:150]}...")

    # Initialize client
    try:
        client = LLMClient(endpoint=ENDPOINT, model=MODEL, api_key=API_KEY)
        print(f"\n✓ LLM Client initialized: {MODEL} at {ENDPOINT}")
        print("  (Will fail on actual API call with placeholder credentials)")
        can_call_api = False  # Set to True if you have real credentials
    except Exception as e:
        print(f"\n✗ Client initialization failed: {e}")
        client = None
        can_call_api = False

    # ============================================================
    # Example 1: OLD WAY - Legacy .md prompts
    # ============================================================
    print("\n" + "-" * 80)
    print("Example 1: OLD WAY - Using legacy .md prompts")
    print("-" * 80)
    print("\nCode:")
    print("""  refs = extract_text_references(
      text,
      client,
      prompt_name="prompts/reference_extraction.md",
      temperature=0.3
  )""")

    if can_call_api and client:
        try:
            refs = extract_text_references(
                test_text,
                client,
                prompt_name="prompts/reference_extraction.md",
                temperature=0.3,
            )
            print(f"\n✓ Extracted {len(refs)} reference strings")
            for i, ref in enumerate(refs[:3], 1):
                print(f"  {i}. {ref[:80]}...")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("\n⚠️  Skipped (requires valid API credentials)")

    # ============================================================
    # Example 2: NEW WAY - YAML prompts
    # ============================================================
    print("\n" + "-" * 80)
    print("Example 2: NEW WAY - Using YAML prompts with system/user separation")
    print("-" * 80)
    print("\nCode:")
    print("""  refs = extract_text_references(
      text,
      client,
      prompt_name="prompts/prompts.yaml:extraction.default",
      temperature=0.3
  )""")

    if can_call_api and client:
        try:
            refs = extract_text_references(
                test_text,
                client,
                prompt_name="prompts/prompts.yaml:extraction.default",
                temperature=0.3,
            )
            print(f"\n✓ Extracted {len(refs)} reference strings")
            for i, ref in enumerate(refs[:3], 1):
                print(f"  {i}. {ref[:80]}...")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("\n⚠️  Skipped (requires valid API credentials)")

    # ============================================================
    # Example 3: Show prompt differences (no API call needed)
    # ============================================================
    print("\n" + "-" * 80)
    print("Example 3: Prompt structure comparison (no API call needed)")
    print("-" * 80)

    test_input = "Test text with references section."

    # Legacy prompt
    print("\nLegacy .md prompt:")
    legacy_prompt = ReferenceExtractionPrompt(
        prompt="prompts/reference_extraction.md", input_text=test_input
    )
    print("  Format: markdown")
    print(f"  Type: {type(legacy_prompt.prompt).__name__}")
    print(f"  Length: {len(legacy_prompt.prompt)} chars")
    print(f"  Preview: {legacy_prompt.prompt[:100]}...")

    # YAML prompt
    print("\nYAML prompt:")
    yaml_prompt = ReferenceExtractionPrompt(
        prompt="prompts/prompts.yaml",
        prompt_key="extraction.default",
        input_text=test_input,
    )
    messages = yaml_prompt.messages
    print("  Format: yaml")
    print(f"  Type: {type(messages).__name__}")
    print(f"  System ({len(messages['system'])} chars): {messages['system'][:80]}...")
    print(f"  User ({len(messages['user'])} chars): {messages['user'][:80]}...")
