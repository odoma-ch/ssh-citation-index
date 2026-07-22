"""Pipelines for end-to-end reference extraction and parsing (text or PDF)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceExtractionAndParsingPrompt
from citation_index.utils.json_helper import safe_json_parse
from citation_index.core.models import References
from citation_index.core.segmenters.semantic_reference_locator import (
    locate_reference_sections_semantic,
)
from .reference_parsing import parse_reference_strings
from .reference_extraction import (
    extract_text_references,
    extract_text_references_by_page,
)
from .text_extraction import split_pages, extract_text

logger = logging.getLogger(__name__)


class EndToEndParsingError(ValueError):
    """Raised when an LLM answer cannot be converted to references."""


def _parse_json_to_references(response: str) -> References:
    parsed = safe_json_parse(response)
    if parsed is None:
        logger.error(
            "Could not parse end-to-end LLM response as JSON: chars=%d, preview=%r",
            len(response),
            response[:500].replace("\n", "\\n"),
        )
        raise EndToEndParsingError("LLM returned invalid JSON")

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
        raise EndToEndParsingError(
            f"LLM returned unsupported JSON type: {type(parsed).__name__}"
        )
    return References.from_dict(data) if data else References(references=[])


def run_pdf_one_step(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    prompt_name: str = "prompts/end_to_end_parsing.md",
    temperature: float = 0.1,
    include_schema: bool = True,
) -> References:
    """Method 1: One-step extraction+parsing on full text using a single prompt.

    - If `extractor` is provided and `text_or_pdf` is a valid file path, extract text first.
    - Otherwise, treat `text_or_pdf` as raw text.

    Args:
        prompt_name: Use "file.md" for legacy or "file.yaml:namespace.key" for YAML

    Examples:
        # Old way: run_pdf_one_step(pdf, client, prompt_name="prompts/file.md")
        # New way: run_pdf_one_step(pdf, client, prompt_name="prompts/prompts.yaml:extraction_and_parsing.default")
    """
    if (
        extractor is not None
        and isinstance(text_or_pdf, (str, Path))
        and Path(text_or_pdf).exists()
    ):
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    # Auto-detect format: YAML uses "path.yaml:namespace.key" syntax
    if ":" in prompt_name and prompt_name.split(":")[0].endswith((".yaml", ".yml")):
        # NEW WAY: YAML format with system/user separation
        yaml_path, prompt_key = prompt_name.split(":", 1)
        prompt = ReferenceExtractionAndParsingPrompt(
            prompt=yaml_path,
            prompt_key=prompt_key,
            input_text=input_text,
            include_json_schema=include_schema,
        )
        response = llm_client.call(
            messages=prompt.messages,
            json_output=True,
            temperature=temperature,
            json_schema=prompt.json_schema,
        )
    else:
        # OLD WAY: Legacy .md format
        prompt = ReferenceExtractionAndParsingPrompt(
            prompt=prompt_name,
            input_text=input_text,
            include_json_schema=include_schema,
        )
        response = llm_client.call(
            prompt.prompt,
            json_output=True,
            temperature=temperature,
            json_schema=prompt.json_schema,
        )

    return _parse_json_to_references(response)


def run_pdf_two_step(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    temperature: float = 0.3,
    include_schema: bool = True,
) -> References:
    """Method 2: Two-step – extract reference strings, then parse to structured refs."""
    if (
        extractor is not None
        and isinstance(text_or_pdf, (str, Path))
        and Path(text_or_pdf).exists()
    ):
        input_text = extract_text(text_or_pdf, extractor=extractor).text
    else:
        input_text = str(text_or_pdf)

    lines = extract_text_references(
        input_text, llm_client=llm_client, temperature=temperature
    )
    return parse_reference_strings(
        lines,
        llm_client=llm_client,
        temperature=temperature,
        include_schema=include_schema,
    )


def run_pdf_one_step_by_page(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    prompt_name: str = "prompts/end_to_end_parsing.md",
    temperature: float = 0.3,
    include_schema: bool = False,
    max_workers: int = 8,
) -> References:
    """Method 4: Page-wise one-step extraction+parsing, then aggregate (batched)."""
    # Split into pages based on extractor rules if provided, or heuristics otherwise
    if (
        extractor is not None
        and isinstance(text_or_pdf, (str, Path))
        and Path(text_or_pdf).exists()
    ):
        pages = split_pages(
            extract_text(text_or_pdf, extractor=extractor).text,
            extractor_type=extractor,
        )
    else:
        pages = split_pages(str(text_or_pdf), extractor_type=extractor)

    def _worker(page_text: str) -> Optional[List[dict]]:
        # Auto-detect format for parallel processing
        if ":" in prompt_name and prompt_name.split(":")[0].endswith((".yaml", ".yml")):
            # NEW WAY: YAML format
            yaml_path, prompt_key = prompt_name.split(":", 1)
            prompt = ReferenceExtractionAndParsingPrompt(
                prompt=yaml_path,
                prompt_key=prompt_key,
                input_text=page_text,
                include_json_schema=include_schema,
            )
            response = llm_client.call(
                messages=prompt.messages,
                json_output=True,
                temperature=temperature,
                json_schema=prompt.json_schema,
            )
        else:
            # OLD WAY: Legacy .md format
            prompt = ReferenceExtractionAndParsingPrompt(
                prompt=prompt_name,
                input_text=page_text,
                include_json_schema=include_schema,
            )
            response = llm_client.call(
                prompt.prompt,
                json_output=True,
                temperature=temperature,
                json_schema=prompt.json_schema,
            )

        refs = _parse_json_to_references(response)
        return [r.model_dump() for r in refs]

    all_refs: List[dict] = []
    completed_pages = 0
    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, p) for p in pages if p and p.strip()]
        for f in as_completed(futures):
            try:
                res = f.result()
                completed_pages += 1
                if isinstance(res, list):
                    all_refs.extend(res)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.exception("End-to-end LLM parsing failed for one page")

    if first_error is not None and completed_pages == 0:
        raise EndToEndParsingError(
            "End-to-end LLM parsing failed for every page"
        ) from first_error
    if first_error is not None:
        logger.warning(
            "Page-level end-to-end parsing completed partially: "
            "successful_pages=%d, total_pages=%d",
            completed_pages,
            len(futures),
        )
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
    return parse_reference_strings(
        all_lines,
        llm_client=llm_client,
        temperature=temperature,
        include_schema=include_schema,
    )


def run_pdf_semantic_one_step(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    embed_client,
    chunker=None,
    chunks=None,
    extractor: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
    prompt_name: str = "prompts/end_to_end_parsing.md",
    temperature: float = 0.3,
    include_schema: bool = True,
    fast_path: bool = False,
) -> References:
    """Method 3: Semantic section detection + one-step extraction and parsing.

    Uses embedding-based semantic search to locate reference sections, then
    applies one-step LLM-based extraction and parsing to those sections.

    Args:
        text_or_pdf: Input text or PDF path
        llm_client: LLM client for extraction and parsing
        embed_client: EmbedClient for getting embeddings
        chunker: Text chunker object with chunk() method. Ignored if chunks parameter is provided.
        chunks: Pre-computed chunks from the text. If provided, chunker is ignored.
        extractor: Text extractor type (if PDF input)
        embedding_model: Model for semantic embeddings
        prompt_name: Prompt template for extraction and parsing
        temperature: LLM temperature
        include_schema: Include JSON schema in prompt
        fast_path: Try regex matching first

    Returns:
        References object containing parsed references
    """
    # Extract text if PDF input
    if (
        extractor is not None
        and isinstance(text_or_pdf, (str, Path))
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

    # One-step extraction and parsing on the located sections
    if ":" in prompt_name and prompt_name.split(":")[0].endswith((".yaml", ".yml")):
        # NEW WAY: YAML format
        yaml_path, prompt_key = prompt_name.split(":", 1)
        prompt = ReferenceExtractionAndParsingPrompt(
            prompt=yaml_path,
            prompt_key=prompt_key,
            input_text=reference_sections,
            include_json_schema=include_schema,
        )
        response = llm_client.call(
            messages=prompt.messages,
            json_output=True,
            temperature=temperature,
            json_schema=prompt.json_schema,
        )
    else:
        # OLD WAY: Legacy .md format
        prompt = ReferenceExtractionAndParsingPrompt(
            prompt=prompt_name,
            input_text=reference_sections,
            include_json_schema=include_schema,
        )
        response = llm_client.call(
            prompt.prompt,
            json_output=True,
            temperature=temperature,
            json_schema=prompt.json_schema,
        )

    references = _parse_json_to_references(response)
    # if references is empty, use method 1 as fallback
    if not references:
        references = run_pdf_one_step(
            input_text,
            llm_client=llm_client,
            extractor=extractor,
            prompt_name=prompt_name,
            temperature=temperature,
            include_schema=include_schema,
        )
    return references


# Backwards-compatible alias (previous default behavior matched method 1)
def run_pdf_extract_and_parse(
    text_or_pdf: str | Path,
    llm_client: LLMClient,
    extractor: Optional[str] = None,
    include_schema: bool = True,
    temperature: float = 0.3,
    prompt_name: str = "prompts/end_to_end_parsing.md",
) -> References:
    return run_pdf_one_step(
        text_or_pdf,
        llm_client=llm_client,
        extractor=extractor,
        prompt_name=prompt_name,
        temperature=temperature,
        include_schema=include_schema,
    )


if __name__ == "__main__":
    """Simple test demonstrating both old and new prompt formats."""
    print("=" * 80)
    print("EXTRACTION AND PARSING PIPELINE - Test Both Prompt Formats")
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
    This paper discusses recent advances in machine learning.
    
    References
    
    1. Smith, J., & Brown, A. (2020). Deep Learning in NLP. AI Journal, 15(3), 100-120.
    2. Jones, M. (2019). Machine Translation Systems. MIT Press.
    3. Chen, L., Wang, X., & Li, Y. (2021). Transformer Models for Sequence Processing. Nature, 567, 45-50.
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
    print("""  result = run_pdf_one_step(
      text,
      client,
      prompt_name="prompts/end_to_end_parsing.md",
      include_schema=True
  )""")

    if can_call_api and client:
        try:
            result = run_pdf_one_step(
                test_text,
                client,
                prompt_name="prompts/end_to_end_parsing.md",
                include_schema=True,
            )
            print(f"\n✓ Extracted and parsed {len(result.references)} references")
            for i, ref in enumerate(result.references[:2], 1):
                print(f"  {i}. {ref.full_title}")
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
    print("""  result = run_pdf_one_step(
      text,
      client,
      prompt_name="prompts/prompts.yaml:extraction_and_parsing.default",
      include_schema=True
  )""")

    if can_call_api and client:
        try:
            result = run_pdf_one_step(
                test_text,
                client,
                prompt_name="prompts/prompts.yaml:extraction_and_parsing.default",
                include_schema=True,
            )
            print(f"\n✓ Extracted and parsed {len(result.references)} references")
            for i, ref in enumerate(result.references[:2], 1):
                print(f"  {i}. {ref.full_title}")
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

    test_input = "Test text with references."

    # Legacy prompt
    print("\nLegacy .md prompt:")
    legacy_prompt = ReferenceExtractionAndParsingPrompt(
        prompt="prompts/end_to_end_parsing.md",
        input_text=test_input,
        include_json_schema=False,
    )
    print("  Format: markdown")
    print(f"  Type: {type(legacy_prompt.prompt).__name__}")
    print(f"  Length: {len(legacy_prompt.prompt)} chars")

    # YAML prompt
    print("\nYAML prompt:")
    yaml_prompt = ReferenceExtractionAndParsingPrompt(
        prompt="prompts/prompts.yaml",
        prompt_key="extraction_and_parsing.default",
        input_text=test_input,
        include_json_schema=False,
    )
    messages = yaml_prompt.messages
    print("  Format: yaml")
    print(f"  Type: {type(messages).__name__}")
    print(f"  System length: {len(messages['system'])} chars")
    print(f"  User length: {len(messages['user'])} chars")
