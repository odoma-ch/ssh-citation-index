"""Pipelines for parsing reference strings into structured References using an LLM."""

from __future__ import annotations

from pathlib import Path
from typing import List

from citation_index.core.models import References
from citation_index.llm.client import LLMClient
from citation_index.llm.prompt_loader import ReferenceParsingPrompt
from citation_index.utils.json_helper import safe_json_parse


def parse_reference_strings(
    reference_lines: List[str],
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_parsing.md",
    include_schema: bool = True,
    temperature: float = 0.0,
    use_streaming: bool = True,
) -> References:
    """Parse a list of reference strings into structured References via LLM."""
    text = "\n".join(reference_lines)
    prompt_obj = ReferenceParsingPrompt(
        prompt=prompt_name, input_text=text, include_json_schema=include_schema
    )
    response = llm_client.call(
        prompt_obj.prompt,
        json_output=True,
        json_schema=prompt_obj.json_schema,
        temperature=temperature,
        use_streaming=use_streaming,
    )
    parsed = safe_json_parse(response)
    if isinstance(parsed, list):
        data = parsed
    elif isinstance(parsed, dict):
        data = parsed.get("references") or parsed.get("parsed_references") or parsed.get("refs")
        if data is None:
            data = [parsed]
    else:
        data = []
    return References.from_dict(data) if data else References(references=[])


def parse_reference_file(
    path: str | Path,
    llm_client: LLMClient,
    prompt_name: str = "prompts/reference_parsing.md",
    include_schema: bool = True,
    temperature: float = 0.0,
) -> References:
    """Parse a text file with one reference per line into structured References."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return parse_reference_strings(
        lines,
        llm_client=llm_client,
        prompt_name=prompt_name,
        include_schema=include_schema,
        temperature=temperature,
    )


