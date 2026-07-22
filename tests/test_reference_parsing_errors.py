from unittest.mock import Mock, patch

import pytest

from citation_index.llm.client import LLMClient, LLMEmptyResponseError
from citation_index.pipelines.end_to_end_parsing import (
    EndToEndParsingError,
    run_pdf_one_step,
)
from citation_index.pipelines.reference_parsing import (
    ReferenceParsingError,
    parse_reference_strings,
)
from citation_index.utils.json_helper import safe_json_parse


class StubLLMClient:
    def __init__(self, response: str):
        self.response = response

    def call(self, *args, **kwargs) -> str:
        return self.response


def _stream(content=None, reasoning=None):
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.reasoning_content = reasoning
    stream = Mock()
    stream.__iter__ = Mock(return_value=iter([chunk]))
    return stream


def test_safe_json_parse_extracts_json_from_reasoning_wrapper():
    response = '<think>checking fields</think>\n{"references": []}'

    assert safe_json_parse(response) == {"references": []}


def test_parse_reference_strings_accepts_wrapped_valid_json():
    response = (
        "<think>done</think>\n"
        '{"references": [{"reference": {"full_title": "A title"}}]}'
    )

    result = parse_reference_strings(["Author. A title."], StubLLMClient(response))

    assert len(result) == 1
    assert result[0].full_title == "A title"


def test_parse_reference_strings_rejects_invalid_json():
    with pytest.raises(ReferenceParsingError, match="invalid JSON"):
        parse_reference_strings(
            ["Author. A title."], StubLLMClient("I found one reference")
        )


def test_parse_reference_strings_rejects_empty_result_for_nonempty_input():
    with pytest.raises(ReferenceParsingError, match="no references"):
        parse_reference_strings(
            ["Author. A title."], StubLLMClient('{"references": []}')
        )


def test_llm_client_retries_reasoning_only_stream():
    client = LLMClient(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        max_retries=1,
    )

    with (
        patch.object(
            client.client.chat.completions,
            "create",
            side_effect=[_stream(reasoning="thinking"), _stream(content="answer")],
        ),
        patch("citation_index.llm.client.time.sleep"),
    ):
        assert client.call("hello") == "answer"


def test_llm_client_raises_after_empty_responses():
    client = LLMClient(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        max_retries=0,
    )

    with patch.object(
        client.client.chat.completions,
        "create",
        return_value=_stream(),
    ):
        with pytest.raises(LLMEmptyResponseError, match="empty response"):
            client.call("hello")


def test_llm_client_disables_qwen_thinking_for_vllm():
    client = LLMClient(
        endpoint="http://localhost:8000/v1",
        model="Qwen3.6-27B-FP8",
        api_key="test-key",
        max_retries=0,
        enable_thinking=False,
    )

    with patch.object(
        client.client.chat.completions,
        "create",
        return_value=_stream(content='{"references": []}'),
    ) as create:
        client.call("hello")

    assert create.call_args.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_end_to_end_pipeline_rejects_invalid_llm_json():
    with pytest.raises(EndToEndParsingError, match="invalid JSON"):
        run_pdf_one_step(
            "References\nAuthor. A title.",
            StubLLMClient("the server generated text but no JSON"),
        )
