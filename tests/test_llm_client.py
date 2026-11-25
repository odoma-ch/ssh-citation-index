"""
Comprehensive tests for the improved LLM client implementations.
Tests DeepSeekClient, VLLMClient, and base LLMClient functionality.
"""

import json
import pytest
import numpy as np
from unittest.mock import Mock, patch
from citation_index.llm.client import LLMClient, DeepSeekClient, VLLMClient, EmbedClient


def create_mock_stream(content):
    """Helper to create mock streaming response."""
    mock_stream = Mock()
    mock_stream.choices = [Mock()]
    mock_stream.choices[0].delta.content = content
    return [mock_stream]


class TestLLMClient:
    """Test cases for base LLMClient functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_api_key = "test-key"
        self.mock_endpoint = "http://localhost:8000/v1"

    def test_response_format_handling(self):
        """Test base client response format handling."""
        client = LLMClient(
            endpoint=self.mock_endpoint, model="test-model", api_key=self.mock_api_key
        )

        # Test simple JSON output
        response_format, prompt, max_tokens = client._get_response_format_and_prompt(
            prompt="Generate JSON", json_output=True
        )

        assert response_format == {"type": "json_object"}
        assert prompt == "Generate JSON"  # Should not be modified

        # Test with JSON schema
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format, prompt, max_tokens = client._get_response_format_and_prompt(
            prompt="Generate data", json_schema=schema
        )

        expected_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": schema,
            },
        }
        assert response_format == expected_format
        assert prompt == "Generate data"  # Should not be modified

    def test_guided_decoding_disabled_falls_back_to_json_object(self):
        """guided_decoding=False keeps JSON mode but sends no grammar."""
        client = LLMClient(
            endpoint=self.mock_endpoint,
            model="test-model",
            api_key=self.mock_api_key,
            guided_decoding=False,
        )

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format, _, _ = client._get_response_format_and_prompt(
            prompt="Generate data",
            json_schema=schema,
        )

        assert response_format == {"type": "json_object"}


class TestDeepSeekClient:
    """Test cases for DeepSeekClient functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_api_key = "test-key"

    def test_deepseek_response_format_handling(self):
        """Test DeepSeek-specific response format handling."""
        client = DeepSeekClient(api_key=self.mock_api_key)

        # Test JSON output without continuation
        response_format, prompt, max_tokens = (
            client._get_deepseek_response_format_and_prompt(
                prompt="Generate data",
                json_output=True,
                max_tokens=4000,
                use_continuation=False,
            )
        )

        assert response_format == {"type": "json_object"}
        assert max_tokens == 8000  # Should be increased to 8000
        assert "JSON format" in prompt  # Should add JSON instruction
        assert "<start>" not in prompt  # No continuation tags

        # Test JSON output with continuation
        response_format, prompt, max_tokens = (
            client._get_deepseek_response_format_and_prompt(
                prompt="Extract references",
                json_output=True,
                max_tokens=4000,
                use_continuation=True,
            )
        )

        assert response_format == {"type": "json_object"}
        assert max_tokens == 8000
        assert "JSON format" in prompt
        assert "<start>" in prompt and "<end>" in prompt  # Should add continuation tags

    def test_deepseek_default_continuation_behavior(self):
        """Test that DeepSeek uses continuation by default for JSON."""
        client = DeepSeekClient(api_key=self.mock_api_key)

        # Mock continuation responses
        def mock_side_effect(*args, **kwargs):
            if mock_side_effect.call_count == 0:
                mock_side_effect.call_count += 1
                return create_mock_stream('<start>\n{"data": [1, 2')
            else:
                return create_mock_stream(", 3]}\n<end>")

        mock_side_effect.call_count = 0

        mock_create = Mock(side_effect=mock_side_effect)

        with patch.object(client.client.chat.completions, "create", mock_create):
            response = client.call(prompt="Generate JSON data", json_output=True)

            # Should use continuation by default for JSON; client strips tags from response
            assert mock_create.call_count == 2
            assert isinstance(response, str)  # Should return string, not tuple
            assert json.loads(response) == {"data": [1, 2, 3]}


class TestVLLMClient:
    """Test cases for VLLMClient functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_endpoint = "http://localhost:8000/v1"
        self.mock_model = "llama-3-8b"

    def test_vllm_vs_deepseek_response_format(self):
        """Test response format differences between vLLM and DeepSeek."""
        # Test vLLM
        vllm_client = VLLMClient(
            endpoint=self.mock_endpoint, model=self.mock_model, api_key="test-key"
        )

        mock_response = '{"data": "test"}'
        mock_create = Mock()
        mock_create.return_value = create_mock_stream(mock_response)

        with patch.object(vllm_client.client.chat.completions, "create", mock_create):
            response = vllm_client.call(prompt="Generate JSON data", json_output=True)

            # Verify vLLM uses simple json_object without modifications
            call_args = mock_create.call_args_list[0][1]
            assert call_args["response_format"] == {"type": "json_object"}

            # Verify prompt was not modified
            prompt_content = call_args["messages"][0]["content"]
            assert prompt_content == "Generate JSON data"

            # Verify no special max_tokens handling
            assert call_args.get("max_tokens") is None

        # Compare with DeepSeek
        deepseek_client = DeepSeekClient(api_key="test-key")

        mock_create.reset_mock()
        mock_create.return_value = create_mock_stream(mock_response)

        with patch.object(
            deepseek_client.client.chat.completions, "create", mock_create
        ):
            response = deepseek_client.call(
                prompt="Generate data", json_output=True, use_continuation=False
            )

            # Verify DeepSeek applies modifications
            call_args = mock_create.call_args_list[0][1]
            assert call_args["response_format"] == {"type": "json_object"}

            # Verify prompt was modified (adds JSON instruction when "json" not in prompt)
            prompt_content = call_args["messages"][0]["content"]
            assert "JSON format" in prompt_content or "json" in prompt_content.lower()

            # Verify max_tokens was set
            assert call_args["max_tokens"] == 8000

    def test_vllm_parsed_structured_output(self):
        """Test vLLM structured output with Pydantic models."""
        client = VLLMClient(
            endpoint=self.mock_endpoint, model=self.mock_model, api_key="test-key"
        )

        # Mock the parsed response
        mock_parsed_response = Mock()
        mock_parsed_response.choices = [Mock()]
        mock_parsed_response.choices[0].message.parsed = {"name": "test", "value": 42}

        mock_parse = Mock()
        mock_parse.return_value = mock_parsed_response

        # Mock a simple Pydantic-like class
        class TestModel:
            pass

        with patch.object(client.client.beta.chat.completions, "parse", mock_parse):
            result = client.call_with_parsed_structured_output(
                prompt="Generate structured data", json_class=TestModel
            )

            assert result == {"name": "test", "value": 42}

            # Verify the parse method was called correctly
            call_args = mock_parse.call_args_list[0][1]
            assert call_args["response_format"] == TestModel


class TestEmbedClient:
    """Test cases for EmbedClient functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_endpoint = "https://api.example.com/v1"
        self.mock_model = "text-embedding-model"
        self.mock_api_key = "test-embed-key"

    def test_embed_client_strips_embeddings_path(self):
        """Full '/v1/embeddings' URL must not become '/v1/embeddings/embeddings'."""
        client = EmbedClient(
            endpoint="https://api.example.com/v1/embeddings/", model=self.mock_model
        )
        assert client.endpoint == "https://api.example.com/v1"
        assert str(client.client.base_url).rstrip("/") == "https://api.example.com/v1"

    def test_get_embeddings_batch(self):
        """Test getting embeddings for multiple texts in one call."""
        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
        )

        # Mock response for batch
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
            Mock(embedding=[0.5, 0.6]),
        ]

        mock_create = Mock(return_value=mock_response)

        with patch.object(client.client.embeddings, "create", mock_create):
            texts = ["text1", "text2", "text3"]
            embeddings = client.get_embeddings(texts=texts)

            assert embeddings.shape == (3, 2)
            np.testing.assert_array_equal(embeddings[0], [0.1, 0.2])
            np.testing.assert_array_equal(embeddings[1], [0.3, 0.4])
            np.testing.assert_array_equal(embeddings[2], [0.5, 0.6])

            # SDK handles batch in single call
            assert mock_create.call_count == 1
            call_args = mock_create.call_args_list[0][1]
            assert call_args["input"] == texts

    def test_get_embeddings_with_retry(self):
        """Test retry logic on retryable embedding failures."""
        from citation_index.llm.client import LLMTimeoutError

        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
            max_retries=2,
        )

        # Mock: fail twice with retryable errors, succeed third time
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]

        mock_create = Mock(
            side_effect=[
                LLMTimeoutError("Timeout"),
                LLMTimeoutError("Timeout again"),
                mock_response,
            ]
        )

        with patch.object(client.client.embeddings, "create", mock_create):
            with patch("time.sleep"):  # Skip actual sleep in test
                embeddings = client.get_embeddings(texts=["test"])

                # Should succeed after 2 retries
                assert mock_create.call_count == 3
                assert embeddings.shape == (1, 2)

    def test_get_embeddings_max_retries_exceeded(self):
        """Test that max retries are respected for retryable errors."""
        from citation_index.llm.client import LLMTimeoutError

        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
            max_retries=1,
        )

        mock_create = Mock(side_effect=LLMTimeoutError("Persistent timeout"))

        with patch.object(client.client.embeddings, "create", mock_create):
            with patch("time.sleep"):
                with pytest.raises(LLMTimeoutError, match="Persistent timeout"):
                    client.get_embeddings(texts=["test"])

                # Should try: initial + 1 retry = 2 total
                assert mock_create.call_count == 2

    def test_get_embeddings_non_retryable_error_fails_immediately(self):
        """Test that non-retryable errors fail immediately without retry."""
        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
            max_retries=3,
        )

        mock_create = Mock(side_effect=ValueError("Bad input"))

        with patch.object(client.client.embeddings, "create", mock_create):
            with patch("time.sleep"):
                with pytest.raises(ValueError, match="Bad input"):
                    client.get_embeddings(texts=["test"])

                # Should fail on first attempt - no retries
                assert mock_create.call_count == 1

    def test_get_embeddings_timeout_handling(self):
        """Test timeout handling for embeddings."""
        from citation_index.llm.client import LLMTimeoutError

        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
            timeout=5.0,
        )

        mock_create = Mock(side_effect=LLMTimeoutError("Request timed out"))

        with patch.object(client.client.embeddings, "create", mock_create):
            with patch("time.sleep"):
                with pytest.raises(LLMTimeoutError):
                    client.get_embeddings(texts=["test"])

    def test_get_embeddings_empty_input(self):
        """Test handling empty text list."""
        client = EmbedClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key=self.mock_api_key,
        )

        # OpenAI SDK should handle empty list or raise error
        # Test that we don't crash
        mock_response = Mock()
        mock_response.data = []
        mock_create = Mock(return_value=mock_response)

        with patch.object(client.client.embeddings, "create", mock_create):
            embeddings = client.get_embeddings(texts=[])
            assert embeddings.shape[0] == 0
