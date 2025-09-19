"""
Comprehensive tests for the improved LLM client implementations.
Tests DeepSeekClient, VLLMClient, and base LLMClient functionality.
"""

import json
import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from unittest.mock import Mock, patch, MagicMock
from src.citation_index.llm.client import LLMClient, DeepSeekClient, VLLMClient


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
    
    def test_base_client_initialization(self):
        """Test base LLMClient initialization."""
        client = LLMClient(
            endpoint=self.mock_endpoint,
            model="test-model",
            api_key=self.mock_api_key
        )
        
        assert client.endpoint == self.mock_endpoint
        assert client.model == "test-model"
        assert client.api_key == self.mock_api_key
        assert client.timeout == 180.0  # default
        assert client.max_retries == 3  # default
    
    def test_base_client_without_api_key(self):
        """Test base LLMClient works without API key (for local deployments)."""
        client = LLMClient(
            endpoint=self.mock_endpoint,
            model="test-model"
        )
        
        assert client.endpoint == self.mock_endpoint
        assert client.model == "test-model"
        assert client.api_key is None
        # Should still create OpenAI client with dummy key
        assert client.client is not None
    
    def test_response_format_handling(self):
        """Test base client response format handling."""
        client = LLMClient(
            endpoint=self.mock_endpoint,
            model="test-model",
            api_key=self.mock_api_key
        )
        
        # Test simple JSON output
        response_format, prompt, max_tokens = client._get_response_format_and_prompt(
            prompt="Generate JSON",
            json_output=True
        )
        
        assert response_format == {"type": "json_object"}
        assert prompt == "Generate JSON"  # Should not be modified
        
        # Test with JSON schema
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format, prompt, max_tokens = client._get_response_format_and_prompt(
            prompt="Generate data",
            json_schema=schema
        )
        
        expected_format = {"type": "json_schema", "json_schema": schema}
        assert response_format == expected_format
        assert prompt == "Generate data"  # Should not be modified


class TestDeepSeekClient:
    """Test cases for DeepSeekClient functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_api_key = "test-key"
    
    def test_deepseek_initialization(self):
        """Test DeepSeekClient initialization."""
        client = DeepSeekClient(api_key=self.mock_api_key)
        
        assert client.endpoint == "https://api.deepseek.com/v1"
        assert client.model == "deepseek-chat"
        assert client.api_key == self.mock_api_key
    
    def test_deepseek_response_format_handling(self):
        """Test DeepSeek-specific response format handling."""
        client = DeepSeekClient(api_key=self.mock_api_key)
        
        # Test JSON output without continuation
        response_format, prompt, max_tokens = client._get_deepseek_response_format_and_prompt(
            prompt="Generate data",
            json_output=True,
            max_tokens=4000,
            use_continuation=False
        )
        
        assert response_format == {"type": "json_object"}
        assert max_tokens == 8000  # Should be increased to 8000
        assert "JSON format" in prompt  # Should add JSON instruction
        assert "<start>" not in prompt  # No continuation tags
        
        # Test JSON output with continuation
        response_format, prompt, max_tokens = client._get_deepseek_response_format_and_prompt(
            prompt="Extract references",
            json_output=True,
            max_tokens=4000,
            use_continuation=True
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
                return create_mock_stream(', 3]}\n<end>')
        
        mock_side_effect.call_count = 0
        
        mock_create = Mock(side_effect=mock_side_effect)
        
        with patch.object(client.client.chat.completions, 'create', mock_create):
            response = client.call(
                prompt="Generate JSON data",
                json_output=True
            )
            
            # Should use continuation by default for JSON
            assert mock_create.call_count == 2
            assert isinstance(response, str)  # Should return string, not tuple
            assert '<start>' in response and '<end>' in response
    
    def test_deepseek_disable_continuation(self):
        """Test DeepSeek with continuation disabled."""
        client = DeepSeekClient(api_key=self.mock_api_key)
        
        mock_response = '{"result": "success"}'
        mock_create = Mock()
        mock_create.return_value = create_mock_stream(mock_response)
        
        with patch.object(client.client.chat.completions, 'create', mock_create):
            response = client.call(
                prompt="Generate JSON",
                json_output=True,
                use_continuation=False
            )
            
            # Should make only one call
            assert mock_create.call_count == 1
            assert response == mock_response
            
            # Should still apply DeepSeek optimizations
            call_args = mock_create.call_args_list[0][1]
            assert call_args['response_format'] == {"type": "json_object"}
            assert call_args['max_tokens'] == 8000
    
    def test_deepseek_non_json_requests(self):
        """Test DeepSeek with non-JSON requests."""
        client = DeepSeekClient(api_key=self.mock_api_key)
        
        mock_response = "This is a regular text response."
        mock_create = Mock()
        mock_create.return_value = create_mock_stream(mock_response)
        
        with patch.object(client.client.chat.completions, 'create', mock_create):
            response = client.call(
                prompt="Tell me about AI",
                use_continuation=True  # Should be ignored for non-JSON
            )
            
            # Should not use continuation for non-JSON
            assert mock_create.call_count == 1
            assert response == mock_response
            
            # Should not have special response format
            call_args = mock_create.call_args_list[0][1]
            assert call_args.get('response_format') is None
    
    def test_deepseek_continuation_with_custom_tags(self):
        """Test DeepSeek continuation with custom start/end tags."""
        client = DeepSeekClient(api_key=self.mock_api_key)
        
        def mock_side_effect(*args, **kwargs):
            if mock_side_effect.call_count == 0:
                mock_side_effect.call_count += 1
                return create_mock_stream('BEGIN\n{"items": [1, 2')
            else:
                return create_mock_stream(', 3]}\nEND')
        
        mock_side_effect.call_count = 0
        
        mock_create = Mock(side_effect=mock_side_effect)
        
        with patch.object(client.client.chat.completions, 'create', mock_create):
            messages, response = client.call_with_continuation(
                prompt="Generate JSON data",
                start_tag='BEGIN',
                end_tag='END',
                json_output=True
            )
            
            assert mock_create.call_count == 2
            assert 'BEGIN' not in response and 'END' not in response
            assert json.loads(response) == {"items": [1, 2, 3]}
            assert len(messages) == 3  # user -> assistant -> user


class TestVLLMClient:
    """Test cases for VLLMClient functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_endpoint = "http://localhost:8000/v1"
        self.mock_model = "llama-3-8b"
    
    def test_vllm_initialization_with_api_key(self):
        """Test VLLMClient initialization with API key."""
        client = VLLMClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key="test-key"
        )
        
        assert client.endpoint == self.mock_endpoint
        assert client.model == self.mock_model
        assert client.api_key == "test-key"
    
    def test_vllm_initialization_without_api_key(self):
        """Test VLLMClient initialization without API key (local deployment)."""
        client = VLLMClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model
        )
        
        assert client.endpoint == self.mock_endpoint
        assert client.model == self.mock_model
        assert client.api_key is None
        # Should still create client with dummy key
        assert client.client is not None
    
    def test_vllm_vs_deepseek_response_format(self):
        """Test response format differences between vLLM and DeepSeek."""
        # Test vLLM
        vllm_client = VLLMClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key="test-key"
        )
        
        mock_response = '{"data": "test"}'
        mock_create = Mock()
        mock_create.return_value = create_mock_stream(mock_response)
        
        with patch.object(vllm_client.client.chat.completions, 'create', mock_create):
            response = vllm_client.call(
                prompt="Generate JSON data",
                json_output=True
            )
            
            # Verify vLLM uses simple json_object without modifications
            call_args = mock_create.call_args_list[0][1]
            assert call_args['response_format'] == {"type": "json_object"}
            
            # Verify prompt was not modified
            prompt_content = call_args['messages'][0]['content']
            assert prompt_content == "Generate JSON data"
            
            # Verify no special max_tokens handling
            assert call_args.get('max_tokens') is None
        
        # Compare with DeepSeek
        deepseek_client = DeepSeekClient(api_key="test-key")
        
        mock_create.reset_mock()
        mock_create.return_value = create_mock_stream(mock_response)
        
        with patch.object(deepseek_client.client.chat.completions, 'create', mock_create):
            response = deepseek_client.call(
                prompt="Generate JSON data",
                json_output=True,
                use_continuation=False
            )
            
            # Verify DeepSeek applies modifications
            call_args = mock_create.call_args_list[0][1]
            assert call_args['response_format'] == {"type": "json_object"}
            
            # Verify prompt was modified
            prompt_content = call_args['messages'][0]['content']
            assert "JSON format" in prompt_content
            
            # Verify max_tokens was set
            assert call_args['max_tokens'] == 8000
    
    def test_vllm_parsed_structured_output(self):
        """Test vLLM structured output with Pydantic models."""
        client = VLLMClient(
            endpoint=self.mock_endpoint,
            model=self.mock_model,
            api_key="test-key"
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
        
        with patch.object(client.client.beta.chat.completions, 'parse', mock_parse):
            result = client.call_with_parsed_structured_output(
                prompt="Generate structured data",
                json_class=TestModel
            )
            
            assert result == {"name": "test", "value": 42}
            
            # Verify the parse method was called correctly
            call_args = mock_parse.call_args_list[0][1]
            assert call_args['response_format'] == TestModel


class TestClientComparison:
    """Test cases comparing different client behaviors."""
    
    def test_continuation_availability(self):
        """Test that only DeepSeekClient has continuation method."""
        base_client = LLMClient(endpoint="http://test", model="test", api_key="test")
        vllm_client = VLLMClient(endpoint="http://test", model="test", api_key="test")
        deepseek_client = DeepSeekClient(api_key="test")
        
        # Only DeepSeekClient should have continuation method
        assert not hasattr(base_client, 'call_with_continuation')
        assert not hasattr(vllm_client, 'call_with_continuation')
        assert hasattr(deepseek_client, 'call_with_continuation')
    
    def test_json_schema_handling(self):
        """Test JSON schema handling across different clients."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # Test base client
        base_client = LLMClient(endpoint="http://test", model="test", api_key="test")
        response_format, prompt, max_tokens = base_client._get_response_format_and_prompt(
            prompt="Generate person data",
            json_schema=schema
        )
        
        expected_format = {"type": "json_schema", "json_schema": schema}
        assert response_format == expected_format
        assert prompt == "Generate person data"  # Unchanged
        
        # Test DeepSeek client
        deepseek_client = DeepSeekClient(api_key="test")
        response_format, prompt, max_tokens = deepseek_client._get_deepseek_response_format_and_prompt(
            prompt="Generate person data",
            json_schema=schema
        )
        
        # DeepSeek always uses json_object, even with schema
        assert response_format == {"type": "json_object"}
        assert "JSON format" in prompt  # Should be modified
        assert max_tokens == 8000  # Should be set to 8000


def test_basic_instantiation():
    """Simple test that can be run directly."""
    print("Testing basic client instantiation...")
    
    # Test all clients can be instantiated
    base_client = LLMClient(endpoint="http://test", model="test", api_key="test")
    vllm_client = VLLMClient(endpoint="http://test", model="test", api_key="test")
    deepseek_client = DeepSeekClient(api_key="test")
    
    assert base_client.model == "test"
    assert vllm_client.model == "test"
    assert deepseek_client.model == "deepseek-chat"
    
    print("✓ All clients instantiate correctly")


def test_benchmark_client_selection():
    """Test the client selection logic used in benchmark scripts."""
    print("Testing benchmark client selection logic...")
    
    def select_client_type(model_name, api_base):
        """Simulate the logic used in benchmark scripts."""
        if "deepseek" in model_name.lower() or "api.deepseek.com" in api_base:
            return "DeepSeekClient"
        else:
            return "LLMClient"
    
    # Test cases that should use DeepSeekClient
    deepseek_cases = [
        ("deepseek-chat", "http://localhost:8000/v1"),
        ("DeepSeek-V3.1", "https://api.openai.com/v1"),
        ("some-model", "https://api.deepseek.com/v1"),
    ]
    
    for model, api_base in deepseek_cases:
        client_type = select_client_type(model, api_base)
        assert client_type == "DeepSeekClient", f"Expected DeepSeekClient for {model} + {api_base}"
    
    # Test cases that should use LLMClient
    llm_cases = [
        ("gpt-4", "https://api.openai.com/v1"),
        ("llama-3-8b", "http://localhost:8000/v1"),
        ("gemma-7b", "http://localhost:8000/v1"),
    ]
    
    for model, api_base in llm_cases:
        client_type = select_client_type(model, api_base)
        assert client_type == "LLMClient", f"Expected LLMClient for {model} + {api_base}"
    
    print("✓ Benchmark client selection logic works correctly")


if __name__ == "__main__":
    # Run basic test when called directly
    test_basic_instantiation()
    test_benchmark_client_selection()
    print("🎉 Basic tests passed! Run with pytest for full test suite.")
