"""
Comprehensive tests for the refactored PromptLoader system.

Tests both legacy .md format and new YAML format with Jinja2 templates.
Covers corner cases, error handling, and backward compatibility.

Run with: pytest tests/test_prompt_loader.py -v
Or: conda run -n citation_index pytest tests/test_prompt_loader.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from citation_index.llm.prompt_loader import (
    PromptLoader,
    ReferenceExtractionPrompt,
    ReferenceParsingPrompt,
    ReferenceExtractionAndParsingPrompt,
)


class TestLegacyMarkdownFormat:
    """Test backward compatibility with existing .md prompt files."""
    
    def test_md_file_detection(self):
        """Verify .md files are correctly detected as markdown format."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test"
        )
        assert prompt_obj._format == "markdown"
    
    def test_md_prompt_loads_correctly(self):
        """Test that .md prompts load and return valid string content."""
        input_text = "1. Test Reference (2020). Title. Journal."
        
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text=input_text,
            include_json_schema=False
        )
        
        # Should return a non-empty string
        assert isinstance(prompt_obj.prompt, str)
        assert len(prompt_obj.prompt) > 0
        
        # Should contain expected content
        assert "expert" in prompt_obj.prompt.lower()
        assert "parse" in prompt_obj.prompt.lower()
    
    def test_md_with_json_schema_injection(self):
        """Test that JSON schema is correctly injected into .md prompts."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test",
            include_json_schema=True
        )
        
        # Schema should be injected
        assert prompt_obj.json_schema is not None
        assert isinstance(prompt_obj.json_schema, dict)
    
    def test_md_extraction_prompt(self):
        """Test extraction-specific .md prompt loading."""
        input_text = "References: 1. Smith (2020). Paper. Journal."
        
        prompt_obj = ReferenceExtractionPrompt(
            prompt="prompts/reference_extraction.md",
            input_text=input_text
        )
        
        assert prompt_obj._format == "markdown"
        assert isinstance(prompt_obj.prompt, str)
    
    def test_md_extraction_and_parsing_prompt(self):
        """Test combined extraction+parsing .md prompt."""
        input_text = "Test document with references."
        
        prompt_obj = ReferenceExtractionAndParsingPrompt(
            prompt="prompts/reference_extraction_and_parsing.md",
            input_text=input_text,
            include_json_schema=True
        )
        
        assert prompt_obj._format == "markdown"
        assert prompt_obj.json_schema is not None


class TestYAMLFormatBasics:
    """Test basic YAML format functionality."""
    
    def test_yaml_file_detection(self):
        """Verify .yaml files are correctly detected."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test"
        )
        assert prompt_obj._format == "yaml"
    
    def test_yaml_system_user_separation(self):
        """Test that YAML prompts return separated system and user messages."""
        input_text = "1. Test Ref (2020). Title."
        
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=input_text,
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Should have both system and user keys
        assert "system" in messages
        assert "user" in messages
        
        # Both should be non-empty strings
        assert isinstance(messages["system"], str)
        assert isinstance(messages["user"], str)
        assert len(messages["system"]) > 0
        assert len(messages["user"]) > 0
        
        # System should contain expert/task description
        assert "expert" in messages["system"].lower()
        
        # User should contain the actual task
        assert "task" in messages["user"].lower() or "parse" in messages["user"].lower()
    
    def test_yaml_input_text_injection(self):
        """Verify that input_text is properly injected into YAML prompts."""
        input_text = "UNIQUE_TEST_STRING_12345"
        
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=input_text,
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Input text should appear in user message
        assert input_text in messages["user"]
        # But not in system message
        assert input_text not in messages["system"]
    
    def test_yaml_namespace_keys(self):
        """Test that different namespace keys load different prompts."""
        input_text = "test"
        
        # Test parsing.default
        parsing_prompt = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=input_text,
            include_json_schema=False
        )
        
        # Test extraction.default
        extraction_prompt = ReferenceExtractionPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="extraction.default",
            input_text=input_text
        )
        
        # Test extraction_and_parsing.default
        combined_prompt = ReferenceExtractionAndParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="extraction_and_parsing.default",
            input_text=input_text,
            include_json_schema=False
        )
        
        # All should load successfully
        assert parsing_prompt.messages["system"] != extraction_prompt.messages["system"]
        assert extraction_prompt.messages["system"] != combined_prompt.messages["system"]


class TestJinja2Templating:
    """Test Jinja2 template rendering functionality."""
    
    def test_examples_rendering(self):
        """Test that examples are rendered in the user prompt."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Examples should be rendered in user message
        assert "Example" in messages["user"]
        # Check for example content (from prompts.yaml)
        assert "Smith" in messages["user"] or "example" in messages["user"].lower()
    
    def test_json_schema_injection_yaml(self):
        """Test JSON schema injection in YAML prompts via Jinja2."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=True
        )
        
        messages = prompt_obj.messages
        
        # Schema should be present in user message
        assert "properties" in messages["user"] or "Reference" in messages["user"]
        # Should contain schema markers
        assert "{" in messages["user"]
    
    def test_conditional_rendering(self):
        """Test Jinja2 conditional blocks (if examples)."""
        # The YAML prompts use {% if examples %} blocks
        # When examples exist, they should render
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Should have examples section since parsing.default has examples
        assert "Example" in messages["user"]
    
    def test_loop_rendering(self):
        """Test Jinja2 loop rendering (for ex in examples)."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # parsing.default has 1 example with specific content
        assert "Smith" in messages["user"]
        assert "United Nations" in messages["user"]


class TestExternalExamplesFiles:
    """Test loading examples from external JSON/YAML files."""
    
    def test_external_json_examples(self, tmp_path):
        """Test loading examples from an external JSON file."""
        # Create temporary JSON examples file
        examples_data = [
            {"input": "Test input 1", "output": "Test output 1"},
            {"input": "Test input 2", "output": "Test output 2"}
        ]
        examples_file = tmp_path / "examples.json"
        with open(examples_file, 'w') as f:
            json.dump(examples_data, f)
        
        # Create temporary YAML prompt that references external examples
        yaml_content = {
            "test.external": {
                "system": "System prompt",
                "user": "{% for ex in examples %}{{ ex.input }}{% endfor %}\n{{ input_text }}",
                "examples": str(examples_file)
            }
        }
        yaml_file = tmp_path / "test_prompts.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        # Load the prompt
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.external",
            input_text="MY_INPUT"
        )
        
        messages = prompt_obj.messages
        
        # External examples should be loaded and rendered
        assert "Test input 1" in messages["user"]
        assert "Test input 2" in messages["user"]
        assert "MY_INPUT" in messages["user"]
    
    def test_external_yaml_examples(self, tmp_path):
        """Test loading examples from an external YAML file."""
        # Create temporary YAML examples file
        examples_data = [
            {"input": "YAML input 1", "output": "YAML output 1"},
            {"input": "YAML input 2", "output": "YAML output 2"}
        ]
        examples_file = tmp_path / "examples.yml"
        with open(examples_file, 'w') as f:
            yaml.dump(examples_data, f)
        
        # Create temporary YAML prompt
        yaml_content = {
            "test.yaml_external": {
                "system": "System",
                "user": "{% for ex in examples %}{{ ex.input }}|{% endfor %}{{ input_text }}",
                "examples": str(examples_file)
            }
        }
        yaml_file = tmp_path / "test.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.yaml_external",
            input_text="INPUT"
        )
        
        messages = prompt_obj.messages
        
        assert "YAML input 1" in messages["user"]
        assert "YAML input 2" in messages["user"]
    
    def test_inline_examples(self, tmp_path):
        """Test inline examples (list in YAML) vs external file."""
        yaml_content = {
            "test.inline": {
                "system": "System",
                "user": "{% for ex in examples %}{{ ex.input }}{% endfor %}",
                "examples": [
                    {"input": "inline1", "output": "out1"},
                    {"input": "inline2", "output": "out2"}
                ]
            }
        }
        yaml_file = tmp_path / "inline.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.inline",
            input_text="test"
        )
        
        messages = prompt_obj.messages
        
        assert "inline1" in messages["user"]
        assert "inline2" in messages["user"]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_prompt_key_with_no_default(self):
        """Test error when prompt_key not found and no default exists."""
        with pytest.raises(ValueError) as exc_info:
            prompt_obj = ReferenceParsingPrompt(
                prompt="prompts/prompts.yaml",
                # Don't specify prompt_key - should try to find "default"
                input_text="test",
                include_json_schema=False
            )
            # Trigger loading
            _ = prompt_obj.messages
        
        # Error message should mention available keys
        assert "Available keys" in str(exc_info.value)
        assert "parsing.default" in str(exc_info.value)
    
    def test_invalid_prompt_key(self):
        """Test error when specifying a non-existent prompt_key."""
        with pytest.raises(ValueError) as exc_info:
            prompt_obj = ReferenceParsingPrompt(
                prompt="prompts/prompts.yaml",
                prompt_key="nonexistent.key",
                input_text="test",
                include_json_schema=False
            )
            _ = prompt_obj.messages
        
        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.key" in str(exc_info.value)
    
    def test_nonexistent_file(self):
        """Test error when prompt file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            prompt_obj = ReferenceParsingPrompt(
                prompt="prompts/does_not_exist.md",
                input_text="test"
            )
            _ = prompt_obj.prompt
    
    def test_empty_input_text(self):
        """Test handling of empty input text."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="",  # Empty string
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Should still work, just with empty input
        assert "system" in messages
        assert "user" in messages
    
    def test_very_long_input_text(self):
        """Test handling of very long input text."""
        # Create a very long input (10000 characters)
        long_input = "Test reference. " * 1000
        
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=long_input,
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Should contain the long input
        assert long_input in messages["user"]
    
    def test_special_characters_in_input(self):
        """Test handling of special characters and unicode."""
        special_input = """
        References with special chars: <>&"'
        Unicode: café, 日本語, émigré
        Newlines and tabs:\t\n\tindented
        """
        
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=special_input,
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # Special characters should be preserved
        assert "café" in messages["user"]
        assert "日本語" in messages["user"]


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def test_prompt_property_on_yaml(self):
        """Test that .prompt property works on YAML format (concatenates system+user)."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=False
        )
        
        # .prompt should return a string (backward compat)
        prompt_str = prompt_obj.prompt
        
        assert isinstance(prompt_str, str)
        assert len(prompt_str) > 0
        
        # Should contain both system and user content
        messages = prompt_obj.messages
        assert messages["system"] in prompt_str
        assert messages["user"] in prompt_str
    
    def test_prompt_property_on_markdown(self):
        """Test that .prompt property works on .md format."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test",
            include_json_schema=False
        )
        
        prompt_str = prompt_obj.prompt
        
        assert isinstance(prompt_str, str)
        assert len(prompt_str) > 0
    
    def test_messages_on_markdown(self):
        """Test that .messages works on .md format (returns empty system, content in user)."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test",
            include_json_schema=False
        )
        
        messages = prompt_obj.messages
        
        # For markdown, system should be empty
        assert messages["system"] == ""
        # All content in user
        assert len(messages["user"]) > 0
        assert messages["user"] == prompt_obj.prompt
    
    def test_raw_string_prompt(self):
        """Test using raw string as prompt (no file)."""
        raw_prompt = "This is a raw prompt string with {{ input_text }}"
        
        prompt_obj = PromptLoader(
            prompt=raw_prompt,
            input_text="MY_INPUT"
        )
        
        # Should be detected as raw format
        assert prompt_obj._format == "raw"
        
        # Should return the raw string
        assert prompt_obj.prompt == raw_prompt


class TestSubclassSpecificBehavior:
    """Test behavior specific to each prompt subclass."""
    
    def test_reference_extraction_prompt_defaults(self):
        """Test ReferenceExtractionPrompt default parameters."""
        prompt_obj = ReferenceExtractionPrompt(input_text="test")
        
        # Should default to .md file
        assert "reference_extraction.md" in prompt_obj.prompt_path
    
    def test_reference_parsing_prompt_defaults(self):
        """Test ReferenceParsingPrompt default parameters."""
        prompt_obj = ReferenceParsingPrompt(input_text="test")
        
        # Should default to .md file
        assert "reference_parsing.md" in prompt_obj.prompt_path
    
    def test_reference_extraction_and_parsing_defaults(self):
        """Test ReferenceExtractionAndParsingPrompt default parameters."""
        prompt_obj = ReferenceExtractionAndParsingPrompt(input_text="test")
        
        # Should default to .md file
        assert "reference_extraction_and_parsing" in prompt_obj.prompt_path
    
    def test_json_schema_loading(self):
        """Test that JSON schema is properly loaded from References model."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=True
        )
        
        # Schema should be loaded
        assert prompt_obj.json_schema is not None
        assert isinstance(prompt_obj.json_schema, dict)
        
        # Should contain expected schema keys
        assert "$defs" in prompt_obj.json_schema or "properties" in prompt_obj.json_schema


class TestFormatDetection:
    """Test automatic format detection logic."""
    
    def test_detect_yaml_extension(self):
        """Test .yaml extension detection."""
        prompt_obj = PromptLoader(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test"
        )
        assert prompt_obj._format == "yaml"
    
    def test_detect_yml_extension(self):
        """Test .yml extension detection."""
        # Create temp .yml file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({"test": {"system": "S", "user": "U"}}, f)
            yml_file = f.name
        
        try:
            prompt_obj = PromptLoader(
                prompt=yml_file,
                prompt_key="test",
                input_text="test"
            )
            assert prompt_obj._format == "yaml"
        finally:
            os.unlink(yml_file)
    
    def test_detect_md_extension(self):
        """Test .md extension detection."""
        prompt_obj = PromptLoader(
            prompt="prompts/reference_parsing.md",
            input_text="test"
        )
        assert prompt_obj._format == "markdown"
    
    def test_detect_txt_extension(self):
        """Test .txt extension detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test prompt content")
            txt_file = f.name
        
        try:
            prompt_obj = PromptLoader(prompt=txt_file, input_text="test")
            assert prompt_obj._format == "markdown"
        finally:
            os.unlink(txt_file)


class TestEdgeCasesAndCornerCases:
    """Test edge cases and corner cases."""
    
    def test_prompt_with_only_system_message(self, tmp_path):
        """Test YAML prompt with system but empty user message."""
        yaml_content = {
            "test.only_system": {
                "system": "System message only",
                "user": ""  # Empty user
            }
        }
        yaml_file = tmp_path / "edge.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.only_system",
            input_text="test"
        )
        
        messages = prompt_obj.messages
        assert messages["system"] == "System message only"
        assert messages["user"] == ""
    
    def test_prompt_with_only_user_message(self, tmp_path):
        """Test YAML prompt with user but empty system message."""
        yaml_content = {
            "test.only_user": {
                "system": "",
                "user": "User message only: {{ input_text }}"
            }
        }
        yaml_file = tmp_path / "edge2.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.only_user",
            input_text="INPUT"
        )
        
        messages = prompt_obj.messages
        assert messages["system"] == ""
        assert "User message only: INPUT" in messages["user"]
    
    def test_no_examples_in_yaml(self, tmp_path):
        """Test YAML prompt without examples key."""
        yaml_content = {
            "test.no_examples": {
                "system": "System",
                "user": "{% if examples %}Has examples{% else %}No examples{% endif %}"
            }
        }
        yaml_file = tmp_path / "no_ex.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.no_examples",
            input_text="test"
        )
        
        messages = prompt_obj.messages
        # examples should be empty list
        assert "No examples" in messages["user"]
    
    def test_empty_examples_list(self, tmp_path):
        """Test YAML prompt with empty examples list."""
        yaml_content = {
            "test.empty_list": {
                "system": "System",
                "user": "{% if examples %}Has{% else %}Empty{% endif %}",
                "examples": []
            }
        }
        yaml_file = tmp_path / "empty.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.empty_list",
            input_text="test"
        )
        
        messages = prompt_obj.messages
        # Empty list is falsy in Jinja2
        assert "Empty" in messages["user"]
    
    def test_multiple_namespaces(self, tmp_path):
        """Test loading from file with many namespace keys."""
        yaml_content = {
            "task1.variant1": {"system": "T1V1", "user": "U1"},
            "task1.variant2": {"system": "T1V2", "user": "U2"},
            "task2.variant1": {"system": "T2V1", "user": "U3"},
            "task2.variant2": {"system": "T2V2", "user": "U4"},
        }
        yaml_file = tmp_path / "multi.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        # Load each variant
        p1 = PromptLoader(prompt=str(yaml_file), prompt_key="task1.variant1", input_text="")
        p2 = PromptLoader(prompt=str(yaml_file), prompt_key="task1.variant2", input_text="")
        p3 = PromptLoader(prompt=str(yaml_file), prompt_key="task2.variant1", input_text="")
        p4 = PromptLoader(prompt=str(yaml_file), prompt_key="task2.variant2", input_text="")
        
        assert p1.messages["system"] == "T1V1"
        assert p2.messages["system"] == "T1V2"
        assert p3.messages["system"] == "T2V1"
        assert p4.messages["system"] == "T2V2"
    
    def test_nested_jinja2_variables(self, tmp_path):
        """Test complex Jinja2 template with nested structures."""
        yaml_content = {
            "test.complex": {
                "system": "System",
                "user": """
{% if examples %}
Examples:
{% for ex in examples %}
  - Input: {{ ex.input }}
    Output: {{ ex.output }}
    {% if ex.note %}Note: {{ ex.note }}{% endif %}
{% endfor %}
{% endif %}
Input: {{ input_text }}
""",
                "examples": [
                    {"input": "I1", "output": "O1", "note": "N1"},
                    {"input": "I2", "output": "O2"}  # No note
                ]
            }
        }
        yaml_file = tmp_path / "complex.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        prompt_obj = PromptLoader(
            prompt=str(yaml_file),
            prompt_key="test.complex",
            input_text="MY_INPUT"
        )
        
        messages = prompt_obj.messages
        
        # Should have rendered nested template
        assert "I1" in messages["user"]
        assert "O1" in messages["user"]
        assert "N1" in messages["user"]
        assert "I2" in messages["user"]
        assert "O2" in messages["user"]
        assert "MY_INPUT" in messages["user"]


# Example usage when run directly (not with pytest)
if __name__ == "__main__":
    print("=" * 80)
    print("PROMPT LOADER TEST SUITE - EXAMPLE MODE")
    print("=" * 80)
    print("\nThis file is designed to run with pytest:")
    print("  pytest tests/test_prompt_loader.py -v\n")
    print("Running a few example tests manually...\n")
    
    # Example 1: Test legacy .md format
    print("📝 Example 1: Legacy .md format")
    print("-" * 40)
    prompt = ReferenceParsingPrompt(
        prompt="prompts/reference_parsing.md",
        input_text="1. Smith, J. (2020). Test. Journal."
    )
    print(f"Format: {prompt._format}")
    print(f"Prompt length: {len(prompt.prompt)} chars")
    print(f"First 150 chars: {prompt.prompt[:150]}...\n")
    
    # Example 2: Test YAML with system/user
    print("📝 Example 2: YAML format with system/user separation")
    print("-" * 40)
    prompt = ReferenceParsingPrompt(
        prompt="prompts/prompts.yaml",
        prompt_key="parsing.default",
        input_text="1. Test Reference (2020). Title.",
        include_json_schema=False
    )
    msgs = prompt.messages
    print(f"Format: {prompt._format}")
    print(f"System: {msgs['system'][:100]}...")
    print(f"User (first 200 chars): {msgs['user'][:200]}...\n")
    
    # Example 3: Test JSON schema injection
    print("📝 Example 3: JSON schema injection in YAML")
    print("-" * 40)
    prompt = ReferenceParsingPrompt(
        prompt="prompts/prompts.yaml",
        prompt_key="parsing.default",
        input_text="Test",
        include_json_schema=True
    )
    msgs = prompt.messages
    has_schema = "properties" in msgs["user"] or "$defs" in msgs["user"]
    print(f"Schema injected: {has_schema}")
    print(f"User prompt length: {len(msgs['user'])} chars\n")
    
    # Example 4: Test error handling
    print("📝 Example 4: Error handling - invalid prompt_key")
    print("-" * 40)
    try:
        prompt = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="nonexistent.key",
            input_text="Test"
        )
        _ = prompt.messages
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:100]}...\n")
    
    print("=" * 80)
    print("Examples completed! Run with pytest for full test coverage.")
    print("=" * 80)
