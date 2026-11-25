"""
Comprehensive tests for the refactored PromptLoader system.

Tests both legacy .md format and new YAML format with Jinja2 templates.
Covers corner cases, error handling, and backward compatibility.

Run with: pytest tests/test_prompt_loader.py -v
Or: conda run -n citation_index pytest tests/test_prompt_loader.py -v
"""

import json

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

    def test_md_with_json_schema_injection(self):
        """Test that JSON schema is correctly injected into .md prompts."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test",
            include_json_schema=True,
        )

        # Schema should be injected
        assert prompt_obj.json_schema is not None
        assert isinstance(prompt_obj.json_schema, dict)


class TestYAMLFormatBasics:
    """Test basic YAML format functionality."""

    def test_yaml_system_user_separation(self):
        """Test that YAML prompts return separated system and user messages."""
        input_text = "1. Test Ref (2020). Title."

        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=input_text,
            include_json_schema=False,
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

    def test_yaml_namespace_keys(self):
        """Test that different namespace keys load different prompts."""
        input_text = "test"

        # Test parsing.default
        parsing_prompt = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text=input_text,
            include_json_schema=False,
        )

        # Test extraction.default
        extraction_prompt = ReferenceExtractionPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="extraction.default",
            input_text=input_text,
        )

        # Test extraction_and_parsing.default
        combined_prompt = ReferenceExtractionAndParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="extraction_and_parsing.default",
            input_text=input_text,
            include_json_schema=False,
        )

        # All should load successfully
        assert parsing_prompt.messages["system"] != extraction_prompt.messages["system"]
        assert (
            extraction_prompt.messages["system"] != combined_prompt.messages["system"]
        )


class TestJinja2Templating:
    """Test Jinja2 template rendering functionality."""

    def test_examples_rendering(self):
        """Test that examples are rendered in the user prompt."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=False,
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
            include_json_schema=True,
        )

        messages = prompt_obj.messages

        # Schema should be present in user message
        assert "properties" in messages["user"] or "Reference" in messages["user"]
        # Should contain schema markers
        assert "{" in messages["user"]


class TestExternalExamplesFiles:
    """Test loading examples from external JSON/YAML files."""

    def test_external_json_examples(self, tmp_path):
        """Test loading examples from an external JSON file."""
        # Create temporary JSON examples file
        examples_data = [
            {"input": "Test input 1", "output": "Test output 1"},
            {"input": "Test input 2", "output": "Test output 2"},
        ]
        examples_file = tmp_path / "examples.json"
        with open(examples_file, "w") as f:
            json.dump(examples_data, f)

        # Create temporary YAML prompt that references external examples
        yaml_content = {
            "test.external": {
                "system": "System prompt",
                "user": "{% for ex in examples %}{{ ex.input }}{% endfor %}\n{{ input_text }}",
                "examples": str(examples_file),
            }
        }
        yaml_file = tmp_path / "test_prompts.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Load the prompt
        prompt_obj = PromptLoader(
            prompt=str(yaml_file), prompt_key="test.external", input_text="MY_INPUT"
        )

        messages = prompt_obj.messages

        # External examples should be loaded and rendered
        assert "Test input 1" in messages["user"]
        assert "Test input 2" in messages["user"]
        assert "MY_INPUT" in messages["user"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_prompt_key_with_no_default(self):
        """Test error when prompt_key not found and no default exists."""
        with pytest.raises(ValueError) as exc_info:
            prompt_obj = ReferenceParsingPrompt(
                prompt="prompts/prompts.yaml",
                # Don't specify prompt_key - should try to find "default"
                input_text="test",
                include_json_schema=False,
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
                include_json_schema=False,
            )
            _ = prompt_obj.messages

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent.key" in str(exc_info.value)

    def test_nonexistent_file(self):
        """Test error when prompt file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            prompt_obj = ReferenceParsingPrompt(
                prompt="prompts/does_not_exist.md", input_text="test"
            )
            _ = prompt_obj.prompt


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_messages_on_markdown(self):
        """Test that .messages works on .md format (returns empty system, content in user)."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/reference_parsing.md",
            input_text="test",
            include_json_schema=False,
        )

        messages = prompt_obj.messages

        # For markdown, system should be empty
        assert messages["system"] == ""
        # All content in user
        assert len(messages["user"]) > 0
        assert messages["user"] == prompt_obj.prompt


class TestSubclassSpecificBehavior:
    """Test behavior specific to each prompt subclass."""

    def test_json_schema_loading(self):
        """Test that JSON schema is properly loaded from References model."""
        prompt_obj = ReferenceParsingPrompt(
            prompt="prompts/prompts.yaml",
            prompt_key="parsing.default",
            input_text="test",
            include_json_schema=True,
        )

        # Schema should be loaded
        assert prompt_obj.json_schema is not None
        assert isinstance(prompt_obj.json_schema, dict)

        # Should contain expected schema keys
        assert (
            "$defs" in prompt_obj.json_schema or "properties" in prompt_obj.json_schema
        )


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
        input_text="1. Smith, J. (2020). Test. Journal.",
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
        include_json_schema=False,
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
        include_json_schema=True,
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
            input_text="Test",
        )
        _ = prompt.messages
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:100]}...\n")

    print("=" * 80)
    print("Examples completed! Run with pytest for full test coverage.")
    print("=" * 80)


@pytest.mark.parametrize(
    "prompt_cls,prompt_name",
    [
        (ReferenceExtractionPrompt, "prompts/reference_extraction.md"),
        (ReferenceExtractionPrompt, "prompts/prompts.yaml:extraction.default"),
        (ReferenceParsingPrompt, "prompts/reference_parsing.md"),
        (ReferenceParsingPrompt, "prompts/prompts.yaml:parsing.default"),
        (ReferenceParsingPrompt, "prompts/prompts.yaml:metadata.default"),
        (ReferenceExtractionAndParsingPrompt, "prompts/end_to_end_parsing.md"),
        (
            ReferenceExtractionAndParsingPrompt,
            "prompts/prompts.yaml:extraction_and_parsing.default",
        ),
    ],
)
def test_pipeline_default_prompts_render_with_input(prompt_cls, prompt_name):
    """Every prompt the API/pipelines default to loads and carries the input text."""
    path, _, key = prompt_name.partition(":")
    marker = "Smith, J. (2020). A unique marker title."
    prompt = prompt_cls(prompt=path, input_text=marker, prompt_key=key or None)

    assert marker in prompt.messages["user"]
