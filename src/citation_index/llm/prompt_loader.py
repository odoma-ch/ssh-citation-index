import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.json_schema import GenerateJsonSchema

from citation_index.core.models.references import References

class PromptLoader:
    """Loader for prompts with support for both legacy .md files and new YAML format.
    
    Supports two formats:
    1. Legacy: Markdown files (.md) with {{PLACEHOLDER}} syntax
    2. YAML: Multi-prompt files with namespace structure and Jinja2 templates
    
    YAML format:
        namespace.key:
          system: "System prompt with {{ variable }} placeholders"
          user: "User prompt with {{ input_text }} placeholder"
          examples: [...] or "path/to/examples.json"
    
    Args:
        prompt: Path to prompt file (.md, .yaml, .yml) or raw prompt string
        examples: Examples string (legacy) or will be loaded from YAML
        input_text: The input text to process
        prompt_key: Key to select from YAML file (e.g., "parsing.default")
                   If None and loading YAML, tries "default" key
    """

    def __init__(
        self,
        prompt: str,
        examples: str = "",
        input_text: str = "",
        prompt_key: Optional[str] = None,
    ):
        self.prompt_path = prompt
        self.examples = examples
        self.input_text = input_text
        self.prompt_key = prompt_key
        self._format = self._detect_format(prompt)
        self._messages: Optional[Dict[str, str]] = None
        self._prompt_str: Optional[str] = None
        
    def _detect_format(self, prompt: str) -> str:
        """Detect prompt format based on file extension or content."""
        if prompt.endswith(('.yaml', '.yml')):
            return 'yaml'
        elif prompt.endswith(('.md', '.txt')):
            return 'markdown'
        elif os.path.exists(prompt):
            # File exists but no clear extension, sniff content
            try:
                with open(prompt, 'r') as f:
                    first_line = f.readline().strip()
                    # Check for namespace pattern like "extraction:" or "parsing:"
                    if ':' in first_line and not first_line.startswith('#'):
                        return 'yaml'
            except Exception:
                pass
            return 'markdown'
        else:
            # Treat as raw string
            return 'raw'
    
    def _load_yaml(self, prompt_key: Optional[str] = None) -> Dict[str, Any]:
        """Load YAML prompt configuration and extract by key.
        
        Args:
            prompt_key: Namespace key like "parsing.default" or "extraction.toon"
                       If None, tries "default" key
                       
        Returns:
            Dict with 'system', 'user', and optionally 'examples' keys
            
        Raises:
            ValueError: If prompt_key not found and no "default" key exists
        """
        with open(self.prompt_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Try specified key first
        if prompt_key and prompt_key in data:
            return data[prompt_key]
        
        # Fall back to "default" key
        if "default" in data:
            return data["default"]
        
        # Error if neither exists
        available_keys = list(data.keys())
        if prompt_key:
            raise ValueError(
                f"Prompt key '{prompt_key}' not found in {self.prompt_path}. "
                f"Available keys: {available_keys}"
            )
        else:
            raise ValueError(
                f"No 'default' key found in {self.prompt_path}. "
                f"Available keys: {available_keys}. "
                f"Please specify prompt_key parameter."
            )
    
    def _load_examples(self, examples: Union[List, str]) -> List:
        """Load examples from inline list or external file.
        
        Args:
            examples: Either a list of examples or a file path string
            
        Returns:
            List of examples
        """
        if isinstance(examples, list):
            return examples
        elif isinstance(examples, str) and examples:
            # Treat as file path
            path = Path(examples)
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        return []
    
    def _render_jinja(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render Jinja2 template with given context.
        
        Args:
            template_str: Jinja2 template string
            context: Variables to pass to template
            
        Returns:
            Rendered string
        """
        template = Template(template_str)
        return template.render(**context)
    
    def load_markdown_prompt(self, prompt: str) -> str:
        """Load the prompt from the markdown file (legacy format)."""
        with open(prompt, "r") as f:
            return f.read()
        
    @classmethod
    def build_prompt(
        cls,
        prompt: str,
        examples: str,
        input_text: str,
        json_schema: dict | None = None,
    ) -> str:
        """Build prompt with legacy string replacement (backward compatibility).
        
        Read prompt (path or raw), inject input text and optional schema.
        Uses {{PLACEHOLDER}} syntax for replacement.
        """
        if prompt.endswith(".md"):
            content = cls.load_markdown_prompt(cls, prompt)
        else:
            content = prompt
        if examples:
            content = content.replace("{{EXAMPLES}}", examples)
        content = content.replace("{{INPUT_TEXT}}", input_text)
        if json_schema is not None:
            import json
            content = content.replace(
                "{{JSON_SCHEMA_FOR_REFERENCES_WRAPPER}}",
                json.dumps(json_schema, indent=2),
            )
        return content
    
    @property
    def prompt(self) -> str:
        """Get prompt as single string (backward compatibility).
        
        For YAML format, concatenates system and user prompts.
        For markdown/raw format, returns the processed prompt string.
        """
        if self._prompt_str is None:
            if self._format == 'yaml':
                messages = self.messages
                self._prompt_str = f"{messages['system']}\n\n{messages['user']}"
            elif self._format == 'markdown':
                self._prompt_str = self.load_markdown_prompt(self.prompt_path)
            else:  # raw
                self._prompt_str = self.prompt_path
        return self._prompt_str
    
    @property
    def messages(self) -> Dict[str, str]:
        """Get prompt as structured messages with system/user separation.
        
        Returns:
            Dict with 'system' and 'user' keys containing rendered prompts
            
        Raises:
            ValueError: If YAML format but prompt_key not found
        """
        if self._messages is None:
            if self._format == 'yaml':
                config = self._load_yaml(self.prompt_key)
                
                # Load examples if specified
                examples = []
                if 'examples' in config:
                    examples = self._load_examples(config['examples'])
                
                # Build Jinja2 context
                context = {
                    'input_text': self.input_text,
                    'examples': examples,
                }
                
                # Render system and user prompts
                self._messages = {
                    'system': self._render_jinja(config['system'], context),
                    'user': self._render_jinja(config['user'], context),
                }
            else:
                # For markdown/raw, no system/user separation
                # Return everything as user message for backward compatibility
                self._messages = {
                    'system': '',
                    'user': self.prompt,
                }
        return self._messages
    

        

class ReferenceExtractionPrompt(PromptLoader):
    """Prompt for reference extraction."""

    def __init__(
        self,
        prompt: str = "prompts/reference_extraction.md",
        examples: str = "",
        input_text: str = "",
        prompt_key: Optional[str] = None,
    ):
        super().__init__(prompt, examples, input_text, prompt_key)
        # For backward compatibility with legacy format
        if self._format != 'yaml':
            self._prompt_str = self.build_prompt(prompt, examples, input_text, json_schema=None)


class ReferenceParsingPrompt(PromptLoader):
    """Prompt for reference parsing."""

    def __init__(
        self,
        prompt: str = "prompts/reference_parsing.md",
        examples: str = "",
        input_text: str = "",
        include_json_schema: bool = True,
        prompt_key: Optional[str] = None,
    ):
        super().__init__(prompt, examples, input_text, prompt_key)
        self.json_schema = None
        if include_json_schema and self._format != 'yaml':
            self.json_schema = self.load_json_schema()
            self._prompt_str = self.build_prompt(
                prompt, examples, input_text, json_schema=self.json_schema
            )
        elif self._format != 'yaml':
            self._prompt_str = self.build_prompt(prompt, examples, input_text, json_schema=None)
        
        # For YAML format, inject schema into context
        if self._format == 'yaml' and include_json_schema:
            self.json_schema = self.load_json_schema()

    def load_json_schema(self) -> dict:
        """Load the JSON schema from the References model."""
        return References.schema_without_excluded()
    
    @property
    def messages(self) -> Dict[str, str]:
        """Get messages with optional JSON schema injection for YAML format."""
        if self._messages is None:
            if self._format == 'yaml':
                config = self._load_yaml(self.prompt_key)
                
                # Load examples if specified
                examples = []
                if 'examples' in config:
                    examples = self._load_examples(config['examples'])
                
                # Build Jinja2 context with schema if needed
                context = {
                    'input_text': self.input_text,
                    'examples': examples,
                }
                if self.json_schema is not None:
                    context['json_schema'] = json.dumps(self.json_schema, indent=2)
                
                # Render system and user prompts
                self._messages = {
                    'system': self._render_jinja(config['system'], context),
                    'user': self._render_jinja(config['user'], context),
                }
            else:
                # For markdown/raw, use parent behavior
                self._messages = super().messages
        return self._messages

    
class ReferenceExtractionAndParsingPrompt(PromptLoader):
    """Prompt for reference extraction and parsing."""

    def __init__(
        self,
        prompt: str = "prompts/reference_extraction_and_parsing.md",
        examples: str = "",
        input_text: str = "",
        include_json_schema: bool = True,
        prompt_key: Optional[str] = None,
    ):
        super().__init__(prompt, examples, input_text, prompt_key)
        self.json_schema = None
        if include_json_schema and self._format != 'yaml':
            self.json_schema = self.load_json_schema()
            self._prompt_str = self.build_prompt(
                prompt, examples, input_text, json_schema=self.json_schema
            )
        elif self._format != 'yaml':
            self._prompt_str = self.build_prompt(prompt, examples, input_text, json_schema=None)
            
        # For YAML format, inject schema into context
        if self._format == 'yaml' and include_json_schema:
            self.json_schema = self.load_json_schema()

    def load_json_schema(self) -> dict:
        """Load the JSON schema from the References model."""
        return References.schema_without_excluded()
    
    @property
    def messages(self) -> Dict[str, str]:
        """Get messages with optional JSON schema injection for YAML format."""
        if self._messages is None:
            if self._format == 'yaml':
                config = self._load_yaml(self.prompt_key)
                
                # Load examples if specified
                examples = []
                if 'examples' in config:
                    examples = self._load_examples(config['examples'])
                
                # Build Jinja2 context with schema if needed
                context = {
                    'input_text': self.input_text,
                    'examples': examples,
                }
                if self.json_schema is not None:
                    context['json_schema'] = json.dumps(self.json_schema, indent=2)
                
                # Render system and user prompts
                self._messages = {
                    'system': self._render_jinja(config['system'], context),
                    'user': self._render_jinja(config['user'], context),
                }
            else:
                # For markdown/raw, use parent behavior
                self._messages = super().messages
        return self._messages


if __name__ == "__main__":

    # input_text = "Smith, J. A., & Johnson, B. C. (2020). The impact of climate change on biodiversity. Nature Ecology & Evolution, 4(5), 123-145. https://doi.org/10.1038/s41559-020-1234-5"
    
    input_text = """
    This paper builds on previous work (Smith et al., 2020; Jones, 2019). According to recent studies...
    Smith, J. A., & Johnson, B. C. (2020). The impact of climate change on biodiversity. Nature Ecology & Evolution, 4(5), 123-145. https://doi.org/10.1038/s41559-020-1234-5
    Brown, M. L., Davis, R. K., & Wilson, E. F. (2019). Machine learning approaches to natural language processing. Journal of Artificial Intelligence Research, 65, 789-812.
    Garcia, S., & Martinez, P. (2021). Sustainable development goals: A comprehensive review. Sustainability Science, 16(3), 456-478.
    Thompson, K. R., & Anderson, L. M. (2018). Quantum computing: Principles and applications. Quantum Information Processing, 17(4), 234-256. https://doi.org/10.1007/qip.2018.1234
    """
    prompt = ReferenceExtractionAndParsingPrompt(input_text=input_text)
    print(prompt.prompt)


    