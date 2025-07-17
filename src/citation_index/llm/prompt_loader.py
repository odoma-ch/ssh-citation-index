import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.json_schema import GenerateJsonSchema

from citation_index.core.models.references import References

class PromptLoader:
    """Loader for prompts."""

    def __init__(self, prompt: str, examples: str, input_text: str):
        if prompt.endswith(".md"):
            self.prompt = self.load_markdown_prompt(prompt)
        else:
            self.prompt = prompt
        self.examples = examples
        self.input_text = input_text


    def load_markdown_prompt(self, prompt: str) -> str:
        """Load the prompt from the markdown file."""
        with open(prompt, "r") as f:
            return f.read()
        
    @classmethod
    def build_prompt(cls, prompt: str, examples: str, input_text: str, json_schema: str = None) -> str:
        return
    

        

class ReferenceExtractionPrompt(PromptLoader):
    """Prompt for reference extraction."""

    def __init__(self, prompt: str = "prompts/reference_extraction.md", examples: str = "", input_text: str = ""):
        super().__init__(prompt, examples, input_text)
        self.prompt = self.prompt.replace("{{INPUT_TEXT}}", self.input_text)


class ReferenceParsingPrompt(PromptLoader):
    """Prompt for reference parsing."""

    def __init__(self, prompt: str = "prompts/reference_parsing.md", examples: str = "", input_text: str = "", include_json_schema: bool = True):
        super().__init__(prompt, examples, input_text)
        self.prompt = self.prompt.replace("{{INPUT_TEXT}}", self.input_text)
        self.json_schema = None
        if include_json_schema:
            self.json_schema = self.load_json_schema()
            self.prompt = self.prompt.replace("{{JSON_SCHEMA_FOR_REFERENCES_WRAPPER}}", json.dumps(self.json_schema, indent=2))

    def load_json_schema(self) -> str:
        """Load the JSON schema from the file."""
        return References.schema_without_excluded()
    
class ReferenceExtractionAndParsingPrompt(PromptLoader):
    """Prompt for reference extraction and parsing."""

    def __init__(self, prompt: str = "prompts/reference_extraction_and_parsing_pydantic.md", examples: str = "", input_text: str = "",include_json_schema: bool = True):
        super().__init__(prompt, examples, input_text)
        self.prompt = self.prompt.replace("{{INPUT_TEXT}}", self.input_text)
        self.json_schema = None
        if include_json_schema:
            self.json_schema = self.load_json_schema()
            self.prompt = self.prompt.replace("{{JSON_SCHEMA_FOR_REFERENCES_WRAPPER}}", json.dumps(self.json_schema, indent=2))
            

    def load_json_schema(self) -> str:
        """Load the JSON schema from the file."""
        return References.schema_without_excluded()


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


    