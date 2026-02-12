#!/usr/bin/env python3
"""
Example demonstrating the new YAML-based prompt system with system/user message separation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from citation_index.llm.prompt_loader import ReferenceParsingPrompt

def example_yaml_prompts():
    """Example using new YAML prompts with namespace structure."""
    
    input_text = """
    1. Smith, J., & Brown, A. (2020). Deep Learning in NLP. AI Journal, 15(3), 100-120.
    2. Jones, M. (2019). Machine Translation Systems. MIT Press.
    """
    
    # Load prompt from YAML with namespace key
    prompt_obj = ReferenceParsingPrompt(
        prompt="prompts/prompts.yaml",
        prompt_key="parsing.default",  # Use namespace.variant pattern
        input_text=input_text,
        include_json_schema=True
    )
    
    # New way: Get structured messages for system/user separation
    messages = prompt_obj.messages
    print("System Prompt:")
    print("-" * 80)
    print(messages['system'])
    print("\n")
    
    print("User Prompt (first 500 chars):")
    print("-" * 80)
    print(messages['user'][:500])
    print("...")
    print("\n")
    
    # Can also use messages with LLMClient
    # response = llm_client.call(messages=messages, json_output=True)
    
    # Or for backward compatibility, still works as single string:
    # response = llm_client.call(prompt=prompt_obj.prompt, json_output=True)


def example_legacy_prompts():
    """Example using legacy .md prompts (still supported)."""
    
    input_text = "1. Test Reference (2020). Title. Journal."
    
    # Legacy approach still works
    prompt_obj = ReferenceParsingPrompt(
        prompt="prompts/reference_parsing.md",
        input_text=input_text,
        include_json_schema=False
    )
    
    print("Legacy Prompt (first 300 chars):")
    print("-" * 80)
    print(prompt_obj.prompt[:300])
    print("...")
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EXAMPLE: New YAML-based Prompts")
    print("=" * 80 + "\n")
    example_yaml_prompts()
    
    print("=" * 80)
    print("EXAMPLE: Legacy .md Prompts (still supported)")
    print("=" * 80 + "\n")
    example_legacy_prompts()
