"""
LLM client utilities
"""

import re
import json
import openai
import aiohttp
import concurrent.futures
from typing import Dict, List, Tuple, Optional
from .prompt_loader import PromptLoader

class LLMClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None):
        """Initialize the LLM client.
        
        Args:
            endpoint: LLM API endpoint
            model: Model name to use
            api_key: API key for authentication
            prompt: System prompt for evaluation
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
    
    
        # if prompt is None:
        #     self.prompt = None
        # elif prompt.endswith('.md'):
        #     with open(prompt, 'r') as f:
        #         self.prompt = f.read()
        # else:
        #     self.prompt = prompt
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(base_url=endpoint, api_key=api_key)
        else:
            self.client = openai.OpenAI(base_url=endpoint)
    
    def call(self, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 8192) -> str:
        """Call the LLM API."""
        if model is None:
            model = self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens)
        return response.choices[0].message.content
    
    def call_with_continuation(self, prompt: str, start_tag: str = "```json", end_tag: str = "```", 
                             model: str = None, temperature: float = 0.0, max_tokens: int = 8192,
                             max_continuations: int = 5) -> str:
        """Call the LLM API with automatic continuation if response is incomplete.
        
        Args:
            prompt: The initial prompt to send
            start_tag: Tag(s) that indicate the start of the response content (string or list of strings)
            end_tag: Tag(s) that indicate the end of the response content (string or list of strings)
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response
            max_continuations: Maximum number of continuation attempts
            
        Returns:
            Complete response string
        """
        if model is None:
            model = self.model
            
        # Convert single tags to lists for consistent handling
        start_tags = [start_tag] if isinstance(start_tag, str) else start_tag
        end_tags = [end_tag] if isinstance(end_tag, str) else end_tag
            
        # Initialize conversation history
        messages = [{"role": "user", "content": prompt}]
        full_response = ""
        continuation_count = 0
        
        while continuation_count < max_continuations:
            # Make the API call
            call_number = continuation_count + 1
            print(f"API call #{call_number}...")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            current_response = response.choices[0].message.content
            full_response += current_response
            
            # Check if response is complete by looking for any end tag
            response_complete = any(tag in full_response for tag in end_tags)
            if response_complete:
                # Response is complete
                messages.append({"role": "assistant", "content": full_response})
                print(f"Response completed in {call_number} call(s)")
                break
            else:
                # Response is incomplete, continue the conversation
                continuation_count += 1
                if continuation_count >= max_continuations:
                    print(f"Warning: Reached maximum continuations ({max_continuations}). Response may be incomplete.")
                    break
                
                # Add the current response to conversation history
                messages.append({"role": "assistant", "content": current_response})
                
                # Add a continuation prompt
                end_tags_str = ", ".join(end_tags)
                continuation_prompt = f"Please continue your response from where you left off. Make sure to include one of the following end tags when you finish: {end_tags_str}"
                messages.append({"role": "user", "content": continuation_prompt})
        
        return messages, full_response
    
    def call_with_json_continuation(self, prompt: str, model: str = None, temperature: float = 0.0, 
                                  max_tokens: int = 8192, max_continuations: int = 5) -> str:
        """Call the LLM API with automatic continuation specifically for JSON responses.
        
        Args:
            prompt: The initial prompt to send
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response
            max_continuations: Maximum number of continuation attempts
            
        Returns:
            Complete JSON response string
        """
        return self.call_with_continuation(
            prompt=prompt,
            start_tag="```json",
            end_tag="```",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_continuations=max_continuations
        )
    

    
   
    
  