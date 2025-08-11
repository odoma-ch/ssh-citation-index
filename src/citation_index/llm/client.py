"""
LLM client utilities
"""

import re
import json
# import openai
from langfuse.openai import openai
import aiohttp
import concurrent.futures
import logging
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
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(base_url=endpoint, api_key=api_key)
        else:
            self.client = openai.OpenAI(base_url=endpoint)
    
    def call(self, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = None, json_schema: str = None, json_output: bool = False) -> str:
        """Call the LLM API."""
        model = model if model else self.model
        
        response_format = None
        is_deepseek = "api.deepseek.com" in self.endpoint

        if is_deepseek and (json_schema or json_output):
            response_format = {"type": "json_object"}
            max_tokens = 8000
        elif json_schema:
            response_format = {
               "type": "json_schema",
               "json_schema": json_schema
            }
        elif json_output:
            response_format = {
                "type": "json_object"
            }
        
            
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format)
        return response.choices[0].message.content

    def call_structured(
        self,
        prompt: str,
        json_schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Thin wrapper to request structured JSON output when schema is provided."""
        return self.call(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            json_output=bool(json_schema),
        )
    
    def call_with_continuation(self, prompt: str, start_tag: str = "```json", end_tag: str = "```", 
                             model: str = None, temperature: float = 0.0, max_tokens: int = 8192,
                             max_continuations: int = 5, json_schema: str = None, json_output: bool = False) -> str:
        """Call the LLM API with automatic continuation if response is incomplete.
        
        Args:
            prompt: The initial prompt to send
            start_tag: Tag(s) that indicate the start of the response content (string or list of strings)
            end_tag: Tag(s) that indicate the end of the response content (string or list of strings)
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response
            max_continuations: Maximum number of continuation attempts
            json_schema: JSON schema for the response
        Returns:
            Complete response string
        """
        model = model if model else self.model
        
        response_format = None
        is_deepseek = "api.deepseek.com" in self.endpoint

        if is_deepseek and (json_schema or json_output):
            response_format = {"type": "json_object"}
        elif json_schema:
            response_format = {
               "type": "json_schema",
               "json_schema": json_schema
            }
        elif json_output:
            response_format = {"type": "json_object"}
        
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
            logging.info(f"API call #{call_number}...")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            current_response = response.choices[0].message.content
            full_response += current_response
            
            # Check if response is complete by looking for any end tag
            response_complete = any(tag in full_response for tag in end_tags)
            if response_complete:
                # Response is complete
                messages.append({"role": "assistant", "content": full_response})
                logging.info(f"Response completed in {call_number} call(s)")
                break
            else:
                # Response is incomplete, continue the conversation
                continuation_count += 1
                if continuation_count >= max_continuations:
                    logging.warning(f"Reached maximum continuations ({max_continuations}). Response may be incomplete.")
                    break
                
                # Add the current response to conversation history
                messages.append({"role": "assistant", "content": current_response})
                
                # Add a continuation prompt
                end_tags_str = ", ".join(end_tags)
                continuation_prompt = f"Please continue your response from where you left off. Make sure to include one of the following end tags when you finish: {end_tags_str}"
                messages.append({"role": "user", "content": continuation_prompt})
        
        return messages, full_response
    


class DeepSeekClient(LLMClient):
    """Client for interacting with DeepSeek API"""
    
    def __init__(self, api_key: str):
        super().__init__(endpoint="https://api.deepseek.com/v1", model="deepseek-chat", api_key=api_key)
        

class VLLMClient(LLMClient):
    """Client for interacting with VLLM API"""
    
    def call_with_parsed_structured_output(self, prompt: str, model: str = None, temperature: float = 0.5, json_class: object = None) -> str:
        """Call the LLM API with structured output."""
        model = model if model else self.model
        if json_class is None:
            raise ValueError("a pydantic model is required")
        response_format = json_class
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=response_format
        )
        return response.choices[0].message.parsed


    

    
   
    
  