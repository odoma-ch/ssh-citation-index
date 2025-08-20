"""
LLM client utilities
"""

import re
import json
import time
import threading
import asyncio
from contextlib import contextmanager
# import openai
from langfuse.openai import openai
import aiohttp
import concurrent.futures
import logging
from typing import Dict, List, Tuple, Optional, Iterator
from .prompt_loader import PromptLoader


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class ThreadSafeTimeout:
    """Thread-safe timeout mechanism using threading.Timer."""
    
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.timer = None
        self.timed_out = threading.Event()
        
    def __enter__(self):
        def timeout_handler():
            self.timed_out.set()
            
        self.timer = threading.Timer(self.seconds, timeout_handler)
        self.timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
            
    def check_timeout(self):
        """Check if timeout occurred and raise exception if so."""
        if self.timed_out.is_set():
            raise TimeoutError(f"Operation timed out after {self.seconds} seconds")


@contextmanager
def timeout_context(seconds: float):
    """Thread-safe context manager for timing out operations."""
    with ThreadSafeTimeout(seconds) as timeout_manager:
        try:
            yield timeout_manager
        finally:
            timeout_manager.check_timeout()


class LLMClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None, 
                 timeout: float = 180.0, max_retries: int = 3, first_token_timeout: float = 30.0):
        """Initialize the LLM client.
        
        Args:
            endpoint: LLM API endpoint
            model: Model name to use
            api_key: API key for authentication
            timeout: Maximum time to wait for complete response (seconds)
            max_retries: Maximum number of retry attempts for failed calls
            first_token_timeout: Maximum time to wait for first token (seconds)
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.first_token_timeout = first_token_timeout
        
        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(base_url=endpoint, api_key=api_key)
        else:
            self.client = openai.OpenAI(base_url=endpoint)
    
    def _stream_with_timeout(self, **kwargs) -> Iterator[str]:
        """Stream response with first-token timeout detection."""
        stream = self.client.chat.completions.create(stream=True, **kwargs)
        
        first_token_received = False
        start_time = time.time()
        
        with ThreadSafeTimeout(self.first_token_timeout) as first_token_timer:
            for chunk in stream:
                if not first_token_received:
                    # Check if first token timeout occurred
                    first_token_timer.check_timeout()
                    elapsed = time.time() - start_time
                    first_token_received = True
                    logging.info(f"First token received after {elapsed:.1f} seconds")
                
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
    
    def _call_with_retry(self, prompt: str, model: str = None, temperature: float = 0.3, 
                        max_tokens: int = None, json_schema: str = None, json_output: bool = False,
                        use_streaming: bool = True) -> str:
        """Call LLM with timeout and retry logic."""
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
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["\n\n\n\n\n"],  # Stop sequence to prevent long non-stopped responses
            "response_format": response_format
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if use_streaming:
                    # Use streaming for first-token timeout detection
                    with timeout_context(self.timeout) as timeout_manager:
                        content_parts = []
                        for part in self._stream_with_timeout(**kwargs):
                            timeout_manager.check_timeout()
                            content_parts.append(part)
                        return "".join(content_parts)
                else:
                    # Use regular call with full timeout
                    with timeout_context(self.timeout) as timeout_manager:
                        response = self.client.chat.completions.create(**kwargs)
                        timeout_manager.check_timeout()
                        return response.choices[0].message.content
                        
            except (TimeoutError, Exception) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                if isinstance(e, TimeoutError):
                    logging.warning(f"Timeout on {attempt_info}: {e}")
                else:
                    logging.warning(f"Error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff with cap
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {self.max_retries + 1} attempts failed")
                    break
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected: no response and no exception")

    def call(self, prompt: str, model: str = None, temperature: float = 0.3, max_tokens: int = None, json_schema: str = None, json_output: bool = False, use_streaming: bool = True) -> str:
        """Call the LLM API with timeout and retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            json_schema: JSON schema for structured output
            json_output: Whether to request JSON output
            use_streaming: Whether to use streaming for first-token timeout detection
        """
        return self._call_with_retry(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            json_output=json_output,
            use_streaming=use_streaming
        )

    def call_structured(
        self,
        prompt: str,
        json_schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.3,
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
                             model: str = None, temperature: float = 0.3, max_tokens: int = 8192,
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
            # Make the API call with timeout protection
            call_number = continuation_count + 1
            logging.info(f"API call #{call_number}...")
            
            # Use the retry mechanism for each continuation call
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format
            }
            
            try:
                with timeout_context(self.timeout) as timeout_manager:
                    content_parts = []
                    for part in self._stream_with_timeout(**kwargs):
                        timeout_manager.check_timeout()
                        content_parts.append(part)
                    current_response = "".join(content_parts)
            except (TimeoutError, Exception) as e:
                logging.error(f"Continuation call #{call_number} failed: {e}")
                raise
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
    
    def call_with_parsed_structured_output(self, prompt: str, model: str = None, temperature: float = 0.3, json_class: object = None) -> str:
        """Call the LLM API with structured output and timeout protection."""
        model = model if model else self.model
        if json_class is None:
            raise ValueError("a pydantic model is required")
        
        response_format = json_class
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                with timeout_context(self.timeout) as timeout_manager:
                    response = self.client.beta.chat.completions.parse(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        response_format=response_format
                    )
                    timeout_manager.check_timeout()
                    return response.choices[0].message.parsed
                    
            except (TimeoutError, Exception) as e:
                last_exception = e
                attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
                
                if isinstance(e, TimeoutError):
                    logging.warning(f"Structured output timeout on {attempt_info}: {e}")
                else:
                    logging.warning(f"Structured output error on {attempt_info}: {type(e).__name__}: {e}")
                
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff with cap
                    logging.info(f"Retrying structured output in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {self.max_retries + 1} structured output attempts failed")
                    break
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        
        raise RuntimeError("Unexpected: no response and no exception in structured output")


    

    
   
    
  