"""
LLM client utilities
"""

import re
import json
import openai
import aiohttp
import concurrent.futures
from typing import Dict, List, Tuple, Optional


class LLMClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None, prompt: Optional[str] = None):
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
    
   
  
    
   
    
  