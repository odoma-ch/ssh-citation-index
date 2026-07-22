"""
LLM client utilities.

Timeout hierarchy (innermost to outermost):
    1. first_token_timeout → httpx per-chunk read timeout.
       Enforced at the TCP socket level by httpx.  If the server stops
       sending data (e.g. model finishes but connection hangs), the read
       is aborted after this many seconds.
    2. timeout → total wall-clock limit per attempt.
       For streaming: checked between chunks in _stream_with_timeout.
       For non-streaming: used as the httpx read timeout.
    3. Queue timeout (RQ job timeout, set in config.py / api.py).
       Hard-kills the worker process.  Must be larger than
       timeout × (max_retries + 1) + buffer.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Iterator, Union

import httpx
import numpy as np
import openai


logger = logging.getLogger(__name__)


# ========================
# Timeout helpers
# ========================

# Fixed httpx timeout values (seconds) for non-read phases
CONNECT_TIMEOUT = 15.0  # TCP + TLS handshake
WRITE_TIMEOUT = 30.0  # sending the request body
POOL_TIMEOUT = 15.0  # acquiring a connection from the pool


class LLMTimeoutError(Exception):
    """Raised when an LLM operation exceeds its wall-clock timeout."""

    pass


class LLMEmptyResponseError(Exception):
    """Raised when an LLM request succeeds but contains no answer text."""

    pass


# Exceptions that are safe to retry with exponential back-off
RETRYABLE_EXCEPTIONS = (
    LLMTimeoutError,
    LLMEmptyResponseError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMClient:
    """Client for interacting with LLM APIs"""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        first_token_timeout: float = 30.0,
        enable_thinking: Optional[bool] = None,
    ):
        """Initialize the LLM client.

        Args:
            endpoint: LLM API endpoint
            model: Model name to use
            api_key: API key for authentication
            timeout: Total wall-clock timeout for one LLM attempt (seconds).
            max_retries: Maximum number of retry attempts for failed calls.
            first_token_timeout: Per-chunk httpx read timeout (seconds).
                Controls how long to wait for each chunk of data from the
                server, including the very first token.  This is enforced at
                the TCP socket level by httpx and is the primary defence
                against hung connections (server stops sending but TCP stays
                open).
            enable_thinking: Optional Qwen/vLLM chat-template switch. Set to
                ``False`` for citation tasks so the final answer is returned in
                ``content`` instead of consuming the output budget in the
                reasoning channel. Leave as ``None`` for non-vLLM endpoints.
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.first_token_timeout = first_token_timeout
        self.enable_thinking = enable_thinking

        # Client-level timeout is generous; per-call overrides give
        # precise control for streaming vs non-streaming paths.
        default_timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=timeout,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=api_key or "dummy-key",
            timeout=default_timeout,
        )

    # ------------------------------------------------------------------
    # httpx.Timeout helpers (called per-request to override client default)
    # ------------------------------------------------------------------

    def _streaming_httpx_timeout(self) -> httpx.Timeout:
        """Per-chunk timeout for streaming requests.

        Uses first_token_timeout as the read timeout.  If the server
        stops sending data for longer than this, httpx raises ReadTimeout
        (surfaced as openai.APITimeoutError).
        """
        return httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=self.first_token_timeout,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )

    def _non_streaming_httpx_timeout(self) -> httpx.Timeout:
        """Timeout for non-streaming requests.

        Uses total timeout as the read timeout since the entire
        response body must arrive in one go.
        """
        return httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=self.timeout,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Streaming with timeout
    # ------------------------------------------------------------------

    def _stream_with_timeout(self, **kwargs) -> Iterator[str]:
        """Stream response with per-chunk *and* total wall-clock timeouts.

        Per-chunk timeout: enforced by httpx read timeout at the socket
        level.  Catches stuck connections where the server stops sending
        data (the exact scenario where the model finishes but HTTP hangs).

        Total timeout: checked between chunks.  Catches slow-but-steady
        responses that exceed the overall time budget.
        """
        start_time = time.time()
        first_token_received = False
        content_chars = 0
        reasoning_chars = 0

        # Override the client-level timeout for this streaming call
        stream = self.client.chat.completions.create(
            stream=True,
            timeout=self._streaming_httpx_timeout(),
            **kwargs,
        )

        try:
            for chunk in stream:
                elapsed = time.time() - start_time

                if not first_token_received:
                    first_token_received = True
                    logger.info("LLM first token received after %.1fs", elapsed)

                # Check total wall-clock timeout between chunks
                if elapsed > self.timeout:
                    raise LLMTimeoutError(
                        f"Total wall-clock timeout ({self.timeout}s) exceeded "
                        f"after {elapsed:.1f}s of streaming"
                    )

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if not isinstance(reasoning, str):
                        reasoning = getattr(delta, "reasoning", None)
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_chars += len(reasoning)
                    if delta.content is not None:
                        content_chars += len(delta.content)
                        yield delta.content

            if content_chars == 0:
                detail = (
                    f"; received {reasoning_chars} reasoning characters but no answer text"
                    if reasoning_chars
                    else ""
                )
                raise LLMEmptyResponseError(f"LLM returned an empty response{detail}")
        finally:
            # Always close the stream to release the HTTP connection
            try:
                stream.close()
            except Exception:
                pass

    def _get_response_format_and_prompt(
        self,
        prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        json_output: bool = False,
        max_tokens: int = None,
    ):
        """Get response format and potentially modified prompt for non-DeepSeek models."""
        response_format = None
        modified_prompt = prompt
        modified_max_tokens = max_tokens

        if json_schema:
            # OpenAI-compatible servers expect the actual schema under a named
            # ``json_schema.schema`` wrapper. Passing a raw schema here is
            # accepted by the SDK but reaches vLLM as an empty constraint.
            schema_name = str(json_schema.get("title") or "structured_response")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            }
        elif json_output:
            # Use json_object for simple JSON (vLLM/OpenAI)
            response_format = {"type": "json_object"}

        return response_format, modified_prompt, modified_max_tokens

    def _call_with_retry(
        self,
        prompt: Union[str, Dict[str, str]] = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        json_schema: Optional[Dict[str, Any]] = None,
        json_output: bool = False,
        use_streaming: bool = True,
        messages: Optional[Dict[str, str]] = None,
    ) -> str:
        """Call LLM with timeout and retry logic.

        Args:
            prompt: Legacy parameter - prompt string (deprecated, use messages instead)
            messages: Dict with 'system' and 'user' keys for structured messages
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            json_schema: JSON schema for structured output
            json_output: Whether to request JSON output
            use_streaming: Whether to use streaming for first-token timeout detection

        Returns:
            Generated text response
        """
        model = model if model else self.model

        # Handle backward compatibility: accept either prompt string or messages dict
        if messages is None:
            if isinstance(prompt, dict):
                # If prompt is a dict, treat it as messages
                messages = prompt
            else:
                # Legacy behavior: wrap prompt string as user message
                messages = {"system": "", "user": prompt}

        response_format, modified_prompt, modified_max_tokens = (
            self._get_response_format_and_prompt(
                messages["user"], json_schema, json_output, max_tokens
            )
        )

        # Build messages list for API call
        message_list = []
        if messages.get("system"):
            message_list.append({"role": "system", "content": messages["system"]})
        message_list.append({"role": "user", "content": modified_prompt})

        kwargs = {
            "model": model,
            "messages": message_list,
            "temperature": temperature,
            "max_tokens": modified_max_tokens,
            "stop": [
                "\n\n\n\n\n"
            ],  # Stop sequence to prevent long non-stopped responses
            "response_format": response_format,
        }
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            }

        last_exception = None

        for attempt in range(self.max_retries + 1):
            attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
            attempt_started = time.monotonic()
            try:
                logger.info(
                    "LLM request started: model=%s, streaming=%s, %s",
                    model,
                    use_streaming,
                    attempt_info,
                )
                if use_streaming:
                    # Streaming: httpx read timeout catches stuck connections,
                    # wall-clock check in _stream_with_timeout catches overall timeout
                    content_parts = []
                    for part in self._stream_with_timeout(**kwargs):
                        content_parts.append(part)
                    content = "".join(content_parts)
                else:
                    # Non-streaming: httpx read timeout = total timeout
                    response = self.client.chat.completions.create(
                        timeout=self._non_streaming_httpx_timeout(),
                        **kwargs,
                    )
                    message = response.choices[0].message
                    content = message.content or ""
                    if not content.strip():
                        reasoning = getattr(message, "reasoning_content", None)
                        if not isinstance(reasoning, str):
                            reasoning = getattr(message, "reasoning", None)
                        detail = (
                            f"; received {len(reasoning)} reasoning characters but no answer text"
                            if isinstance(reasoning, str) and reasoning
                            else ""
                        )
                        raise LLMEmptyResponseError(
                            f"LLM returned an empty response{detail}"
                        )

                logger.info(
                    "LLM request completed: model=%s, chars=%d, elapsed=%.1fs, %s",
                    model,
                    len(content),
                    time.monotonic() - attempt_started,
                    attempt_info,
                )
                return content

            except Exception as e:
                last_exception = e

                # Classify the error for logging and retry decisions
                if isinstance(e, (LLMTimeoutError, openai.APITimeoutError)):
                    logger.warning("LLM timeout on %s: %s", attempt_info, e)
                elif isinstance(e, RETRYABLE_EXCEPTIONS):
                    logger.warning(
                        "LLM retryable error on %s after %.1fs: %s: %s",
                        attempt_info,
                        time.monotonic() - attempt_started,
                        type(e).__name__,
                        e,
                    )
                else:
                    # Non-retryable error (bad request, auth, etc.) – fail immediately
                    logger.exception(
                        "LLM non-retryable error on %s: %s",
                        attempt_info,
                        type(e).__name__,
                    )
                    raise

                if attempt < self.max_retries:
                    wait_time = min(2**attempt, 10)  # Exponential backoff capped at 10s
                    logger.info("Retrying LLM request in %ss", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("All %d LLM attempts failed", self.max_retries + 1)
                    break

        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected: no response and no exception")

    def call(
        self,
        prompt: Union[str, Dict[str, str]] = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        json_schema: Optional[Dict[str, Any]] = None,
        json_output: bool = False,
        use_streaming: bool = True,
        messages: Optional[Dict[str, str]] = None,
    ) -> str:
        """Call the LLM API with timeout and retry logic.

        Args:
            prompt: The prompt to send to the LLM (string or dict with system/user keys)
            messages: Dict with 'system' and 'user' keys (alternative to prompt parameter)
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            json_schema: JSON schema for structured output
            json_output: Whether to request JSON output
            use_streaming: Whether to use streaming for first-token timeout detection
        """
        return self._call_with_retry(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            json_output=json_output,
            use_streaming=use_streaming,
        )

    def call_structured(
        self,
        prompt: str,
        json_schema: dict | None = None,
        model: str | None = None,
        temperature: float = 0.1,
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


class DeepSeekClient(LLMClient):
    """Client for interacting with DeepSeek API with continuation support and optimized settings"""

    def __init__(
        self,
        endpoint: str = "https://api.deepseek.com/v1",
        api_key: str = None,
        model: str = "deepseek-chat",
    ):
        super().__init__(endpoint=endpoint, model=model, api_key=api_key)

    def _get_deepseek_response_format_and_prompt(
        self,
        prompt: str,
        json_schema: str = None,
        json_output: bool = False,
        max_tokens: int = None,
        use_continuation: bool = False,
    ):
        """DeepSeek-specific response format and prompt handling."""
        response_format = None
        modified_prompt = prompt
        modified_max_tokens = max_tokens

        if json_schema or json_output:
            # DeepSeek: always use json_object + modify prompt + larger max_tokens
            response_format = {"type": "json_object"}
            modified_max_tokens = 8000  # Ensure at least 8k tokens

            # Add JSON instruction to prompt
            json_instruction = "\n\nPlease respond in valid JSON format."
            if "json" not in prompt.lower():
                modified_prompt = prompt + json_instruction

        # Add continuation tags instruction if using continuation
        if use_continuation:
            if json_output or json_schema:
                tag_instruction = (
                    "Make sure you wrap your JSON response with <start> and <end> tags."
                )
            else:
                tag_instruction = "\n\n Wrap your answer with <start> and <end> tags."

            modified_prompt = modified_prompt + tag_instruction

        return response_format, modified_prompt, modified_max_tokens

    def call(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        json_schema: str = None,
        json_output: bool = False,
        use_streaming: bool = True,
        use_continuation: bool = True,
    ) -> str:
        """Call DeepSeek API with optional continuation support.

        Args:
            prompt: The prompt to send
            model: Model name (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response
            json_schema: JSON schema for structured output
            json_output: Whether to request JSON output
            use_streaming: Whether to use streaming
            use_continuation: Whether to use continuation for complete responses (default True for JSON)

        Returns:
            Complete response string
        """
        if use_continuation and (json_output or json_schema):
            # Use continuation for reliable complete JSON responses
            messages, response = self.call_with_continuation(
                prompt=prompt,
                start_tag="<start>",
                end_tag="<end>",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_output=False,
            )
            return response
        else:
            # Use regular call with DeepSeek optimizations
            return self._deepseek_call_with_retry(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
                json_output=json_output,
                use_streaming=use_streaming,
            )

    def _deepseek_call_with_retry(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        json_schema: str = None,
        json_output: bool = False,
        use_streaming: bool = True,
    ) -> str:
        """DeepSeek-specific call with retry logic and optimizations."""
        model = model if model else self.model

        response_format, modified_prompt, modified_max_tokens = (
            self._get_deepseek_response_format_and_prompt(
                prompt, json_schema, json_output, max_tokens, use_continuation=False
            )
        )

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": modified_prompt}],
            "temperature": temperature,
            "max_tokens": modified_max_tokens,
            "stop": [
                "\n\n\n\n\n"
            ],  # Stop sequence to prevent long non-stopped responses
            "response_format": response_format,
        }

        last_exception = None

        for attempt in range(self.max_retries + 1):
            attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
            try:
                if use_streaming:
                    content_parts = []
                    for part in self._stream_with_timeout(**kwargs):
                        content_parts.append(part)
                    return "".join(content_parts)
                else:
                    response = self.client.chat.completions.create(
                        timeout=self._non_streaming_httpx_timeout(),
                        **kwargs,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                last_exception = e

                if isinstance(e, (LLMTimeoutError, openai.APITimeoutError)):
                    logging.warning(f"DeepSeek timeout on {attempt_info}: {e}")
                elif isinstance(e, RETRYABLE_EXCEPTIONS):
                    logging.warning(
                        f"DeepSeek retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                else:
                    logging.error(
                        f"DeepSeek non-retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                    raise

                if attempt < self.max_retries:
                    wait_time = min(2**attempt, 10)
                    logging.info(f"Retrying DeepSeek call in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"All {self.max_retries + 1} DeepSeek attempts failed"
                    )
                    break

        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected: no response and no exception in DeepSeek call")

    def call_with_continuation(
        self,
        prompt: str,
        start_tag: str = "```json",
        end_tag: str = "```",
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_continuations: int = 5,
        json_schema: str = None,
        json_output: bool = False,
    ) -> str:
        """Call the DeepSeek API with automatic continuation if response is incomplete.

        This function is specifically designed for DeepSeek models that have a max_token limit of 8k.
        It automatically continues the conversation if the response is incomplete based on start/end tags.

        Args:
            prompt: The initial prompt to send
            start_tag: Tag(s) that indicate the start of the response content (string or list of strings)
            end_tag: Tag(s) that indicate the end of the response content (string or list of strings)
            model: Model name to use (uses self.model if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens per response (for DeepSeek, will be set to 8k)
            max_continuations: Maximum number of continuation attempts
            json_schema: JSON schema for the response
            json_output: Whether to request JSON output
        Returns:
            Tuple of (conversation_messages, complete_response_string)
        """
        model = model if model else self.model

        response_format, modified_prompt, modified_max_tokens = (
            self._get_deepseek_response_format_and_prompt(
                prompt, json_schema, json_output, max_tokens, use_continuation=True
            )
        )

        # Convert single tags to lists for consistent handling
        start_tags = [start_tag] if isinstance(start_tag, str) else start_tag
        end_tags = [end_tag] if isinstance(end_tag, str) else end_tag

        start_tags = start_tags + ["```json"]
        end_tags = end_tags + ["```"]
        # Initialize conversation history
        messages = [{"role": "user", "content": modified_prompt}]
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
                "max_tokens": modified_max_tokens,
                "response_format": response_format,
            }

            try:
                content_parts = []
                for part in self._stream_with_timeout(**kwargs):
                    content_parts.append(part)
                current_response = "".join(content_parts)
            except Exception as e:
                logging.error(
                    f"Continuation call #{call_number} failed: {type(e).__name__}: {e}"
                )
                raise
            full_response += current_response

            # Check if response is complete by looking for any end tag
            response_complete = any(tag in full_response for tag in end_tags)
            if response_complete:
                # Response is complete
                logging.info(f"Response completed in {call_number} call(s)")
                break
            else:
                # Response is incomplete, continue the conversation
                continuation_count += 1
                if continuation_count >= max_continuations:
                    logging.warning(
                        f"Reached maximum continuations ({max_continuations}). Response may be incomplete."
                    )
                    break

                # Add the current response to conversation history
                messages.append({"role": "assistant", "content": current_response})

                # Add a continuation prompt
                end_tags_str = ", ".join(end_tags)
                continuation_prompt = f"Please continue your response from where you left off. Make sure to include one of the following end tags when you finish: {end_tags_str}"
                messages.append({"role": "user", "content": continuation_prompt})

        # remove start and end tag
        # print(full_response[:200])
        # print(full_response[-200:])
        clean_response = full_response

        for t in start_tags:
            clean_response = clean_response.replace(t, "")

        for t in end_tags:
            clean_response = clean_response.replace(t, "")
        # print(clean_response[:200])
        return messages, clean_response


class VLLMClient(LLMClient):
    """Client for interacting with vLLM API with optimized settings"""

    def __init__(self, endpoint: str, model: str, api_key: str = None, **kwargs):
        """Initialize vLLM client with endpoint and model.

        Args:
            endpoint: vLLM API endpoint (e.g., 'http://localhost:8000/v1')
            model: Model name to use
            api_key: API key if required (optional for local vLLM)
            **kwargs: Additional arguments passed to LLMClient
        """
        super().__init__(endpoint=endpoint, model=model, api_key=api_key, **kwargs)

    def call_with_parsed_structured_output(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        json_class: object = None,
    ) -> str:
        """Call the LLM API with structured output and timeout protection."""
        model = model if model else self.model
        if json_class is None:
            raise ValueError("a pydantic model is required")

        response_format = json_class
        last_exception = None

        for attempt in range(self.max_retries + 1):
            attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
            try:
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format=response_format,
                    timeout=self._non_streaming_httpx_timeout(),
                )
                return response.choices[0].message.parsed

            except Exception as e:
                last_exception = e

                if isinstance(e, (LLMTimeoutError, openai.APITimeoutError)):
                    logging.warning(f"Structured output timeout on {attempt_info}: {e}")
                elif isinstance(e, RETRYABLE_EXCEPTIONS):
                    logging.warning(
                        f"Structured output retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                else:
                    logging.error(
                        f"Structured output non-retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                    raise

                if attempt < self.max_retries:
                    wait_time = min(2**attempt, 10)
                    logging.info(f"Retrying structured output in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"All {self.max_retries + 1} structured output attempts failed"
                    )
                    break

        if last_exception:
            raise last_exception

        raise RuntimeError(
            "Unexpected: no response and no exception in structured output"
        )


class EmbedClient(LLMClient):
    """Client for embedding APIs using OpenAI SDK.

    Inherits from LLMClient to reuse:
    - OpenAI client initialization with API key handling
    - Retry logic with exponential backoff
    - httpx-based timeout management
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize embedding client.

        Args:
            endpoint: Embedding API base URL (e.g., 'https://api.example.com/v1')
            model: Embedding model name
            api_key: API key for authentication (optional for local services)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Initialize parent to set up OpenAI client and retry configuration
        super().__init__(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            first_token_timeout=timeout,  # Not used for embeddings
        )

    def get_embeddings(
        self, texts: List[str], model: str = None, timeout: float = None
    ) -> np.ndarray:
        """Get embeddings with retry logic.

        Args:
            texts: List of texts to embed
            model: Model name (defaults to self.model)
            timeout: Timeout in seconds (defaults to self.timeout)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        model = model if model else self.model
        effective_timeout = timeout if timeout else self.timeout

        embed_timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=effective_timeout,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )

        last_exception = None

        for attempt in range(self.max_retries + 1):
            attempt_info = f"attempt {attempt + 1}/{self.max_retries + 1}"
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=texts,
                    timeout=embed_timeout,
                )

                embeddings = np.array(
                    [data.embedding for data in response.data], dtype=np.float64
                )
                return embeddings

            except Exception as e:
                last_exception = e

                if isinstance(e, (LLMTimeoutError, openai.APITimeoutError)):
                    logging.warning(f"Embedding timeout on {attempt_info}: {e}")
                elif isinstance(e, RETRYABLE_EXCEPTIONS):
                    logging.warning(
                        f"Embedding retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                else:
                    logging.error(
                        f"Embedding non-retryable error on {attempt_info}: {type(e).__name__}: {e}"
                    )
                    raise

                if attempt < self.max_retries:
                    wait_time = min(2**attempt, 10)
                    logging.info(f"Retrying embeddings in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"All {self.max_retries + 1} embedding attempts failed"
                    )
                    break

        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected: no embeddings response and no exception")


if __name__ == "__main__":
    client = LLMClient(
        endpoint="https://llm.graphia-ssh.eu",
        model="DeepSeek-V3.1",
        api_key=os.getenv("LITELLM_API_KEY"),
    )
    response = client.call("Hello, how are you?")
    print(response)
