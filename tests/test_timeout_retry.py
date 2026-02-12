"""
Tests for the timeout and retry logic in the LLM client.

Covers:
- httpx.Timeout configuration (streaming vs non-streaming)
- _stream_with_timeout: per-chunk + wall-clock enforcement
- _call_with_retry: retryable vs non-retryable error classification
- Config-level queue-timeout hierarchy validation
"""

import logging
import sys
import os
import time

import httpx
import openai
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.citation_index.llm.client import (
    LLMClient,
    LLMTimeoutError,
    RETRYABLE_EXCEPTIONS,
    CONNECT_TIMEOUT,
    WRITE_TIMEOUT,
    POOL_TIMEOUT,
)


# ========================
# Helpers
# ========================

def _make_client(**overrides) -> LLMClient:
    """Create an LLMClient with sensible test defaults."""
    defaults = dict(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        timeout=10.0,
        max_retries=2,
        first_token_timeout=3.0,
    )
    defaults.update(overrides)
    return LLMClient(**defaults)


def _make_chunk(content: str | None) -> Mock:
    """Build a single mock streaming chunk."""
    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta.content = content
    return chunk


def _make_stream(contents: list[str | None]) -> Mock:
    """Build a mock iterable stream that also has a .close() method."""
    chunks = [_make_chunk(c) for c in contents]
    stream = MagicMock()
    stream.__iter__ = Mock(return_value=iter(chunks))
    stream.close = Mock()
    return stream


def _make_non_streaming_response(content: str) -> Mock:
    """Build a mock non-streaming chat completion response."""
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = content
    return resp


# ========================
# httpx.Timeout configuration
# ========================

class TestHttpxTimeoutConfig:
    """Verify that the correct httpx.Timeout objects are built."""

    def test_streaming_timeout_uses_first_token_timeout_as_read(self):
        client = _make_client(first_token_timeout=42.0)
        t = client._streaming_httpx_timeout()

        assert t.read == 42.0
        assert t.connect == CONNECT_TIMEOUT
        assert t.write == WRITE_TIMEOUT
        assert t.pool == POOL_TIMEOUT

    def test_non_streaming_timeout_uses_total_timeout_as_read(self):
        client = _make_client(timeout=99.0)
        t = client._non_streaming_httpx_timeout()

        assert t.read == 99.0
        assert t.connect == CONNECT_TIMEOUT
        assert t.write == WRITE_TIMEOUT
        assert t.pool == POOL_TIMEOUT

    def test_client_level_default_timeout(self):
        """The OpenAI client itself gets a generous default read = total timeout."""
        client = _make_client(timeout=120.0)
        # The openai SDK stores timeout on client._client (httpx.Client)
        # We just verify the client was created without error
        assert client.client is not None

    def test_streaming_and_non_streaming_differ(self):
        client = _make_client(timeout=180.0, first_token_timeout=30.0)
        s = client._streaming_httpx_timeout()
        ns = client._non_streaming_httpx_timeout()

        assert s.read == 30.0
        assert ns.read == 180.0
        # connect/write/pool are identical
        assert s.connect == ns.connect
        assert s.write == ns.write
        assert s.pool == ns.pool


# ========================
# _stream_with_timeout
# ========================

class TestStreamWithTimeout:
    """Test per-chunk and wall-clock timeout enforcement during streaming."""

    def test_yields_content_from_chunks(self):
        client = _make_client(timeout=60.0)
        stream = _make_stream(["Hello", " ", "world"])

        with patch.object(client.client.chat.completions, "create", return_value=stream):
            parts = list(client._stream_with_timeout(model="m", messages=[]))

        assert parts == ["Hello", " ", "world"]

    def test_skips_none_content(self):
        client = _make_client(timeout=60.0)
        stream = _make_stream([None, "data", None])

        with patch.object(client.client.chat.completions, "create", return_value=stream):
            parts = list(client._stream_with_timeout(model="m", messages=[]))

        assert parts == ["data"]

    def test_passes_streaming_timeout_to_create(self):
        client = _make_client(first_token_timeout=7.5)
        stream = _make_stream(["ok"])
        mock_create = Mock(return_value=stream)

        with patch.object(client.client.chat.completions, "create", mock_create):
            list(client._stream_with_timeout(model="m", messages=[]))

        # Verify the timeout kwarg passed to create()
        _, kwargs = mock_create.call_args
        assert kwargs["stream"] is True
        to = kwargs["timeout"]
        assert isinstance(to, httpx.Timeout)
        assert to.read == 7.5

    def test_raises_LLMTimeoutError_on_wall_clock_exceeded(self):
        """If total elapsed > self.timeout between chunks, raise."""
        client = _make_client(timeout=0.0)  # instant wall-clock limit

        # We need at least one chunk to trigger the check
        stream = _make_stream(["a"])

        with patch.object(client.client.chat.completions, "create", return_value=stream):
            with pytest.raises(LLMTimeoutError, match="wall-clock timeout"):
                # Force a tiny delay so elapsed > 0
                list(client._stream_with_timeout(model="m", messages=[]))

    def test_closes_stream_on_success(self):
        client = _make_client(timeout=60.0)
        stream = _make_stream(["ok"])

        with patch.object(client.client.chat.completions, "create", return_value=stream):
            list(client._stream_with_timeout(model="m", messages=[]))

        stream.close.assert_called_once()

    def test_closes_stream_on_error(self):
        """Stream must be closed even when an exception propagates."""
        client = _make_client(timeout=60.0)

        # Stream whose iteration raises
        stream = MagicMock()
        stream.__iter__ = Mock(side_effect=openai.APITimeoutError(request=Mock()))
        stream.close = Mock()

        with patch.object(client.client.chat.completions, "create", return_value=stream):
            with pytest.raises(openai.APITimeoutError):
                list(client._stream_with_timeout(model="m", messages=[]))

        stream.close.assert_called_once()

    def test_propagates_openai_api_timeout(self):
        """httpx read timeout surfaces as openai.APITimeoutError."""
        client = _make_client()
        exc = openai.APITimeoutError(request=Mock())

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(openai.APITimeoutError):
                list(client._stream_with_timeout(model="m", messages=[]))


# ========================
# _call_with_retry – retryable vs non-retryable
# ========================

class TestCallWithRetryClassification:
    """Verify that retryable errors are retried and others fail fast."""

    @pytest.fixture(autouse=True)
    def _patch_sleep(self):
        """Eliminate real sleeps during retry back-off."""
        with patch("src.citation_index.llm.client.time.sleep") as mock_sleep:
            self.mock_sleep = mock_sleep
            yield

    # ---- retryable errors ----

    def test_retries_on_LLMTimeoutError(self):
        client = _make_client(max_retries=2)
        stream_ok = _make_stream(["ok"])

        effects = [
            LLMTimeoutError("timeout 1"),
            LLMTimeoutError("timeout 2"),
            stream_ok,
        ]
        with patch.object(client.client.chat.completions, "create", side_effect=effects):
            result = client.call("hello")

        assert result == "ok"
        assert self.mock_sleep.call_count == 2

    def test_retries_on_openai_APITimeoutError(self):
        client = _make_client(max_retries=1)
        exc = openai.APITimeoutError(request=Mock())
        stream_ok = _make_stream(["ok"])

        with patch.object(
            client.client.chat.completions, "create", side_effect=[exc, stream_ok]
        ):
            result = client.call("hello")

        assert result == "ok"
        assert self.mock_sleep.call_count == 1

    def test_retries_on_openai_APIConnectionError(self):
        client = _make_client(max_retries=1)
        exc = openai.APIConnectionError(request=Mock())
        stream_ok = _make_stream(["done"])

        with patch.object(
            client.client.chat.completions, "create", side_effect=[exc, stream_ok]
        ):
            assert client.call("hello") == "done"

    def test_retries_on_openai_RateLimitError(self):
        client = _make_client(max_retries=1)
        exc = openai.RateLimitError(
            message="rate limited",
            response=Mock(status_code=429, headers={}),
            body=None,
        )
        stream_ok = _make_stream(["ok"])

        with patch.object(
            client.client.chat.completions, "create", side_effect=[exc, stream_ok]
        ):
            assert client.call("hello") == "ok"

    def test_retries_on_openai_InternalServerError(self):
        client = _make_client(max_retries=1)
        exc = openai.InternalServerError(
            message="server error",
            response=Mock(status_code=500, headers={}),
            body=None,
        )
        stream_ok = _make_stream(["ok"])

        with patch.object(
            client.client.chat.completions, "create", side_effect=[exc, stream_ok]
        ):
            assert client.call("hello") == "ok"

    def test_raises_after_exhausting_retries(self):
        client = _make_client(max_retries=1)
        exc = LLMTimeoutError("always fails")

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(LLMTimeoutError, match="always fails"):
                client.call("hello")

    # ---- non-retryable errors ----

    def test_does_not_retry_ValueError(self):
        client = _make_client(max_retries=3)

        with patch.object(
            client.client.chat.completions, "create",
            side_effect=ValueError("bad input"),
        ):
            with pytest.raises(ValueError, match="bad input"):
                client.call("hello")

        # sleep should never have been called (no retry)
        assert self.mock_sleep.call_count == 0

    def test_does_not_retry_openai_BadRequestError(self):
        client = _make_client(max_retries=3)
        exc = openai.BadRequestError(
            message="bad request",
            response=Mock(status_code=400, headers={}),
            body=None,
        )

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(openai.BadRequestError):
                client.call("hello")

        assert self.mock_sleep.call_count == 0

    def test_does_not_retry_openai_AuthenticationError(self):
        client = _make_client(max_retries=3)
        exc = openai.AuthenticationError(
            message="invalid key",
            response=Mock(status_code=401, headers={}),
            body=None,
        )

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(openai.AuthenticationError):
                client.call("hello")

        assert self.mock_sleep.call_count == 0


# ========================
# _call_with_retry – backoff & timeout passing
# ========================

class TestCallWithRetryMechanics:

    @pytest.fixture(autouse=True)
    def _patch_sleep(self):
        with patch("src.citation_index.llm.client.time.sleep") as mock_sleep:
            self.mock_sleep = mock_sleep
            yield

    def test_exponential_backoff_timing(self):
        """Backoff should be 1, 2, 4, … capped at 10."""
        client = _make_client(max_retries=4)
        exc = LLMTimeoutError("timeout")

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(LLMTimeoutError):
                client.call("hello")

        waits = [c.args[0] for c in self.mock_sleep.call_args_list]
        assert waits == [1, 2, 4, 8]  # 2^0, 2^1, 2^2, 2^3 (all < 10 cap)

    def test_backoff_capped_at_10(self):
        client = _make_client(max_retries=5)
        exc = LLMTimeoutError("timeout")

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(LLMTimeoutError):
                client.call("hello")

        waits = [c.args[0] for c in self.mock_sleep.call_args_list]
        # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16→cap=10
        assert waits == [1, 2, 4, 8, 10]

    def test_non_streaming_passes_non_streaming_timeout(self):
        client = _make_client(timeout=55.0)
        resp = _make_non_streaming_response("result")
        mock_create = Mock(return_value=resp)

        with patch.object(client.client.chat.completions, "create", mock_create):
            result = client.call("hello", use_streaming=False)

        assert result == "result"

        _, kwargs = mock_create.call_args
        to = kwargs["timeout"]
        assert isinstance(to, httpx.Timeout)
        assert to.read == 55.0  # non-streaming uses total timeout

    def test_streaming_passes_streaming_timeout(self):
        client = _make_client(first_token_timeout=12.0)
        stream = _make_stream(["ok"])
        mock_create = Mock(return_value=stream)

        with patch.object(client.client.chat.completions, "create", mock_create):
            result = client.call("hello", use_streaming=True)

        assert result == "ok"

        _, kwargs = mock_create.call_args
        to = kwargs["timeout"]
        assert isinstance(to, httpx.Timeout)
        assert to.read == 12.0  # streaming uses first_token_timeout

    def test_max_retries_zero_means_single_attempt(self):
        client = _make_client(max_retries=0)
        exc = LLMTimeoutError("once")

        with patch.object(
            client.client.chat.completions, "create", side_effect=exc
        ):
            with pytest.raises(LLMTimeoutError, match="once"):
                client.call("hello")

        assert self.mock_sleep.call_count == 0  # no retries, no sleep


# ========================
# Config – queue timeout hierarchy validation
# ========================

class TestConfigTimeoutValidation:
    """Verify Settings warns when queue timeouts are too small."""

    def test_warns_when_queue_timeout_too_small(self, caplog):
        from src.citation_index.config import Settings

        with caplog.at_level(logging.WARNING, logger="src.citation_index.config"):
            s = Settings(
                llm_timeout=300.0,
                llm_max_retries=3,
                llm_timeout_reference_parsing=600.0,
                timeout_reference_extraction=100,   # way too small
                timeout_reference_parsing=100,       # way too small
            )

        assert "timeout_reference_extraction" in caplog.text
        assert "timeout_reference_parsing" in caplog.text

    def test_no_warning_when_queue_timeouts_large_enough(self, caplog):
        from src.citation_index.config import Settings

        with caplog.at_level(logging.WARNING, logger="src.citation_index.config"):
            s = Settings(
                llm_timeout=180.0,
                llm_max_retries=3,
                llm_timeout_reference_parsing=300.0,
                timeout_reference_extraction=9999,
                timeout_reference_parsing=9999,
            )

        # No warning should appear for these two settings
        assert "timeout_reference_extraction" not in caplog.text
        assert "timeout_reference_parsing" not in caplog.text


# ========================
# RETRYABLE_EXCEPTIONS tuple sanity
# ========================

class TestRetryableExceptions:

    def test_contains_expected_types(self):
        assert LLMTimeoutError in RETRYABLE_EXCEPTIONS
        assert openai.APITimeoutError in RETRYABLE_EXCEPTIONS
        assert openai.APIConnectionError in RETRYABLE_EXCEPTIONS
        assert openai.RateLimitError in RETRYABLE_EXCEPTIONS
        assert openai.InternalServerError in RETRYABLE_EXCEPTIONS

    def test_does_not_contain_bad_request(self):
        assert openai.BadRequestError not in RETRYABLE_EXCEPTIONS
        assert openai.AuthenticationError not in RETRYABLE_EXCEPTIONS

    def test_isinstance_check_works(self):
        """Verify isinstance matching (used in the retry loop)."""
        assert isinstance(LLMTimeoutError("x"), RETRYABLE_EXCEPTIONS)
        assert not isinstance(ValueError("x"), RETRYABLE_EXCEPTIONS)
