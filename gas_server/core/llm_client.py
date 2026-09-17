"""OpenAI LLM client for GAS orchestration and benchmark execution.

Supports:
- The fixed evaluation model ``gpt-4o-mini`` via OpenAI
Enforces strict rate limiting, explicit socket timeouts, and detailed latency breakdown:
- Sliding-window token bucket rate limiter (RPM, TPM, RPD)
- Explicit 420s HTTP socket timeout to prevent indefinite network hangs
- Exponential backoff with jitter on HTTP 429 / ResourceExhausted / RateLimit errors
- Fine-grained timing metrics: rate_limit_wait_time, provider_retry_wait_time
- Infrastructure error distinction (InfrastructureError)
"""

from __future__ import annotations

import os
import json
import logging
import random
import re
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "gpt-4o-mini"


class InfrastructureError(RuntimeError):
    """Raised when an unrecoverable provider/network failure occurs (429, 5xx, timeout)."""
    pass


# ==============================================================================
# Rate Limiter & Fine-Grained Timing Tracker
# ==============================================================================

class RateLimiter:
    """Sliding-window token bucket rate limiter with fine-grained timing accounting."""

    def __init__(
        self,
        max_rpm: int = 60,
        max_tpm: int = 200000,
        min_interval_seconds: float = 0.5,
    ):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.min_interval = min_interval_seconds
        self._request_timestamps: deque[float] = deque()
        self._token_records: deque[tuple[float, int]] = deque()
        self._last_call_time: float = 0.0

        # Trajectory-level timing metrics (seconds)
        self.total_rate_limit_wait_time: float = 0.0
        self.total_provider_retry_wait_time: float = 0.0

    def reset_timing(self) -> None:
        """Reset per-trajectory timing accumulators."""
        self.total_rate_limit_wait_time = 0.0
        self.total_provider_retry_wait_time = 0.0

    def get_timing(self) -> Dict[str, float]:
        """Return current timing stats."""
        return {
            "rate_limit_wait_time": round(self.total_rate_limit_wait_time, 3),
            "provider_retry_wait_time": round(self.total_provider_retry_wait_time, 3),
        }

    def acquire(self, estimated_tokens: int = 500) -> float:
        """Block until both RPM and TPM budgets allow a new request. Returns time slept."""
        slept = 0.0

        while True:
            now = time.time()

            # Clean up records older than 60 seconds
            cutoff = now - 60.0
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()
            while self._token_records and self._token_records[0][0] < cutoff:
                self._token_records.popleft()

            # 1. Enforce minimum inter-call spacing
            elapsed_since_last = now - self._last_call_time
            if elapsed_since_last < self.min_interval:
                sleep_time = self.min_interval - elapsed_since_last
                time.sleep(sleep_time)
                slept += sleep_time
                continue

            # 2. Check RPM
            if len(self._request_timestamps) >= self.max_rpm:
                oldest = self._request_timestamps[0]
                sleep_time = (oldest + 60.0) - now + 0.1
                if sleep_time > 0:
                    logger.info("RPM limit reached (%d req/min). Pacing for %.2fs", len(self._request_timestamps), sleep_time)
                    time.sleep(sleep_time)
                    slept += sleep_time
                    continue

            # 3. Check TPM
            current_tpm = sum(tokens for _, tokens in self._token_records)
            if current_tpm + estimated_tokens >= self.max_tpm:
                if self._token_records:
                    oldest_t_time = self._token_records[0][0]
                    sleep_time = (oldest_t_time + 60.0) - now + 0.5
                    if sleep_time > 0:
                        logger.info("TPM limit approaching (%d + %d >= %d). Pacing for %.2fs", current_tpm, estimated_tokens, self.max_tpm, sleep_time)
                        time.sleep(sleep_time)
                        slept += sleep_time
                        continue

            # All budget constraints satisfied
            self._last_call_time = time.time()
            self._request_timestamps.append(self._last_call_time)
            break

        self.total_rate_limit_wait_time += slept
        return slept

    def record_usage(self, actual_tokens: int) -> None:
        """Record the actual tokens used for a completed request."""
        self._token_records.append((time.time(), actual_tokens))

    def record_retry_wait(self, wait_seconds: float) -> None:
        """Record time spent in backoff waiting for provider retry."""
        self.total_provider_retry_wait_time += wait_seconds


# Global rate limiter instance
_GLOBAL_RATE_LIMITER = RateLimiter(max_rpm=60, max_tpm=200000, min_interval_seconds=0.5)


def get_llm_timing_stats() -> Dict[str, float]:
    return _GLOBAL_RATE_LIMITER.get_timing()


def reset_llm_timing_stats() -> None:
    _GLOBAL_RATE_LIMITER.reset_timing()


# ==============================================================================
# OpenAI Client Wrapper
# ==============================================================================

class OpenAIClientWrapper:
    """Wraps OpenAI client with rate limiting, exponential backoff, and timing breakdown."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        if model != DEFAULT_MODEL:
            raise ValueError(f"All LLM calls must use {DEFAULT_MODEL!r}; received {model!r}.")
        self.model = DEFAULT_MODEL
        self._client = None
        self._init_client()
        self.chat = SimpleNamespace(completions=self)

    def _init_client(self):
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, timeout=420.0)
        except Exception as e:
            logger.warning("Could not initialize openai Client: %s", e)
            self._client = None

    def create(
        self,
        messages: List[Dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Generate response via OpenAI with rate-limiting, timing metrics, and backoff."""
        if model is not None and model != DEFAULT_MODEL:
            raise ValueError(f"All LLM calls must use {DEFAULT_MODEL!r}; received {model!r}.")
        chosen_model = DEFAULT_MODEL

        # Prepare messages
        call_messages = list(messages or [])
        if not call_messages and "prompt" in kwargs:
            call_messages = [{"role": "user", "content": kwargs["prompt"]}]

        est_tokens = max(100, sum(len(m.get("content", "")) for m in call_messages) // 4)

        last_error = None
        for attempt in range(max_retries):
            _GLOBAL_RATE_LIMITER.acquire(estimated_tokens=est_tokens)

            try:
                if self._client is None:
                    self._init_client()
                if self._client is None:
                    raise InfrastructureError("openai client is not available or failed initialization.")

                response = self._client.chat.completions.create(
                    model=chosen_model,
                    messages=call_messages,
                    temperature=temperature,
                )

                total_tokens = est_tokens
                if hasattr(response, "usage") and response.usage:
                    total_tokens = getattr(response.usage, "total_tokens", est_tokens)
                _GLOBAL_RATE_LIMITER.record_usage(total_tokens)

                return response

            except Exception as e:
                err_str = str(e)
                last_error = e
                is_transient_infra = (
                    "429" in err_str
                    or "rate_limit" in err_str.lower()
                    or "quota" in err_str.lower()
                    or re.search(r"\b5\d{2}\b", err_str) is not None
                    or "timeout" in err_str.lower()
                    or "connection" in err_str.lower()
                    or "timed out" in err_str.lower()
                )

                if is_transient_infra and attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 3.0 + random.uniform(0.5, 2.0)
                    logger.warning("OpenAI provider retry %d/%d (backing off %.2fs): %s", attempt + 1, max_retries, backoff, err_str)
                    _GLOBAL_RATE_LIMITER.record_retry_wait(backoff)
                    time.sleep(backoff)
                else:
                    if attempt == max_retries - 1 and is_transient_infra:
                        logger.error("OpenAI provider infrastructure failure after %d retries: %s", max_retries, err_str)
                        raise InfrastructureError(f"Provider infrastructure failure after {max_retries} attempts: {err_str}") from e
                    raise

        raise InfrastructureError("Failed to generate content from OpenAI.") from last_error


# ==============================================================================
# Helper functions & Client Factories
# ==============================================================================

def format_service_name(name: str | None) -> str:
    """Normalize an agent name for LLM-provider logging/metadata."""
    if not name:
        return "Agent"
    normalized = name.replace("_", " ").strip()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_llm_client(
    *,
    service_name: str,
    gemini_api_key: str | None = None,
    google_api_key: str | None = None,
    openai_api_key: str | None = None,
    gibd_api_key: str | None = None,
    model: str | None = None,
) -> Any | None:
    """Create a request-time client locked to the evaluation model."""
    requested_model = model or os.getenv("BENCHMARK_MODEL") or DEFAULT_MODEL
    if requested_model != DEFAULT_MODEL:
        raise ValueError(
            f"All LLM calls must use {DEFAULT_MODEL!r}; received {requested_model!r}."
        )
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None
    return OpenAIClientWrapper(api_key=openai_key, model=DEFAULT_MODEL)


def configure_agent_client(
    agent: Any,
    *,
    service_name: str,
    gemini_api_key: str | None = None,
    google_api_key: str | None = None,
    openai_api_key: str | None = None,
    gibd_api_key: str | None = None,
) -> Any:
    """Attach request-time LLM client to an agent instance."""
    client = build_llm_client(
        service_name=service_name,
        gemini_api_key=gemini_api_key,
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        gibd_api_key=gibd_api_key,
        model=getattr(agent, "model", None),
    )
    if client is None:
        raise ValueError(
            "Request must include OPENAI_API_KEY."
        )

    agent.client = client
    agent.service_name = format_service_name(service_name)
    return client
