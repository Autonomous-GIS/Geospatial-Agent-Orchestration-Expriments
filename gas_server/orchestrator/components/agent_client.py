"""Adapter from the public agent's existing LLM client to component contracts."""

from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any, Callable

from gas_server.orchestrator.components.common import ComponentResponse


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\\\/]")


def _sanitize_provider_value(value: Any) -> Any:
    """Remove local path disclosure at the final provider serialization boundary."""
    if isinstance(value, dict):
        return {key: _sanitize_provider_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(value)
    ):
        normalized = value.replace("\\", "/").rstrip("/")
        return normalized.rsplit("/", 1)[-1] or "[local-path]"
    return value


class AgentComponentClient:
    """Use separate stateless calls through the existing agent-owned LLM client."""

    def __init__(
        self,
        *,
        chat_completion: Callable[[list[dict[str, str]]], Any],
        extract_json: Callable[[str], dict[str, Any]],
        model_name: Callable[[], str],
        provider_name: str = "configured_llm_provider",
    ):
        self.chat_completion = chat_completion
        self.extract_json = extract_json
        self.model_name = model_name
        self.provider_name = provider_name

    def complete(
        self,
        *,
        component: str,
        system_prompt: str,
        input_payload: dict[str, Any],
        output_schema: dict[str, Any],
        retry_limit: int,
    ) -> ComponentResponse:
        started = monotonic()
        last_error: Exception | None = None
        safe_input_payload = _sanitize_provider_value(input_payload)
        for attempt in range(1, retry_limit + 2):
            try:
                response = self.chat_completion(
                    [
                        {
                            "role": "system",
                            "content": system_prompt
                            + "\nThe output must validate against this JSON Schema:\n"
                            + json.dumps(output_schema, separators=(",", ":")),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"component": component, "input": safe_input_payload},
                                separators=(",", ":"),
                                default=str,
                            ),
                        },
                    ]
                )
                content = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                return ComponentResponse(
                    payload=self.extract_json(content),
                    model=self.model_name(),
                    provider=self.provider_name,
                    input_tokens=(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                    output_tokens=(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                    duration_seconds=monotonic() - started,
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
        raise ValueError(
            f"{component} component failed after {retry_limit + 1} attempt(s): {last_error}"
        )
