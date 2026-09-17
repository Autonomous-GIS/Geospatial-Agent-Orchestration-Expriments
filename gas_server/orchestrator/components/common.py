"""Shared transport metadata, not shared conversational memory."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import Field

from gas_server.orchestrator.core.models import StrictModel


class ComponentResponse(StrictModel):
    payload: dict[str, Any]
    model: str
    provider: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    disposition: str = "successful"


class ComponentClient(Protocol):
    def complete(
        self,
        *,
        component: str,
        system_prompt: str,
        input_payload: dict[str, Any],
        output_schema: dict[str, Any],
        retry_limit: int,
    ) -> ComponentResponse: ...


def stable_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
