"""AutoGen Magentic-One C0 Baseline Adapter with strict LLM Parity.

Runs the multi-agent Magentic-One orchestration baseline using the exact same:
- Model: 'gpt-4o-mini'
- Rate limiting & timing instrumentation
- Access to the 8 deterministic GAS worker services
- Identical step ceilings and input datasets
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gas_server.core.llm_client import (
    build_llm_client,
    get_llm_timing_stats,
    reset_llm_timing_stats,
    InfrastructureError,
)
from gas_server.core.service_registry import get_service_registration

logger = logging.getLogger(__name__)


class MagenticOneC0Adapter:
    """Magentic-One Multi-Agent Loop Adapter for GAS Benchmark Condition C0."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = build_llm_client(service_name="Magentic-One-C0", model=model)
        if self.client is None:
            raise InfrastructureError("Could not initialize LLM client for Magentic-One C0 adapter.")

    def run(
        self,
        query: str,
        input_dataset_paths: List[str] | None = None,
        max_rounds: int = 10,
    ) -> Dict[str, Any]:
        """Execute task using the Magentic-One multi-agent iterative loop."""
        start_time = time.time()
        worker_start_accum = 0.0
        input_tokens = 0
        output_tokens = 0
        model_calls = 0

        # Available tool services
        available_tools = [
            "data_retrieval_agent",
            "vector_analysis_a",
            "vector_analysis_b",
            "raster_analysis_agent",
            "projection_agent",
            "conversion_agent",
            "statistics_agent",
            "mapping_agent",
        ]

        # Multi-agent prompt representing Magentic-One Orchestrator + WebSurfer / Tool Agents
        system_prompt = (
            "You are the Magentic-One multi-agent orchestrator. Your role is to coordinate "
            "specialized worker agents to solve complex geospatial analysis goals.\n"
            "Available tools/services:\n"
            + "\n".join(f"- {t}" for t in available_tools)
            + "\n\nCoordinate the workers iteratively. Return exactly one JSON object each round. "
            "To call a worker use {\"action\":\"CALL_WORKER\",\"agent_id\":\"...\","
            "\"instructions\":\"...\",\"parameters\":{...}}. When the goal is complete use "
            "{\"action\":\"FINAL_ANSWER\",\"summary\":\"...\"}. Use only listed agents and "
            "do not claim an artifact exists unless a worker returned it."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Goal: {query}\nInput Datasets: "
                    + json.dumps(
                        [
                            {
                                "name": Path(path).name,
                                "format": Path(path).suffix.lower().lstrip("."),
                            }
                            for path in (input_dataset_paths or [])
                        ]
                    )
                ),
            },
        ]

        steps_executed = []
        current_artifacts = list(input_dataset_paths or [])
        produced_artifacts: list[str] = []
        final_summary = ""

        for round_idx in range(1, max_rounds + 1):
            # Orchestrator thinking and step selection via the fixed evaluation model.
            resp = self.client.create(
                messages=messages,
                model=self.model,
                temperature=0.0,
            )
            model_calls += 1
            usage = getattr(resp, "usage", None)
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            assistant_text = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            action_payload: dict[str, Any] = {}
            candidate = assistant_text.strip()
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
            if fenced:
                candidate = fenced.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    action_payload = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                action_payload = {}

            action = str(action_payload.get("action") or "").upper()
            if action == "FINAL_ANSWER" or "FINAL_ANSWER" in assistant_text:
                final_summary = str(action_payload.get("summary") or assistant_text)
                break

            # Find matching tool invocation
            selected_tool = str(action_payload.get("agent_id") or "") or None
            if selected_tool not in available_tools:
                selected_tool = None
            for tool_id in available_tools:
                if selected_tool is None and tool_id in assistant_text.lower():
                    selected_tool = tool_id
                    break

            if not selected_tool:
                selected_tool = "data_retrieval_agent" if round_idx == 1 else "vector_analysis_a"

            # Execute tool worker through standard GAS service entry point
            w_t0 = time.time()
            reg = get_service_registration(selected_tool)
            worker = reg.build_agent()
            instruction = str(action_payload.get("instructions") or assistant_text or query)
            parameters = action_payload.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            worker_result = worker.run_service(
                query=instruction,
                input_dataset_paths=current_artifacts,
                parameters=parameters,
            )
            w_dur = time.time() - w_t0
            worker_start_accum += w_dur

            new_artifacts = worker_result.get("artifacts", [])
            if new_artifacts:
                current_artifacts.extend(new_artifacts)
                produced_artifacts.extend(new_artifacts)

            step_record = {
                "step_index": round_idx,
                "agent_id": selected_tool,
                "instruction": instruction[:200],
                "parameters": parameters,
                "artifacts_produced": new_artifacts,
                "error": worker_result.get("error"),
                "duration_seconds": round(w_dur, 3),
            }
            steps_executed.append(step_record)

            # Feed tool response back into conversation loop
            tool_msg = f"Tool '{selected_tool}' executed. Output: {worker_result.get('summary') or worker_result.get('error')}. Artifacts: {new_artifacts}"
            messages.append({"role": "user", "content": tool_msg})

            if worker_result.get("error"):
                # Pass error back to orchestrator for Magentic-One retry loop
                continue

        total_wall_time = time.time() - start_time
        timing_stats = get_llm_timing_stats()

        successful = bool(produced_artifacts) and not any(
            step.get("error") for step in steps_executed[-1:]
        )
        return {
            "status": "successful" if successful else "failed",
            "duration_seconds": round(total_wall_time, 3),
            "worker_execution_time": round(worker_start_accum, 3),
            "rate_limit_wait_time": timing_stats["rate_limit_wait_time"],
            "provider_retry_wait_time": timing_stats["provider_retry_wait_time"],
            "orchestration_active_time": round(total_wall_time - timing_stats["rate_limit_wait_time"] - timing_stats["provider_retry_wait_time"] - worker_start_accum, 3),
            "model_calls": model_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "outputs": {
                "summary": final_summary or "Magentic-One execution complete.",
                "artifacts": produced_artifacts,
                "produced_artifacts": produced_artifacts,
                "workflow_steps": steps_executed,
            },
            "steps_executed": steps_executed,
        }
