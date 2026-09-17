"""Project WorkflowState to legacy workflow/canvas structures."""

from __future__ import annotations

import json
from pathlib import Path

from gas_server.orchestrator.core.models import WorkflowState


def workflow_plan_projection(state: WorkflowState) -> dict:
    steps = []
    for task in state.tasks.values():
        binding = state.bindings.get(task.active_binding_id or "")
        steps.append(
            {
                "step_id": task.task_id,
                "title": task.title,
                "purpose": task.purpose,
                "agent_id": binding.agent_id if binding else task.metadata.get("preferred_agent_id"),
                "operation": task.operation,
                "instructions": task.instructions,
                "parameters": task.parameters,
                "depends_on": task.depends_on,
                "input_from_steps": task.depends_on,
                "status": task.status.value,
                "output_artifact_ids": task.output_artifact_ids,
            }
        )
    return {
        "workflow_id": state.workflow_id,
        "plan_version": state.current_plan_version,
        "goal": state.user_goal,
        "workflow_steps": steps,
        "requirements": [item.model_dump(mode="json") for item in state.requirements.values()],
        "commitments": [item.model_dump(mode="json") for item in state.commitments.values()],
    }


def canvas_projection(state: WorkflowState) -> dict:
    nodes = []
    for index, task in enumerate(state.tasks.values()):
        binding = state.bindings.get(task.active_binding_id or "")
        nodes.append(
            {
                "id": task.task_id,
                "name": task.title,
                "agentId": binding.agent_id if binding else task.metadata.get("preferred_agent_id"),
                "status": task.status.value,
                "instructions": task.instructions,
                "parameters": task.parameters,
                "input_from_steps": task.depends_on,
                "outputArtifacts": [
                    {
                        "artifact_id": artifact_id,
                        "url": state.artifacts[artifact_id].location,
                    }
                    for artifact_id in task.output_artifact_ids
                    if artifact_id in state.artifacts
                ],
                "position": {"x": 100 + (index % 4) * 520, "y": 100 + (index // 4) * 300},
            }
        )
    return {
        "workflow_id": state.workflow_id,
        "state_version": state.state_version,
        "plan_version": state.current_plan_version,
        "workflow_status": state.terminal_state.value if state.terminal_state else "running",
        "nodes": nodes,
        "connections": [
            {"sourceId": source, "targetId": target}
            for source, target in state.edges
        ],
    }


def write_canvas_projection(state: WorkflowState, directory: str | Path) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"workflow_{state.workflow_id}.json"
    path.write_text(json.dumps(canvas_projection(state), indent=2), encoding="utf-8")
    return path
