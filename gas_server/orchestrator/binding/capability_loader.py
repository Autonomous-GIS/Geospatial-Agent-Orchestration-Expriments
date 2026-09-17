"""Normalize DescribeAgent cards into binder capabilities and Planner snapshots."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gas_server.orchestrator.binding.binder import WorkerCapability


def _operation_names(card: dict[str, Any]) -> list[str]:
    parameters = card.get("execute_task", {}).get("parameters", {})
    operation = parameters.get("operation", {}) if isinstance(parameters, dict) else {}
    description = str(operation.get("description") or "")
    parenthetical = re.search(r"\(([^)]+)\)", description)
    names: list[str] = []
    if parenthetical:
        for value in parenthetical.group(1).split(","):
            normalized = value.strip().lower().replace(" ", "_")
            if normalized:
                names.append(normalized)
    for skill in card.get("skills", []):
        skill_id = str(skill.get("skill_id") or "").strip().lower()
        if skill_id:
            names.append(skill_id)
    return sorted(set(names))


def load_local_capabilities(capability_dir: str | Path) -> tuple[list[WorkerCapability], list[dict[str, Any]]]:
    capabilities: list[WorkerCapability] = []
    snapshots: list[dict[str, Any]] = []
    for path in sorted(Path(capability_dir).glob("*.json")):
        if path.name == "capabilities.json":
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        profile = card.get("profile", {})
        agent_id = str(profile.get("agent_id") or path.stem)
        if agent_id == "autonmous_gis_orchestrator":
            continue
        operations = _operation_names(card)
        outputs = card.get("execute_task", {}).get("outputs", {}).get("primary_artifacts", [])
        formats = sorted(
            {
                str(value).lower()
                for item in outputs
                for value in (item.get("formats", []) if isinstance(item, dict) else [])
            }
        )
        snapshot = {
            "agent_id": agent_id,
            "name": profile.get("name"),
            "description": profile.get("description"),
            "operations": operations,
            "skills": card.get("skills", []),
            "inputs": card.get("execute_task", {}).get("inputs", {}),
            "outputs": outputs,
            "parameters": card.get("execute_task", {}).get("parameters", {}),
        }
        snapshots.append(snapshot)
        capabilities.append(
            WorkerCapability(
                agent_id=agent_id,
                endpoint=(
                    str(profile.get("base_url") or "").rstrip("/") + "/tasks"
                    if profile.get("base_url")
                    else None
                ),
                operations=operations,
                input_formats=formats,
                profile_snapshot=snapshot,
            )
        )
    return capabilities, snapshots
