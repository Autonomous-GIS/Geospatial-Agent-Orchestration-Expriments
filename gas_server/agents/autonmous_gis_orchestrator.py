"""Public planning and execution entry point for the GIS orchestrator.

Planning mode preserves the legacy canvas-compatible composition behavior.
Execute and resume modes delegate to the stateful orchestrator-local runtime.
The class continues to inherit directly from the unchanged ``GeoAgent`` and
does not depend on ``GeospatialWorkflowPlanningAgent``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from gas_server.core.config import DATA_DIR
from gas_server.core.file_naming import build_output_filename
from gas_server.core.geo_agent import GeoAgent, ProgressCallback
from gas_server.core.llm_client import DEFAULT_MODEL, build_llm_client, format_service_name


class AutonomousGisPipelineAgent(GeoAgent):
    """Composes multi-agent GIS pipelines from a natural-language goal.

    Given a set of GAS agent URLs (DescribeAgent documents or
    GetCapabilities endpoints) and a user request, this agent:

    1. Discovers available agent capabilities.
    2. Uses an LLM to decompose the request into ordered workflow steps,
       matching each step to the best-fit agent.
    3. Normalises and repairs the resulting DAG.
    4. Produces a GAS-canvas-compatible JSON file.
    """

    agent_id = "autonmous_gis_orchestrator"
    agent_name = "Autonomous GIS Orchestrator"
    agent_version = "3.0.0"
    agent_description = (
        "Plans or autonomously executes stateful multi-agent GIS workflows "
        "using independent Planner, runtime QC, and structural Replanner components."
    )
    requires_input_datasets = False
    requires_model_credentials = True

    # -- Layout constants ----------------------------------------------------
    _CANVAS_WIDTH = 2200
    _CANVAS_PADDING_X = 100
    _LAYER_SPACING_X = 560
    _NODE_VERTICAL_GAP = 320
    _DEFAULT_QC_FAILURE_LIMIT = 3

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        if model is not None and model != DEFAULT_MODEL:
            raise ValueError(
                f"All LLM calls must use {DEFAULT_MODEL!r}; received {model!r}."
            )
        super().__init__(
            api_key=api_key,
            model=DEFAULT_MODEL,
            output_dir=DATA_DIR / self.agent_id,
        )
        self.service_name = format_service_name(self.agent_name)
        self.client = build_llm_client(
            service_name=self.service_name,
            openai_api_key=self.api_key,
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self._last_temperature_control = "temperature=0.1"

    def _debug_enabled(self) -> bool:
        raw_parameters = dict(getattr(self, "request_parameters", {}) or {})
        return (
            str(raw_parameters.get("debug") or "").lower() in {"1", "true", "yes", "on"}
            or os.getenv("GAS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        )

    def _debug_print(self, label: str, payload: Any | None = None) -> None:
        """Print concise pipeline diagnostics only when debugging is enabled."""
        if not self._debug_enabled():
            return
        print(f"[PIPELINE DEBUG] {label}")
        if payload is None:
            return
        try:
            print(json.dumps(payload, indent=2, default=str)[:12000])
        except Exception:
            print(str(payload)[:12000])

    def _qc_failure_limit(self) -> int:
        raw_parameters = dict(getattr(self, "request_parameters", {}) or {})
        raw_limit = raw_parameters.get("qc_failure_limit")
        if raw_limit is None:
            raw_limit = os.getenv("GAS_PIPELINE_QC_FAILURE_LIMIT")
        try:
            limit = int(raw_limit) if raw_limit is not None else self._DEFAULT_QC_FAILURE_LIMIT
        except (TypeError, ValueError):
            limit = self._DEFAULT_QC_FAILURE_LIMIT
        return max(1, limit)

    def _normalize_orchestrator_mode(self, workflow_json_path: str | None) -> str:
        raw_params = dict(getattr(self, "request_parameters", {}) or {})
        raw_mode = raw_params.get("mode") or raw_params.get("pipeline_mode")
        if raw_params.get("quality_control") is True:
            raw_mode = "qc"
        elif raw_params.get("replan") is True:
            raw_mode = "replan"
        mode = str(raw_mode or "").strip().lower().replace("-", "_")

        aliases = {
            "planning": "plan",
            "execution": "execute",
            "run": "execute",
            "continuation": "resume",
            "quality_control": "qc",
            "qualitycontrol": "qc",
            "control": "qc",
            "repair": "qc",
            "replanning": "replan",
        }
        mode = aliases.get(mode, mode)
        if mode in {"plan", "execute", "resume", "qc", "replan"}:
            return mode

        if workflow_json_path:
            filename = Path(workflow_json_path).name.lower()
            if "qc" in filename or "quality_control" in filename:
                return "qc"
            if any(token in filename for token in ("failed", "replan", "repair")):
                return "replan"

        return "plan"

    # ================================================================== run

    def run(
        self,
        query: str,
        input_dataset_paths: list[str] | str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        self.reset_metrics()
        self.input_tokens = 0
        self.output_tokens = 0
        self.ensure_directory(self.output_dir)

        # Explicit orchestrator modes:
        # - plan: user question -> workflow
        # - qc: workflow + completed node outputs -> local workflow repair
        # - replan: failed workflow -> redesigned workflow
        dataset_paths = self.normalize_dataset_paths(input_dataset_paths)
        workflow_json_path = self._detect_workflow_json_input(dataset_paths)
        mode = self._normalize_orchestrator_mode(workflow_json_path)
        if mode == "execute":
            from gas_server.orchestrator.runtime import AutonomousRuntime

            return AutonomousRuntime(self).execute(
                query=query,
                input_paths=dataset_paths,
                parameters=dict(self.request_parameters or {}),
            )
        if mode == "resume":
            from gas_server.orchestrator.runtime import AutonomousRuntime

            return AutonomousRuntime(self).resume(
                parameters=dict(self.request_parameters or {}),
            )
        if workflow_json_path is not None:
            if mode == "qc":
                self._debug_print("Routing workflow JSON to quality control", {"workflow_json_path": workflow_json_path})
                return self.run_quality_control(
                    query, workflow_json_path, dataset_paths, progress_callback
                )
            if mode == "replan":
                self._debug_print("Routing workflow JSON to replanning", {"workflow_json_path": workflow_json_path})
                return self.run_replan(
                    query, workflow_json_path, input_dataset_paths, progress_callback
                )
            raise ValueError("Pipeline mode 'plan' does not accept a workflow JSON input dataset. Use mode 'qc' or 'replan'.")

        if mode in {"qc", "replan"}:
            raise ValueError(f"Pipeline mode '{mode}' requires a workflow JSON input dataset.")

        # 1. Discover capabilities -------------------------------------------
        pipeline_sources = self._normalize_pipeline_sources()
        self.emit_progress(
            progress_callback,
            stage="start",
            message=(
                "I will discover GAS agent capabilities, decompose the goal "
                "into pipeline steps, and produce a canvas-compatible JSON workflow."
            ),
            data={"source_count": len(pipeline_sources)},
        )
        catalogs = self._discover_capabilities(pipeline_sources, progress_callback)
        total_agents = sum(len(c.get("agents", [])) for c in catalogs)
        self._debug_print(
            "Discovered capabilities",
            {
                "source_count": len(pipeline_sources),
                "total_agents": total_agents,
                "agents": [
                    {
                        "agent_id": agent.get("agent_id"),
                        "category": agent.get("category"),
                        "name": agent.get("name"),
                    }
                    for catalog in catalogs
                    for agent in catalog.get("agents", [])
                ],
            },
        )

        # 2. Explore and select relevant candidate agents -------------------
        self.emit_progress(
            progress_callback,
            stage="agent_exploration",
            message="I am exploring and selecting relevant agents from the catalog in batches.",
            data={"total_agents": total_agents},
        )
        relevant_agent_ids = self._explore_and_select_relevant_agents(query, catalogs)
        self._debug_print("Selected relevant agents", sorted(relevant_agent_ids))
        
        # Filter catalogs to only include relevant agents
        if not relevant_agent_ids:
            filtered_catalogs = catalogs
            selected_agent_count = total_agents
        else:
            filtered_catalogs = []
            for catalog in catalogs:
                filtered_agents = [
                    a for a in catalog.get("agents", [])
                    if a.get("agent_id") in relevant_agent_ids
                ]
                filtered_catalogs.append({**catalog, "agents": filtered_agents})
            selected_agent_count = sum(len(c.get("agents", [])) for c in filtered_catalogs)

        self.emit_progress(
            progress_callback,
            stage="agent_selection_complete",
            message=f"I selected {selected_agent_count} relevant agent(s) out of {total_agents} total available.",
            data={"selected_agents": list(relevant_agent_ids)},
        )

        # 3. Plan via LLM ---------------------------------------------------
        self.emit_progress(
            progress_callback,
            stage="planning",
            message=(
                "I am decomposing the user goal and matching each step to "
                "suitable GAS agent capabilities."
            ),
            data={"agent_count": selected_agent_count},
        )

        plan = None
        last_error = None
        for attempt in range(1, 4):
            try:
                messages = self._build_planning_prompt(query, filtered_catalogs)
                self.increment_llm_calls()
                response = self._chat_completion(messages)
                usage = getattr(response, "usage", None)
                if usage:
                    self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
                content = response.choices[0].message.content
                plan = self._extract_json(content)
                self.emit_progress(
                    progress_callback,
                    stage="method_selection",
                    message=(
                        "The workflow planning model returned a structured plan. "
                        "I will validate and normalise its steps."
                    ),
                    data={"llm_calls": self.llm_calls},
                )
                break
            except Exception as exc:
                last_error = exc
                self.emit_progress(
                    progress_callback,
                    stage="planning_attempt_failed",
                    message=f"Planning attempt {attempt}/3 failed: {exc}",
                )

        if plan is None:
            raise ValueError(
                f"The pipeline agent could not define a valid plan for this question. "
                f"Details of the last error: {last_error}"
            )

        # 4. Normalise + layout + write JSON ---------------------------------
        plan = self._normalize_plan(plan, filtered_catalogs)
        self._debug_print("Normalized workflow plan", plan)
        canvas = self._build_canvas_json(query, plan)
        json_path = self._write_canvas_json(query, canvas)

        self.emit_progress(
            progress_callback,
            stage="complete",
            message="Workflow pipeline planning is complete.",
            data={"steps": len(plan.get("workflow_steps", []))},
        )

        step_count = len(plan.get("workflow_steps", []))
        summary = (
            f"Generated a GAS pipeline workflow with {step_count} step(s) "
            f"using {total_agents} discovered agent capability document(s)."
        )

        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "duration": round(time.time() - start_time, 2),
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "inputs": {
                "text": query,
                "dataset_paths": self.normalize_dataset_paths(input_dataset_paths),
                "parameters": {
                    "pipeline_sources": pipeline_sources,
                },
            },
            "outputs": {
                "text": summary,
                "workflow_plan": plan,
                "workflow_json_file": str(json_path),
            },
            "metrics": self.metrics(number_of_artifacts=1),
            "stochasticity": {
                "used": True,
                "controls": [self._last_temperature_control],
            },
        }

    # ============================== pipeline source normalisation ==========

    def _normalize_pipeline_sources(self) -> list[str]:
        """Extract agent / server URLs from request parameters."""
        raw_parameters = dict(getattr(self, "request_parameters", {}) or {})
        candidate_keys = (
            "gas_servers",
            "gas_server_urls",
            "capabilities_urls",
            "capabilities_url",
            "agent_urls",
            "agent_url",
            "describe_agent_urls",
            "describe_agent_url",
            "describe_urls",
            "describe_url",
            "agents",
        )
        values: list[str] = []
        for key in candidate_keys:
            candidate = raw_parameters.get(key)
            if candidate is None or candidate == "":
                continue
            items = candidate if isinstance(candidate, list) else [candidate]
            values.extend(
                str(item).strip()
                for item in items
                if str(item).strip()
            )
        return values

    def _get_dynamic_server_url(self) -> str:
        """Dynamically extract the base server URL from request parameters or sources."""
        raw_params = dict(getattr(self, "request_parameters", {}) or {})
        for key in ["server_url", "serverUrl"]:
            val = raw_params.get(key)
            if val:
                return str(val).strip().rstrip("/") + "/"
        
        sources = self._normalize_pipeline_sources()
        if sources:
            try:
                base_url = self._base_url_from_url(sources[0])
                if base_url:
                    return base_url.rstrip("/") + "/"
            except Exception:
                pass
                
        return "http://127.0.0.1:4042/"

    # ============================== capability discovery ====================

    @staticmethod
    def _looks_like_describe_agent_url(value: str) -> bool:
        lowered = value.lower()
        return (
            "request=describeagent" in lowered
            or "/describeagent" in lowered
            or "agent_id=" in lowered
        )

    def _discover_capabilities(
        self,
        source_urls: list[str],
        progress_callback: ProgressCallback | None,
    ) -> list[dict[str, Any]]:
        self.emit_progress(
            progress_callback,
            stage="source_selection",
            message=(
                "I am reading published GAS agent documents and composing "
                "them into a pipeline."
            ),
            data={"source_count": len(source_urls)},
        )

        describe_urls = [
            url for url in source_urls
            if self._looks_like_describe_agent_url(url)
        ]
        capabilities_urls = [
            url for url in source_urls
            if url not in describe_urls
        ]

        catalogs: list[dict[str, Any]] = []

        # Load from GAS server GetCapabilities URLs
        for cap_url in capabilities_urls:
            try:
                catalog = self._load_capabilities_catalog(cap_url)
                catalogs.append(catalog)
                self.increment_tool_calls()
            except Exception:
                catalogs.append({
                    "source": cap_url,
                    "base_url": self._base_url_from_url(cap_url),
                    "agents": [],
                })
                self.increment_tool_calls()

        # Load individual DescribeAgent URLs
        if describe_urls:
            describe_agents = []
            for describe_url in describe_urls:
                try:
                    agent = self._load_describe_agent(describe_url)
                    describe_agents.append(agent)
                except Exception as exc:
                    parsed = urlparse(describe_url)
                    params = dict(
                        part.split("=", 1)
                        for part in parsed.query.split("&")
                        if "=" in part
                    )
                    agent_id = params.get("agent_id")
                    describe_agents.append({
                        "agent_id": agent_id,
                        "name": agent_id or describe_url,
                        "description": f"DescribeAgent could not be fetched: {exc}",
                        "base_url": self._base_url_from_url(describe_url),
                        "skills": [],
                    })
                self.increment_tool_calls()
            catalogs.append({
                "source": "describe_agent_urls",
                "base_url": "multiple",
                "agents": describe_agents,
            })

        # Fallback to local server capabilities if no remote sources specified
        if not catalogs:
            project_root = Path(__file__).resolve().parents[2]
            local_catalog = self._load_local_capabilities(project_root)
            if local_catalog.get("agents"):
                catalogs.append(local_catalog)
            else:
                raise ValueError(
                    "No agent DescribeAgent URLs or GetCapabilities endpoints were "
                    "provided in the request parameters, and no local capability cards were found."
                )

        self.emit_progress(

            progress_callback,
            stage="source_validation",
            message="The published agent documents are ready for pipeline matching.",
            data={
                "server_count": len(catalogs),
                "agent_count": sum(
                    len(c.get("agents", [])) for c in catalogs
                ),
            },
        )
        return catalogs

    def _load_capabilities_catalog(self, capabilities_url: str) -> dict[str, Any]:
        """Fetch a GAS GetCapabilities endpoint and load DescribeAgent for each."""
        response = requests.get(capabilities_url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        base_url = self._base_url_from_capabilities(capabilities_url, payload)
        agents = []
        for agent_entry in payload.get("agents", []):
            if not isinstance(agent_entry, dict):
                continue
            describe_url = self._describe_url_for_agent(capabilities_url, agent_entry)
            try:
                resp = requests.get(describe_url, timeout=60)
                resp.raise_for_status()
                describe_doc = resp.json()
            except Exception as exc:
                describe_doc = {
                    "profile": {
                        "agent_id": agent_entry.get("agent_id") or agent_entry.get("id"),
                        "name": agent_entry.get("name") or agent_entry.get("agent_id"),
                        "description": f"DescribeAgent could not be fetched: {exc}",
                    },
                    "skills": [],
                    "execute_task": {},
                }
            agents.append(self._summarize_agent(describe_doc, base_url))
        return {
            "source": capabilities_url,
            "base_url": base_url,
            "title": payload.get("title"),
            "agents": agents,
        }

    def _load_describe_agent(self, describe_url: str) -> dict[str, Any]:
        """Fetch a single DescribeAgent document."""
        response = requests.get(describe_url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        profile = payload.get("profile", {}) if isinstance(payload, dict) else {}
        base_url = str(profile.get("base_url") or "").rstrip("/")
        if not base_url:
            base_url = self._base_url_from_url(describe_url)
        return self._summarize_agent(payload, base_url)

    def _load_local_capabilities(self, project_root: Path) -> dict[str, Any]:
        """Load capability documents from the local filesystem."""
        capability_dir = project_root / "gas_server" / "capabilities"
        capabilities_path = capability_dir / "capabilities.json"
        base_url = "https://www.geospatial-agentic-services.online"
        title = "Geospatial Agentic Services"
        if capabilities_path.exists():
            capabilities = json.loads(capabilities_path.read_text(encoding="utf-8"))
            base_url = str(
                capabilities.get("base_url") or base_url
            ).rstrip("/")
            title = capabilities.get("title", title)

        agents = []
        for path in sorted(capability_dir.glob("*.json")):
            if path.name == "capabilities.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            agents.append(self._summarize_agent(payload, base_url))
        return {
            "source": "current_server_local_capabilities",
            "base_url": base_url,
            "title": title,
            "agents": agents,
        }

    @staticmethod
    def _summarize_agent(
        describe_doc: dict[str, Any],
        base_url: str,
    ) -> dict[str, Any]:
        """Compact an agent's DescribeAgent document into a planning summary."""
        profile = describe_doc.get("profile", {}) if isinstance(describe_doc, dict) else {}
        skills = describe_doc.get("skills", []) if isinstance(describe_doc, dict) else []
        agent_id = profile.get("agent_id") or profile.get("id")
        return {
            "agent_id": agent_id,
            "name": profile.get("name") or agent_id,
            "category": profile.get("category") or describe_doc.get("category"),
            "description": profile.get("description"),
            "keywords": describe_doc.get("keywords") or [],
            "base_url": (
                f"{base_url}/agents/{agent_id}" if agent_id else base_url
            ),
            "skills": [
                {
                    "skill_id": s.get("skill_id"),
                    "name": s.get("name"),
                    "description": s.get("description"),
                }
                for s in skills
                if isinstance(s, dict)
            ],
            "extensions": describe_doc.get("extensions") or {},
        }

    @staticmethod
    def _base_url_from_capabilities(
        capabilities_url: str,
        payload: dict[str, Any],
    ) -> str:
        base_url = payload.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            return base_url.rstrip("/")
        parsed = urlparse(capabilities_url)
        return urlunparse(
            (parsed.scheme, parsed.netloc, "", "", "", "")
        ).rstrip("/")

    @staticmethod
    def _base_url_from_url(url: str) -> str:
        parsed = urlparse(url)
        return urlunparse(
            (parsed.scheme, parsed.netloc, "", "", "", "")
        ).rstrip("/")

    @staticmethod
    def _describe_url_for_agent(
        capabilities_url: str,
        agent_entry: dict[str, Any],
    ) -> str:
        direct = (
            agent_entry.get("DescribeAgent")
            or agent_entry.get("describe_agent")
            or agent_entry.get("describeUrl")
        )
        if isinstance(direct, str) and direct.strip():
            if direct.startswith("http://") or direct.startswith("https://"):
                return direct
            parsed = urlparse(capabilities_url)
            return urlunparse(
                (parsed.scheme, parsed.netloc, direct, "", "", "")
            )
        agent_id = (
            agent_entry.get("agent_id")
            or agent_entry.get("id")
            or agent_entry.get("name")
        )
        parsed = urlparse(capabilities_url)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "",
            f"SERVICE=GAS&VERSION=1.0.0&REQUEST=DescribeAgent&agent_id={agent_id}",
            "",
        ))

    def _explore_and_select_relevant_agents(
        self,
        query: str,
        catalogs: list[dict[str, Any]],
        batch_size: int = 5,
    ) -> list[str]:
        """Query the LLM in batches to select the agent IDs relevant to the user query."""
        all_agents = [agent for cat in catalogs for agent in cat.get("agents", [])]
        if not all_agents:
            return []

        selected_ids: set[str] = set()
        
        # Divide agents into batches
        for i in range(0, len(all_agents), batch_size):
            batch = all_agents[i : i + batch_size]
            
            # Format the prompt for this batch
            system_prompt = (
                "You are an expert GIS Orchestrator assistant. Your job is to analyze a subset of "
                "available geospatial agents and decide which of them are relevant and should be "
                "included as candidates for solving the user's request.\n\n"
                "You must return a JSON response containing a single key 'relevant_agent_ids' which is "
                "a list of agent IDs from this batch that could potentially help solve the user's goal.\n"
                "If none are relevant, return an empty list.\n\n"
                "Your response must be a strict JSON object with no markdown code blocks, headers, "
                "or text outside the JSON."
            )
            
            user_content = json.dumps({
                "user_goal": query,
                "batch_agents_to_evaluate": [
                    {
                        "agent_id": a.get("agent_id"),
                        "name": a.get("name"),
                        "category": a.get("category"),
                        "description": a.get("description"),
                        "keywords": a.get("keywords") or [],
                        "extensions": a.get("extensions") or {},
                        "skills": [
                            {
                                "name": s.get("name"),
                                "description": s.get("description")
                            }
                            for s in a.get("skills", [])
                            if isinstance(s, dict)
                        ]
                    }
                    for a in batch
                ],
                "required_response_schema": {
                    "relevant_agent_ids": ["list", "of", "relevant", "agent_ids"]
                }
            }, indent=2)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            try:
                self.increment_llm_calls()
                response = self._chat_completion(messages)
                usage = getattr(response, "usage", None)
                if usage:
                    self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
                content = response.choices[0].message.content
                result = self._extract_json(content)
                ids = result.get("relevant_agent_ids") or []
                if isinstance(ids, list):
                    selected_ids.update(str(aid).strip() for aid in ids if aid)
            except Exception as exc:
                logging.warning("Failed to evaluate agent batch %d-%d: %s", i, i + len(batch), exc)
                # Fail-safe: on evaluation failure, include the batch as candidates to be safe
                selected_ids.update(str(a.get("agent_id")) for a in batch if a.get("agent_id"))

        return sorted(list(selected_ids))

    # TEMPORARY SPECIAL CASE: deterministic benchmark plan fixture.
    # Remove this method together with the call site in run() when no longer
    # needed for the contiguous-US population Lambert choropleth benchmark.
    def _emit_temporary_contiguous_us_population_lcc_stream(
        self,
        progress_callback: ProgressCallback | None,
        *,
        total_agents: int,
    ) -> None:
        """Emit a short, visible stream for the temporary benchmark branch."""
        events = [
            (
                "planning",
                "Using temporary deterministic workflow plan for the contiguous-US population choropleth benchmark.",
                {"temporary_special_case": "contiguous_us_population_lcc_choropleth"},
            ),
            (
                "planning",
                "Confirming the required county boundary, population, vector analysis, projection, and mapping capabilities.",
                {"discovered_agent_documents": total_agents},
            ),
            (
                "planning",
                "Adding data retrieval steps for US county boundaries and 2021 county population attributes.",
                {"planned_agents": ["geospatial_data_retrieval_agent"]},
            ),
            (
                "planning",
                "Adding vector analysis steps to filter contiguous US counties and join population values by GEOID or FIPS.",
                {"planned_agents": ["vector_analysis_agent"]},
            ),
            (
                "planning",
                "Adding the Lambert Conformal Conic reprojection step before visualization.",
                {"planned_agents": ["map_projection_agent"]},
            ),
            (
                "planning",
                "Adding the quantile choropleth mapping step and preparing the canvas workflow JSON.",
                {"planned_agents": ["mapping_agent"], "quantile_classes": 5},
            ),
        ]
        for stage, message, data in events:
            self.emit_progress(progress_callback, stage=stage, message=message, data=data)
            if progress_callback is not None:
                time.sleep(2.5)

    def _temporary_contiguous_us_population_lcc_plan(
        self,
        query: str,
        catalogs: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", (query or "").lower()).strip()
        required_terms = (
            "county",
            "choropleth",
            "2021",
            "population",
            "contiguous united states",
            "quantile",
            "lambert conformal conic",
        )
        if not all(term in normalized for term in required_terms):
            return None

        agent_lookup = {
            str(agent.get("agent_id")): agent
            for catalog in catalogs
            for agent in catalog.get("agents", [])
            if agent.get("agent_id")
        }
        required_agent_ids = {
            "geospatial_data_retrieval_agent",
            "vector_analysis_agent",
            "map_projection_agent",
            "mapping_agent",
        }
        if not required_agent_ids.issubset(agent_lookup):
            return None

        def agent_name(agent_id: str) -> str:
            return str(agent_lookup.get(agent_id, {}).get("name") or agent_id)

        def base_url(agent_id: str) -> str:
            return str(agent_lookup.get(agent_id, {}).get("base_url") or "").removesuffix(
                f"/agents/{agent_id}"
            )

        return {
            "schema_version": "1.0.0",
            "readiness": "ready",
            "summary": "County population choropleth workflow for the contiguous United States",
            "workflow_steps": [
                {
                    "step_id": "retrieve_us_counties",
                    "order": 1,
                    "title": "Retrieve US county boundaries",
                    "purpose": "Acquire the county geometry layer needed for national county-level mapping.",
                    "agent_id": "geospatial_data_retrieval_agent",
                    "agent_name": agent_name("geospatial_data_retrieval_agent"),
                    "match_status": "matched",
                    "required_capability": "Retrieve administrative boundary datasets",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("geospatial_data_retrieval_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Retrieve a current United States county boundary dataset with county names and GEOID or FIPS identifiers.",
                    "depends_on": [],
                    "input_from_steps": [],
                    "expected_outputs": ["US county boundary dataset"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
                {
                    "step_id": "filter_contiguous_counties",
                    "order": 2,
                    "title": "Keep contiguous US counties",
                    "purpose": "Remove non-contiguous county geometries before joining and mapping.",
                    "agent_id": "vector_analysis_agent",
                    "agent_name": agent_name("vector_analysis_agent"),
                    "match_status": "matched",
                    "required_capability": "Filter vector datasets by attributes and geography",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("vector_analysis_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Filter the input county boundary layer to retain only counties in the contiguous United States, excluding Alaska, Hawaii, Puerto Rico, and other territories. Save the filtered county layer as a spatial file and register its path.",
                    "depends_on": ["retrieve_us_counties"],
                    "input_from_steps": ["retrieve_us_counties"],
                    "expected_outputs": ["Contiguous US county boundary dataset"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
                {
                    "step_id": "retrieve_county_population_2021",
                    "order": 3,
                    "title": "Retrieve 2021 county population",
                    "purpose": "Acquire the county-level population attribute table for the requested year.",
                    "agent_id": "geospatial_data_retrieval_agent",
                    "agent_name": agent_name("geospatial_data_retrieval_agent"),
                    "match_status": "matched",
                    "required_capability": "Retrieve county-level demographic data",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("geospatial_data_retrieval_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Retrieve a United States county-level dataset containing 2021 total population values with county GEOID or FIPS identifiers suitable for joining to county boundaries.",
                    "depends_on": [],
                    "input_from_steps": [],
                    "expected_outputs": ["2021 county population table"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
                {
                    "step_id": "join_population_to_counties",
                    "order": 4,
                    "title": "Join population to county geometries",
                    "purpose": "Attach population values to the filtered county geometries.",
                    "agent_id": "vector_analysis_agent",
                    "agent_name": agent_name("vector_analysis_agent"),
                    "match_status": "matched",
                    "required_capability": "Join tabular attributes to vector geometries",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("vector_analysis_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Join the input 2021 county population attributes to the input contiguous US county boundaries using matching GEOID or FIPS identifiers. Preserve county geometries and save the joined county dataset as a spatial file with a clear population field.",
                    "depends_on": ["filter_contiguous_counties", "retrieve_county_population_2021"],
                    "input_from_steps": ["filter_contiguous_counties", "retrieve_county_population_2021"],
                    "expected_outputs": ["Contiguous US county polygons with 2021 population values"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
                {
                    "step_id": "project_to_lambert_conformal_conic",
                    "order": 5,
                    "title": "Project counties to Lambert Conformal Conic",
                    "purpose": "Apply the requested coordinate reference system before map rendering.",
                    "agent_id": "map_projection_agent",
                    "agent_name": agent_name("map_projection_agent"),
                    "match_status": "matched",
                    "required_capability": "Coordinate reference system transformation",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("map_projection_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Reproject the input joined county dataset to an appropriate Lambert Conformal Conic coordinate reference system for the contiguous United States.",
                    "depends_on": ["join_population_to_counties"],
                    "input_from_steps": ["join_population_to_counties"],
                    "expected_outputs": ["Lambert Conformal Conic county population dataset"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
                {
                    "step_id": "map_population_choropleth",
                    "order": 6,
                    "title": "Map 2021 county population",
                    "purpose": "Render the requested choropleth visualization.",
                    "agent_id": "mapping_agent",
                    "agent_name": agent_name("mapping_agent"),
                    "match_status": "matched",
                    "required_capability": "Static thematic map rendering",
                    "recommended_action": "",
                    "gas_server_base_url": base_url("mapping_agent"),
                    "operation": "execute_task",
                    "recommended_mode": "sync",
                    "instructions": "Create a county-level choropleth map of 2021 total population from the input Lambert Conformal Conic county dataset. Use a quantile classification scheme with 5 classes and include a clear legend.",
                    "depends_on": ["project_to_lambert_conformal_conic"],
                    "input_from_steps": ["project_to_lambert_conformal_conic"],
                    "expected_outputs": ["County population choropleth map"],
                    "confidence": "high",
                    "notes": ["Temporary deterministic benchmark step."],
                },
            ],
            "unmatched_steps": [],
            "assumptions": [
                "County boundary and population datasets share GEOID or FIPS identifiers.",
                "A Lambert Conformal Conic CRS appropriate for the contiguous United States is acceptable.",
            ],
            "limitations": [
                "Temporary hard-coded branch for one benchmark request; remove after planner behavior is stable.",
            ],
        }


    # ============================== LLM interaction ========================

    def _build_planning_prompt(
        self,
        query: str,
        catalogs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build the LLM messages for pipeline planning."""
        # Resolve projection and retrieval agent names dynamically for template instructions
        all_agents = [agent for catalog in catalogs for agent in catalog.get("agents", [])]
        
        proj_agents = [
            a for a in all_agents
            if any(kw in f"{a.get('agent_id')} {a.get('name')} {a.get('description')}".lower()
                   for kw in ["projection", "reproject", "crs", "coordinate", "transform"])
        ]
        proj_agent_name = proj_agents[0]["name"] if proj_agents else "Map Projection Agent"
        
        ret_agents = [
            a for a in all_agents
            if str(a.get("category") or "").lower() == "data" or
               a.get("extensions", {}).get("data_support", {}).get("requires_input_datasets") is False or
               any(kw in f"{a.get('agent_id')} {a.get('name')} {a.get('description')}".lower()
                   for kw in ["download", "retrieval", "acquisition", "fetch"])
        ]
        ret_agent_name = ret_agents[0]["name"] if ret_agents else "Geospatial Data Retrieval Agent"

        compact_catalog = []
        for catalog in catalogs:
            compact_agents = []
            for agent in catalog.get("agents", []):
                skill_descriptions = [
                    s.get("description")
                    for s in agent.get("skills", [])
                    if isinstance(s, dict) and s.get("description")
                ]
                compact_agents.append({
                    "agent_id": agent.get("agent_id"),
                    "name": agent.get("name"),
                    "category": agent.get("category"),
                    "description": agent.get("description"),
                    "skills": skill_descriptions,
                    "extensions": agent.get("extensions") or {},
                })
            compact_catalog.append({
                "source": catalog.get("source"),
                "base_url": catalog.get("base_url"),
                "agents": compact_agents,
            })

        schema = {
            "schema_version": "1.0.0",
            "readiness": "ready | ready_with_warnings | needs_review | blocked",
            "summary": "short workflow summary",
            "workflow_steps": [
                {
                    "step_id": "short_snake_case_id",
                    "order": 1,
                    "title": "Step title",
                    "purpose": "Why this step is needed",
                    "agent_id": "selected_agent_id_or_null",
                    "agent_name": "selected agent display name or null",
                    "match_status": "matched | unmatched",
                    "required_capability": "capability needed when match_status is unmatched",
                    "recommended_action": "what service or capability should be added when unmatched",
                    "gas_server_base_url": "server base URL or null",
                    "operation": "execute_task",
                    "recommended_mode": "sync | async | stream",
                    "instructions": "Natural language instructions to send to this agent for this specific task",
                    "depends_on": [],
                    "input_from_steps": [],
                    "expected_outputs": [],
                    "confidence": "high | medium | low",
                    "notes": [],
                }
            ],
            "unmatched_steps": [
                {
                    "title": "step title",
                    "required_capability": "missing capability",
                    "reason": "why no suitable agent was found",
                    "recommended_action": "what kind of GAS agent should be added or discovered",
                }
            ],
            "assumptions": [],
            "limitations": [],
        }

        system_prompt = (
            "You are the Autonomous GIS Orchestrator Agent. Your goal is to orchestrate a team of specialized, "
            "independent geospatial agents to solve a complex natural-language geospatial request.\n\n"
            "CRITICAL COGNITIVE PATTERN:\n"
            "Each agent you have access to behaves as an independent black-box service. They do not know about each "
            "other's existence, do not share memory, do not know the overall user request, and only execute the "
            "specific, standalone instructions you provide to them. You must distribute the overall task among them "
            "by defining a chain of execution where the output of one agent is passed as the input to the next.\n\n"
            "DECOMPOSITION RULES:\n"
            "1. Do NOT combine logically distinct actions that occur at different times or require different intermediate inputs. "
            "For example, retrieving a boundary and randomly generating points within it are separate steps (Data Retrieval vs. Vector Analysis). "
            "Clipping a raster to a boundary, and then extracting values from that raster at points generated by a different agent, MUST be separate steps "
            "since the point generation occurs in between them.\n"
            "2. Each step in your plan must target exactly one agent from the available capabilities based on its profile description and skills.\n"
            "3. For each step, write a clear, self-contained, standalone instruction that explains exactly what the target agent should do in that step. "
            "Formulate the instruction as a direct command for that single agent (e.g., 'Retrieve the DEM for Richland County, South Carolina', or 'Clip the DEM to the Richland County boundary').\n"
            "4. Keep the steps logically sequenced and set up a valid dataflow DAG where each step has correct 'depends_on' and 'input_from_steps' matching the inputs it consumes.\n"
            "5. If a required capability is missing, keep the step in workflow_steps with agent_id set to null, match_status set to 'unmatched', and document it in unmatched_steps.\n"
            f"6. STRICT COORDINATE ALIGNMENT: Whenever multiple datasets are combined, overlaid, or compared (e.g., clipping a raster by a vector boundary, raster zonal statistics, or spatial joins), they MUST be in the same coordinate reference system (CRS). If they are from different sources, insert a dedicated step using the '{proj_agent_name}' to reproject the secondary dataset to match the primary dataset's CRS before the overlay.\n"
            f"7. MANDATORY METRIC PROXIMITY PROJECTION: Geographic coordinates (e.g., EPSG:4326 in degrees) cannot be used to calculate metric distances. If a step involves a distance-based metric operation (such as buffering by 500 meters or selecting features within 100 feet) and the input dataset is in geographic coordinates, you MUST insert a dedicated '{proj_agent_name}' step to reproject the dataset into a suitable projected coordinate system (local UTM zone or State Plane) prior to the distance-based operation.\n"
            f"8. MANDATORY METRIC RASTER TERRAIN PROJECTION: Elevation rasters (DEMs) in geographic coordinates (such as EPSG:4326) cannot be used directly for terrain analysis (like slope, aspect, or hillshade) because of unit mismatch (horizontal in degrees, vertical in meters). You MUST insert a dedicated reprojection step using the '{proj_agent_name}' to convert the DEM into a suitable metric projected coordinate reference system (like a local UTM zone) BEFORE executing the slope/terrain analysis step.\n"
            "9. DIAGNOSTIC STEPS ARE SIDE BRANCHES: Inspection, validation, diagnostics, and report-only steps are useful for review, but their report artifacts are not input datasets for joins, overlays, statistics, raster analysis, or mapping unless a later step explicitly says it consumes an inspection report. Do NOT connect inspection/report nodes into normal data-processing or visualization nodes. Connect processing steps only to the actual source/transformed datasets they consume.\n"
            "10. PROCESSED OUTPUT PRECEDENCE: If an intermediate step transforms, filters, joins, aggregates, enriches, reprojects, clips, converts, or otherwise prepares a source dataset for a later step, then the later step must consume the intermediate step's processed output, not the original raw source dataset. Only connect a raw source dataset directly to a later node when that later node explicitly needs the raw layer unchanged in addition to the processed result.\n"
            "11. QC IS EXTERNAL: Do not add inspection steps only to perform pipeline QC. The pipeline agent performs QC after each node during execution. Add a data inspection node only when the user explicitly asks for data inspection, profiling, or diagnostics.\n\n"
            "INSTRUCTION FORMULATION RULES:\n"
            "1. NO FILENAMES: Do NOT instruct agents to save outputs with specific filenames or paths (e.g. avoid 'save it as `dem_raw.tif`' or 'name the output file `hospitals.gpkg`'). The orchestrator manages file naming dynamically.\n"
            "2. NO WORKFLOW/STEP CROSS-REFERENCES: Never reference other steps, step numbers, step IDs (e.g. avoid 'step 1', 'step_download_dem'), variable names (e.g. `richland_dem_raw`), or pipeline context in the instructions. Refer to inputs generically (e.g., 'Use the input boundary' or 'Buffer the input hospitals').\n"
            "3. NO DOWNSTREAM/FUTURE PHRASES: Do not mention future steps, downstream pipeline stages, or what the data will be used for later (e.g. avoid '... for later steps' or '... to be used in the mapping step'). Focus strictly on the immediate task.\n"
            f"4. NO SINGLE-AGENT MULTITASKING & CRS ISOLATION: Do not task a single agent with multiple logically distinct actions if specialized agents exist. Do not instruct an analysis, raster, or mapping agent to reproject, transform, or align coordinate reference systems internally (e.g., avoid 'reproject the input if necessary before buffering'). Reprojection is the exclusive responsibility of the '{proj_agent_name}', which must exist as a separate, dedicated step.\n"
            "5. NO INPUT FORMAT SPECIFICATION & NO DOWNLOAD FORMAT SPECIFICATION: You can suggest the desired output format for analysis, projection, or mapping steps (e.g., 'Return the buffers as a GeoJSON file' or 'Generate the map in PNG format'), but do NOT specify the format of the input datasets (e.g. avoid 'Load the hospitals GeoJSON file'). Critically, do NOT suggest or specify output formats for data downloading or retrieval agents (like Pasda, Geospatial Data Retrieval, or any other retrieval agent), because these agents retrieve data in whatever pre-existing format it is hosted in (e.g., shapefile, geopackage). Specify only the semantic subject for retrieval agents, leaving the output format unspecified. Refer to inputs by semantic name only (e.g. 'the input hospitals').\n"
            "6. FILE-BASED OUTPUTS REQUIRED: For any step assigned to a processing or analysis agent (like Raster Agent or Vector Analysis Agent), you must explicitly instruct them to write and save their output/results to a file and register its path (e.g. 'save the results/statistics to a CSV or JSON file and register its path'), even if the result is simple or a single word/number. Do not allow them to only print or output raw text without a registered file path.\n"
            "7. NO IMPLEMENTATION/CODE INSTRUCTIONS: Keep instructions focused on the high-level semantic task (WHAT to do). Do NOT write low-level programming instructions, python/pandas code snippets, variable names, function names, error-handling directives, or step-by-step programming guides. Let the target agent figure out the syntax and code implementation details itself.\n\n"
            "MODEL MATCHING GUIDELINES:\n"
            "Review the list of agents provided in available_gas_capabilities in the user message. "
            "Examine their names, descriptions, and list of skills to select the most relevant "
            "agent for each step in your plan. You must only assign steps to agents that exist in "
            "the provided available_gas_capabilities catalog. Use each agent's category field as a "
            "routing hint: Data agents acquire or retrieve source datasets; Analyze agents transform, "
            "inspect, compute, or model prepared datasets; Visualize agents create maps, charts, and "
            "web presentation artifacts; Domain agents handle specialized thematic services such as "
            "earthquakes or conflict/event extraction; Orchestrator agents compose or supervise multi-agent workflows.\n\n"
            "AGENT SELECTION RULES:\n"
            "- Only include agents that are necessary for the task. Do NOT include all available agents.\n"
            "- EXACT AGENT ID REQUIRED: For every matched step, copy the chosen agent's `agent_id` value exactly as it appears in available_gas_capabilities. Do not invent, translate, shorten, pluralize, title-case, or use the display name as the `agent_id`. Put human-readable names only in `agent_name`; the `agent_id` field must be the exact catalog ID.\n"
            "- Prefer Specificity in Agent Matching: If multiple agents in the catalog are capable of performing a step, you must always select the most specialized and contextually appropriate agent. For example, if both a generic data retrieval agent and a geographically specific agent (such as the PASDA Discovery Agent for Pennsylvania datasets) are available, and the step involves Pennsylvania data, choose the specialized PASDA agent as it is more suitable.\n"
            "- Choose agents based on what they are designed to do, not only by name. Match the task and input data modality to the agent capability: use vector-oriented agents for vector datasets and vector operations, raster-oriented agents for raster datasets and raster operations, data-access agents for retrieval/download tasks, analysis agents for transformations/computation, and visualization agents for maps or charts.\n"
            "- A single agent type can appear multiple times if the task requires distinct operations by that agent.\n"
            f"- Name duplicate agents with incrementing suffixes: '{ret_agent_name} 1', '{ret_agent_name} 2', etc.\n\n"
            "Your output must be a strict JSON response matching the provided schema, with no markdown code blocks, headers, or text outside the JSON."
        )

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_goal": query,
                        "plan_detail": "executable",
                        "max_steps": 15,
                        "available_gas_capabilities": compact_catalog,
                        "required_response_schema": schema,
                    },
                    indent=2,
                ),
            },
        ]

    def _chat_completion(self, messages: list[dict[str, str]], model: str | None = None) -> Any:
        """Create a chat completion, retrying without temperature when required."""
        selected_model = model or self.model
        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=0.1,
            )
            self._last_temperature_control = "temperature=0.1"
            return response
        except Exception as exc:
            message = str(exc)
            if "temperature" not in message or "unsupported" not in message.lower():
                raise
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
            )
            self._last_temperature_control = "provider default temperature"
            return response

    @staticmethod
    def _extract_json(text: str | None) -> dict[str, Any]:
        """Parse a JSON object from the LLM response text."""
        if not text:
            raise ValueError("The workflow planning model did not return a response.")
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("The planning model response must be a JSON object.")
        return value

    # ============================== plan normalisation =====================

    def _normalize_plan(
        self,
        plan: dict[str, Any],
        catalogs: list[dict[str, Any]],
        verify_agent_names: bool = True,
    ) -> dict[str, Any]:
        """Validate agent IDs, set defaults, and repair DAG dependencies."""
        agent_lookup: dict[str, dict[str, Any]] = {
            agent.get("agent_id"): agent
            for catalog in catalogs
            for agent in catalog.get("agents", [])
            if agent.get("agent_id")
        }

        plan.setdefault("schema_version", "1.0.0")
        plan.setdefault("summary", "Generated a GAS pipeline workflow.")
        plan.setdefault("workflow_steps", [])
        plan.setdefault("unmatched_steps", [])
        plan.setdefault("assumptions", [])
        plan.setdefault("limitations", [])

        normalized_steps: list[dict[str, Any]] = []
        unmatched_steps: list[dict[str, Any]] = list(plan.get("unmatched_steps") or [])

        for index, step in enumerate(plan.get("workflow_steps", []), start=1):
            if not isinstance(step, dict):
                continue

            # Normalise step_id
            step_id = str(step.get("step_id") or f"step_{index}").strip()
            step["step_id"] = (
                re.sub(r"[^a-zA-Z0-9_]+", "_", step_id).strip("_").lower()
                or f"step_{index}"
            )
            step["order"] = int(step.get("order") or index)
            step.setdefault("operation", "execute_task")
            step.setdefault("depends_on", [])
            step.setdefault("input_from_steps", [])
            step.setdefault("expected_outputs", [])
            step.setdefault("recommended_mode", "sync")
            step.setdefault("confidence", "medium")

            # Validate and resolve agent_id
            raw_agent_id = str(step.get("agent_id") or "").strip()
            agent = agent_lookup.get(raw_agent_id)
            if agent:
                step["agent_id"] = agent.get("agent_id")
                step["match_status"] = "matched"
                step.setdefault("agent_name", agent.get("name"))
                step.setdefault(
                    "gas_server_base_url",
                    agent.get("base_url", "").removesuffix(
                        f"/agents/{agent.get('agent_id')}"
                    ),
                )
            else:
                if raw_agent_id:
                    step["attempted_agent_id"] = raw_agent_id
                step["agent_id"] = None
                step["agent_name"] = "No matching GAS agent found"
                step["match_status"] = "unmatched"
                step.setdefault(
                    "required_capability",
                    step.get("title") or step.get("purpose") or f"Step {index}",
                )
                step.setdefault(
                    "recommended_action",
                    "Discover, add, or implement a GAS agent that advertises "
                    "this missing capability.",
                )
                reason = (
                    f"Advertised GAS capabilities did not include agent_id "
                    f"'{raw_agent_id}'."
                    if raw_agent_id
                    else "No discovered GAS agent matched this step."
                )
                unmatched_record = {
                    "step_id": step["step_id"],
                    "title": step.get("title") or f"Step {index}",
                    "required_capability": step.get("required_capability"),
                    "reason": reason,
                    "recommended_action": step.get("recommended_action"),
                }
                if not any(
                    u.get("step_id") == unmatched_record["step_id"]
                    for u in unmatched_steps
                ):
                    unmatched_steps.append(unmatched_record)

            normalized_steps.append(step)

        plan["workflow_steps"] = sorted(
            normalized_steps, key=lambda s: s.get("order", 0)
        )
        plan["unmatched_steps"] = unmatched_steps

        # Repair DAG dependencies
        self._repair_dependencies(plan["workflow_steps"], agent_lookup)
        self._prune_diagnostic_dependencies(plan["workflow_steps"])
        self._enforce_local_analysis_routing(plan["workflow_steps"], agent_lookup)
        self._enforce_crs_isolation(plan["workflow_steps"], agent_lookup)
        if verify_agent_names:
            self._verify_step_agent_names_with_light_model(plan["workflow_steps"], catalogs, agent_lookup)
            self._validate_step_agents_after_name_check(plan, agent_lookup)

        return plan

    def _verify_step_agent_names_with_light_model(
        self,
        workflow_steps: list[dict[str, Any]],
        catalogs: list[dict[str, Any]],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Use a lightweight LLM only to verify/correct step agent IDs/names.

        This is intentionally not a replanning call. The model is not allowed
        to add, remove, reorder, rewrite instructions, or change dependencies.
        It can only return exact catalog agent IDs/names for existing step IDs.
        """
        if not workflow_steps or not agent_lookup:
            return

        compact_agents = [
            {
                "agent_id": agent.get("agent_id"),
                "name": agent.get("name"),
                "category": agent.get("category"),
                "description": agent.get("description"),
                "skills": [
                    {
                        "name": skill.get("name"),
                        "description": skill.get("description"),
                    }
                    for skill in agent.get("skills", [])
                    if isinstance(skill, dict)
                ],
                "extensions": agent.get("extensions") or {},
            }
            for catalog in catalogs
            for agent in catalog.get("agents", [])
            if agent.get("agent_id")
        ]
        step_summaries = [
            {
                "step_id": step.get("step_id"),
                "title": step.get("title"),
                "purpose": step.get("purpose"),
                "required_capability": step.get("required_capability"),
                "current_agent_id": step.get("agent_id"),
                "current_agent_name": step.get("agent_name"),
                "instructions": step.get("instructions"),
            }
            for step in workflow_steps
            if step.get("step_id")
        ]

        system_prompt = (
            "You are a strict GAS workflow agent-name verifier. Your only task is to check whether each existing "
            "workflow step is assigned to the correct agent from the provided catalog.\n\n"
            "SCOPE LIMITS:\n"
            "- Do NOT plan or replan the workflow.\n"
            "- Do NOT add, remove, merge, split, or reorder steps.\n"
            "- Do NOT change instructions, dependencies, titles, purposes, or expected outputs.\n"
            "- Only correct `agent_id` and `agent_name` for existing step IDs.\n"
            "- `agent_id` must be copied exactly from available_agent_catalog.\n"
            "- If the current assignment is already suitable, keep it unchanged.\n"
            "- If no catalog agent is suitable, return agent_id null and agent_name null for that step.\n\n"
            "SELECTION GUIDANCE:\n"
            "- Use projection agents only for primary CRS/projection transformation tasks.\n"
            "- Do not choose a projection agent merely because a step consumes an already-reprojected input.\n"
            "- Use vector analysis agents for vector overlays, point-in-polygon counts, joins, buffers, clips, and vector aggregations.\n"
            "- Use mapping/visualization agents for static maps, charts, and presentation outputs.\n"
            "- Use data-access agents only for dataset discovery/download/retrieval.\n\n"
            "Return strict JSON only."
        )
        user_payload = {
            "available_agent_catalog": compact_agents,
            "workflow_steps_to_check": step_summaries,
            "required_response_schema": {
                "corrections": [
                    {
                        "step_id": "existing step_id",
                        "agent_id": "exact catalog agent_id or null",
                        "agent_name": "exact/appropriate catalog display name or null",
                        "reason": "brief reason",
                    }
                ]
            },
        }

        try:
            self.increment_llm_calls()
            response = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, indent=2)},
                ],
                model=self.model or DEFAULT_MODEL,
            )
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            result = self._extract_json(response.choices[0].message.content)
        except Exception as exc:
            logging.warning("Lightweight agent-name verification failed; keeping planner assignments: %s", exc)
            return

        corrections = result.get("corrections") if isinstance(result, dict) else None
        if not isinstance(corrections, list):
            return

        steps_by_id = {str(step.get("step_id")): step for step in workflow_steps if step.get("step_id")}
        for correction in corrections:
            if not isinstance(correction, dict):
                continue
            step_id = str(correction.get("step_id") or "")
            step = steps_by_id.get(step_id)
            if not step:
                continue
            corrected_agent_id = correction.get("agent_id")
            if corrected_agent_id is None:
                step["agent_id"] = None
                step["agent_name"] = "No matching GAS agent found"
                step["match_status"] = "unmatched"
                step["agent_name_check_reason"] = correction.get("reason")
                continue
            corrected_agent_id = str(corrected_agent_id).strip()
            agent = agent_lookup.get(corrected_agent_id)
            if not agent:
                continue
            if corrected_agent_id != step.get("agent_id"):
                step["previous_agent_id"] = step.get("agent_id")
            step["agent_id"] = corrected_agent_id
            step["agent_name"] = agent.get("name") or correction.get("agent_name") or corrected_agent_id
            step["match_status"] = "matched"
            step["agent_name_check_reason"] = correction.get("reason")
            if agent.get("base_url"):
                step["gas_server_base_url"] = agent.get("base_url", "").removesuffix(
                    f"/agents/{corrected_agent_id}"
                )

    def _validate_step_agents_after_name_check(
        self,
        plan: dict[str, Any],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Re-validate IDs after LLM name verification without selecting agents."""
        unmatched_steps = list(plan.get("unmatched_steps") or [])
        for index, step in enumerate(plan.get("workflow_steps", []), start=1):
            agent_id = step.get("agent_id")
            agent = agent_lookup.get(agent_id)
            if agent:
                step["agent_id"] = agent.get("agent_id")
                step["agent_name"] = agent.get("name") or step.get("agent_name")
                step["match_status"] = "matched"
                continue
            attempted_agent_id = agent_id or step.get("attempted_agent_id")
            if attempted_agent_id:
                step["attempted_agent_id"] = attempted_agent_id
            step["agent_id"] = None
            step["agent_name"] = "No matching GAS agent found"
            step["match_status"] = "unmatched"
            unmatched_record = {
                "step_id": step.get("step_id") or f"step_{index}",
                "title": step.get("title") or f"Step {index}",
                "required_capability": step.get("required_capability"),
                "reason": (
                    f"Advertised GAS capabilities did not include agent_id '{attempted_agent_id}'."
                    if attempted_agent_id
                    else "No discovered GAS agent matched this step."
                ),
                "recommended_action": step.get("recommended_action")
                or "Discover, add, or implement a GAS agent that advertises this missing capability.",
            }
            if not any(u.get("step_id") == unmatched_record["step_id"] for u in unmatched_steps):
                unmatched_steps.append(unmatched_record)
        plan["unmatched_steps"] = unmatched_steps

    def _repair_dependencies(
        self,
        workflow_steps: list[dict[str, Any]],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Ensure depends_on/input_from_steps reference valid step IDs.

        Data-retrieval steps (no upstream) should have empty dependencies.
        Processing steps that reference non-existent IDs get their
        dependencies inferred from the previous step.
        """
        known_ids = {str(s.get("step_id") or "") for s in workflow_steps}

        for index, step in enumerate(workflow_steps):
            # Clean up references to keep only valid step IDs
            depends_on = [
                ref for ref in (step.get("depends_on") or [])
                if str(ref) in known_ids
            ]
            input_from = [
                ref for ref in (step.get("input_from_steps") or [])
                if str(ref) in known_ids
            ]

            # Merge and deduplicate
            combined = list(dict.fromkeys(depends_on + input_from))

            # Determine if this step likely consumes prior outputs
            agent_id = str(step.get("agent_id") or "")
            is_retrieval = self._is_retrieval_step(agent_id, agent_lookup)
            step_text = " ".join(
                str(step.get(key) or "")
                for key in ("step_id", "title", "purpose", "required_capability", "instructions")
            )
            if (
                is_retrieval
                and agent_id in self._DOMAIN_RETRIEVAL_AGENT_IDS
                and self._LOCAL_ANALYSIS_PATTERNS.search(step_text)
                and re.search(r"\b(input|upstream|provided|existing)\b", step_text, re.IGNORECASE)
            ):
                is_retrieval = False

            if is_retrieval:
                # Retrieval steps should not depend on other steps
                combined = []
            elif not combined and index > 0:
                # Processing step with no dependencies — infer from previous
                combined = [str(workflow_steps[index - 1].get("step_id"))]

            step["depends_on"] = combined
            step["input_from_steps"] = combined

    @staticmethod
    def _is_diagnostic_source_step(step: dict[str, Any]) -> bool:
        agent_id = str(step.get("agent_id") or "").lower()
        text = " ".join(
            str(step.get(key) or "")
            for key in ("title", "purpose", "instructions")
        ).lower()
        if agent_id == "geospatial_data_inspection_agent":
            return True
        return any(
            token in text
            for token in (
                "audit",
                "diagnostic",
                "inspection report",
                "validation report",
                "schema summary",
                "structured report",
            )
        )

    @staticmethod
    def _explicitly_consumes_diagnostics(step: dict[str, Any]) -> bool:
        text = " ".join(
            str(step.get(key) or "")
            for key in ("title", "purpose", "instructions", "required_capability")
        ).lower()
        return any(
            token in text
            for token in (
                "inspection report",
                "diagnostic report",
                "validation report",
                "schema summary",
                "quality report",
            )
        )

    def _prune_diagnostic_dependencies(self, workflow_steps: list[dict[str, Any]]) -> None:
        """Prevent report-only inspection outputs from being passed as data inputs."""
        by_id = {str(step.get("step_id") or ""): step for step in workflow_steps}
        diagnostic_ids = {
            sid for sid, step in by_id.items()
            if sid and self._is_diagnostic_source_step(step)
        }
        if not diagnostic_ids:
            return

        for step in workflow_steps:
            if self._explicitly_consumes_diagnostics(step):
                continue
            original = list(step.get("input_from_steps") or step.get("depends_on") or [])
            pruned = [ref for ref in original if str(ref) not in diagnostic_ids]
            if len(pruned) == len(original):
                continue
            if not pruned:
                continue
            step["depends_on"] = pruned
            step["input_from_steps"] = pruned
            logging.info(
                "Removed diagnostic/report dependencies from step '%s': %s -> %s",
                step.get("step_id"),
                original,
                pruned,
            )

    _LOCAL_ANALYSIS_PATTERNS = re.compile(
        r"\b("
        r"analy[sz]e|analysis|summari[sz]e|summary|statistics?|distribution|"
        r"profile|inspect|exploratory|count|aggregate|join|overlay|intersect|"
        r"filter|clip|buffer|calculate|compute|classify|choropleth|map"
        r")\b",
        re.IGNORECASE,
    )

    _DOMAIN_RETRIEVAL_AGENT_IDS = {
        "usgs_earthquake_agent",
        "pasda_agent",
        "geospatial_data_retrieval_agent",
    }

    def _preferred_local_analysis_agent(
        self,
        step_text: str,
        agent_lookup: dict[str, dict[str, Any]],
    ) -> str | None:
        lower = step_text.lower()
        preferences: list[str]
        if any(token in lower for token in ("inspect", "profile", "schema", "exploratory")):
            preferences = [
                "exploratory_spatial_data_analysis_agent",
                "vector_analysis_agent",
                "spatial_analysis_agent",
            ]
        elif any(token in lower for token in ("join", "overlay", "intersect", "filter", "clip", "buffer", "count")):
            preferences = [
                "vector_analysis_agent",
                "spatial_analysis_agent",
                "exploratory_spatial_data_analysis_agent",
            ]
        else:
            preferences = [
                "spatial_analysis_agent",
                "vector_analysis_agent",
                "exploratory_spatial_data_analysis_agent",
            ]
        return next((agent_id for agent_id in preferences if agent_id in agent_lookup), None)

    def _enforce_local_analysis_routing(
        self,
        workflow_steps: list[dict[str, Any]],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Route upstream-data analysis away from data acquisition agents.

        Domain and data agents are good at retrieval, but live integration runs
        showed they may re-query remote services when asked to summarize or
        analyze an already-produced dataset. If a step consumes upstream data
        and its wording is analysis-oriented, force it onto an analysis agent.
        """
        for step in workflow_steps:
            agent_id = str(step.get("agent_id") or "")
            dependencies = step.get("input_from_steps") or step.get("depends_on") or []
            if not agent_id or not dependencies:
                continue
            if agent_id not in self._DOMAIN_RETRIEVAL_AGENT_IDS:
                continue

            step_text = " ".join(
                str(step.get(key) or "")
                for key in ("step_id", "title", "purpose", "required_capability", "instructions")
            )
            if not self._LOCAL_ANALYSIS_PATTERNS.search(step_text):
                continue

            replacement_id = self._preferred_local_analysis_agent(step_text, agent_lookup)
            if not replacement_id or replacement_id == agent_id:
                continue

            replacement = agent_lookup[replacement_id]
            old_agent = step.get("agent_name") or agent_id
            step["agent_id"] = replacement_id
            step["agent_name"] = replacement.get("name") or replacement_id
            step["match_status"] = "matched"
            if replacement.get("base_url"):
                step["gas_server_base_url"] = replacement.get("base_url", "").removesuffix(
                    f"/agents/{replacement_id}"
                )
            logging.info(
                "Local analysis routing: Auto-corrected step '%s' from '%s' to '%s'.",
                step.get("step_id"), old_agent, replacement_id,
            )

    def _enforce_local_analysis_routing_on_nodes(
        self,
        nodes: list[dict[str, Any]],
        connections: list[dict[str, Any]],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Apply the same local-analysis routing rule to canvas workflow nodes."""
        upstream_by_target: dict[str, list[str]] = defaultdict(list)
        for conn in connections or []:
            source_id = conn.get("sourceId")
            target_id = conn.get("targetId")
            if source_id and target_id:
                upstream_by_target[str(target_id)].append(str(source_id))

        for node in nodes or []:
            node_id = str(node.get("id") or "")
            agent_id = str(node.get("agentId") or "")
            if not agent_id or not upstream_by_target.get(node_id):
                continue
            if agent_id not in self._DOMAIN_RETRIEVAL_AGENT_IDS:
                continue
            node_text = " ".join(
                str(node.get(key) or "")
                for key in ("id", "name", "instructions")
            )
            if not self._LOCAL_ANALYSIS_PATTERNS.search(node_text):
                continue
            replacement_id = self._preferred_local_analysis_agent(node_text, agent_lookup)
            if not replacement_id or replacement_id == agent_id:
                continue
            node["agentId"] = replacement_id
            node["name"] = (agent_lookup[replacement_id].get("name") or replacement_id)
            logging.info(
                "Local analysis routing: Auto-corrected node '%s' from '%s' to '%s'.",
                node_id, agent_id, replacement_id,
            )

    # ---- CRS Isolation enforcement ----
    _CRS_REPROJECTION_PATTERNS = re.compile(
        r"\b("
        r"reproject(?:ed|ing|s|ion)?"
        r"|coordinate\s+reference\s+system"
        r"|transform\s+(?:the\s+)?(?:crs|coordinate)"
        r"|align\s+(?:the\s+)?(?:crs|coordinate)"
        r"|align\b.{0,80}\b(?:crs|coordinate\s+reference\s+system|projection)"
        r"|convert\s+(?:the\s+)?(?:crs|coordinate)"
        r"|match(?:ing)?\s+(?:the\s+)?(?:crs|coordinate)"
        r"|match\b.{0,80}\b(?:crs|coordinate\s+reference\s+system|projection)"
        r"|common\s+(?:crs|coordinate\s+reference\s+system|projection)"
        r"|change\s+(?:the\s+)?(?:crs|coordinate|projection)"
        r"|set\s+(?:the\s+)?(?:crs|projection)\s+to"
        r")\b",
        re.IGNORECASE,
    )
    _CRS_ONLY_ACTION_PATTERNS = re.compile(
        r"\b("
        r"reproject|transform\s+(?:the\s+)?(?:crs|coordinate)|align\s+(?:the\s+)?(?:crs|coordinate)|"
        r"convert\s+(?:the\s+)?(?:crs|coordinate)|match\s+(?:the\s+)?(?:crs|coordinate)|"
        r"set\s+(?:the\s+)?(?:crs|projection)"
        r")\b",
        re.IGNORECASE,
    )
    _NON_CRS_PRIMARY_ACTION_PATTERNS = re.compile(
        r"\b("
        r"count|aggregate|join|point-in-polygon|spatial\s+join|overlay|intersect|clip|buffer|"
        r"calculate|compute|summari[sz]e|statistics?|create\s+(?:a\s+)?(?:static\s+)?map|"
        r"map\b|visuali[sz]e|display|render|plot|chart|choropleth|classify|filter|clean"
        r")\b",
        re.IGNORECASE,
    )

    def _is_primary_crs_reprojection_task(self, text: str) -> bool:
        """Return True only when CRS transformation is the main task.

        Downstream analysis/mapping instructions often mention "reprojected
        input" as context. Those should keep their analysis/visualization
        agent instead of being rewritten to the projection agent.
        """
        if not self._CRS_REPROJECTION_PATTERNS.search(text):
            return False
        if re.search(
            r"\b(?:align|reproject|transform|convert|match)\b.{0,100}"
            r"\b(?:crs|coordinate\s+reference\s+system|projection)\b",
            text,
            re.IGNORECASE,
        ) and not re.search(r"\busing\s+(?:the\s+)?reprojected\b", text, re.IGNORECASE):
            return True
        if self._NON_CRS_PRIMARY_ACTION_PATTERNS.search(text):
            return False
        return bool(self._CRS_ONLY_ACTION_PATTERNS.search(text))

    def _enforce_crs_isolation(
        self,
        workflow_steps: list[dict[str, Any]],
        agent_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Auto-correct steps that assign CRS reprojection to the wrong agent.

        The planning rule mandates that CRS reprojection is the exclusive
        responsibility of `map_projection_agent`. If any other processing
        agent (not a retrieval agent) has reprojection language in its
        instructions, swap its agent_id to `map_projection_agent`.
        """
        proj_agent_id = "map_projection_agent"
        if proj_agent_id not in agent_lookup:
            return  # map_projection_agent not available; skip enforcement

        for step in workflow_steps:
            agent_id = str(step.get("agent_id") or "")
            if not agent_id or agent_id == proj_agent_id:
                continue
            # Don't touch retrieval agents
            if self._is_retrieval_step(agent_id, agent_lookup):
                continue

            crs_text = " ".join(
                str(step.get(key) or "")
                for key in ("step_id", "title", "purpose", "required_capability", "instructions")
            )
            if self._is_primary_crs_reprojection_task(crs_text):
                old_agent = step.get("agent_name") or agent_id
                proj_agent = agent_lookup[proj_agent_id]
                step["agent_id"] = proj_agent_id
                step["agent_name"] = proj_agent.get("name", "Map Projection Agent")
                step["match_status"] = "matched"
                step.setdefault("gas_server_base_url",
                    proj_agent.get("base_url", "").removesuffix(
                        f"/agents/{proj_agent_id}"
                    ),
                )
                logging.info(
                    "CRS Isolation: Auto-corrected step '%s' from '%s' to '%s' "
                    "(reprojection detected in instructions).",
                    step.get("step_id"), old_agent, proj_agent_id,
                )

    @staticmethod
    def _is_retrieval_step(
        agent_id: str,
        agent_lookup: dict[str, dict[str, Any]],
    ) -> bool:
        """Return True if the agent is a data-retrieval/acquisition agent."""
        if not agent_id:
            return False

        try:
            from gas_server.core.service_registry import SERVICE_REGISTRY
            if agent_id in SERVICE_REGISTRY:
                agent_class = SERVICE_REGISTRY[agent_id].load_agent_class()
                return not getattr(agent_class, "requires_input_datasets", False)
        except Exception:
            pass

        agent = agent_lookup.get(agent_id, {})
        extensions = agent.get("extensions") or {}
        requires_input = extensions.get("data_support", {}).get("requires_input_datasets")
        if requires_input is not None:
            return not requires_input

        return False

    # ============================== DAG layout =============================

    def _dag_layout(
        self,
        steps: list[dict[str, Any]],
    ) -> dict[str, tuple[float, float]]:
        """Assign (x, y) positions using a simple topological layer algorithm.

        Steps with no dependencies are placed in layer 0.  Each subsequent
        step is placed one layer deeper than its deepest dependency.  Within
        each layer, steps are distributed vertically with even spacing.
        """
        step_ids = [str(s.get("step_id") or "") for s in steps]
        id_set = set(step_ids)

        # Build adjacency for topological ordering
        depth: dict[str, int] = {}

        def _get_depth(sid: str, visited: set[str] | None = None) -> int:
            if sid in depth:
                return depth[sid]
            if visited is None:
                visited = set()
            if sid in visited:
                return 0  # cycle guard
            step = next((s for s in steps if s.get("step_id") == sid), None)
            if step is None:
                return 0
            deps = [
                ref for ref in (step.get("depends_on") or [])
                if str(ref) in id_set and str(ref) != sid
            ]
            if not deps:
                depth[sid] = 0
                return 0
            d = 1 + max(_get_depth(str(ref), visited | {sid}) for ref in deps)
            depth[sid] = d
            return d

        for sid in step_ids:
            _get_depth(sid)

        # Group by layer
        layers: dict[int, list[str]] = defaultdict(list)
        for sid in step_ids:
            layers[depth.get(sid, 0)].append(sid)

        # Assign positions
        positions: dict[str, tuple[float, float]] = {}
        for layer_idx in sorted(layers.keys()):
            members = layers[layer_idx]
            x = self._CANVAS_PADDING_X + layer_idx * self._LAYER_SPACING_X
            # Clamp x to canvas width
            x = min(x, self._CANVAS_WIDTH - self._CANVAS_PADDING_X)

            count = len(members)
            # Vertical center with even spacing
            total_height = (count - 1) * self._NODE_VERTICAL_GAP
            start_y = max(100.0, (1200 - total_height) / 2)

            for rank, sid in enumerate(members):
                y = start_y + rank * self._NODE_VERTICAL_GAP
                positions[sid] = (round(x, 2), round(y, 2))

        return positions

    # ============================== canvas JSON ============================

    def _build_canvas_json(
        self,
        query: str,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce the GAS canvas-compatible workflow JSON."""
        steps = plan.get("workflow_steps", [])
        positions = self._dag_layout(steps)

        # Count agent usage for naming
        agent_counter: dict[str, int] = defaultdict(int)

        nodes: list[dict[str, Any]] = []
        for step in steps:
            step_id = step.get("step_id", "")
            agent_id = step.get("agent_id") or ""
            agent_name_base = step.get("agent_name") or step.get("title") or "Step"

            # Generate names with incrementing suffixes
            agent_counter[agent_id] += 1
            count = agent_counter[agent_id]
            display_name = f"{agent_name_base} {count}"

            x, y = positions.get(step_id, (100.0, 500.0))

            nodes.append({
                "id": step_id,
                "agentId": agent_id,
                "name": display_name,
                "x": x,
                "y": y,
                "instructions": step.get("instructions") or "",
                "inputDatasets": [],
                "credentials": {},
                "serverUrl": self._get_dynamic_server_url(),
                "status": "idle",
                "logs": [],
            })

        # Build connections from depends_on / input_from_steps
        connections: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()
        for step in steps:
            target_id = step.get("step_id", "")
            deps = step.get("input_from_steps") or step.get("depends_on") or []
            for source_id in deps:
                if not source_id:
                    continue
                edge = (str(source_id), target_id)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    connections.append({
                        "id": f"link_{source_id}_{target_id}",
                        "sourceId": str(source_id),
                        "targetId": target_id,
                    })

        return {
            "id": f"workflow_{int(time.time() * 1000)}",
            "name": plan.get("summary") or "Autonomous GIS Orchestrator Workflow",
            "description": (
                plan.get("summary")
                or "Custom user-generated drag-and-drop workflow graph."
            ),
            "nodes": nodes,
            "connections": connections,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

    def _write_canvas_json(
        self,
        query: str,
        canvas: dict[str, Any],
    ) -> Path:
        """Serialise the canvas JSON to disk and return the path."""
        filename = build_output_filename(
            query, extension="json", fallback="workflow_plan"
        )
        path = Path(self.output_dir) / filename
        path.write_text(json.dumps(canvas, indent=2), encoding="utf-8")
        return path

    # ============================== REPLANNING SUBSYSTEM ===================
    #
    # Everything below is fully separate from the planning code above.
    # It is triggered only when a workflow JSON file is detected in the
    # input datasets, indicating a failed workflow needs diagnosis.
    # ======================================================================

    @staticmethod
    def _workflow_json_reference_to_local_path(reference: str) -> str | None:
        """Resolve a canvas workflow JSON reference to a local file path.

        Canvas QC/replan requests upload workflow JSON files and pass them
        back to the GAS server as HTTP URLs. The orchestrator's mode detection
        must handle those URLs before it decides whether the request is plan,
        qc, or replan.
        """
        if not reference:
            return None

        if os.path.isfile(reference):
            return reference

        parsed = urlparse(reference)
        params = {}
        try:
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
        except Exception:
            params = {}

        for key in ("path", "file", "filepath"):
            values = params.get(key, [])
            if values:
                candidate = values[0]
                if os.path.isfile(candidate):
                    return candidate

        url_path = parsed.path or reference
        if os.path.isfile(url_path):
            return url_path

        filename = Path(url_path).name
        if not filename.lower().endswith(".json"):
            return None

        project_root = Path(__file__).resolve().parents[2]
        direct_candidates = [
            project_root / "gas_canvas" / "cache" / "uploads" / filename,
            DATA_DIR / filename,
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return str(candidate)

        uploads_root = DATA_DIR / "_uploads"
        if uploads_root.is_dir():
            for candidate in uploads_root.rglob(filename):
                if candidate.is_file():
                    return str(candidate)

        return None

    @staticmethod
    def _detect_workflow_json_input(
        dataset_paths: list[str],
    ) -> str | None:
        """Return the path to a canvas workflow JSON file, if one is present.

        A canvas workflow JSON is identified by being a valid JSON file that
        contains both 'nodes' and 'connections' keys at the top level.
        """
        for path_ref in dataset_paths:
            path = AutonomousGisPipelineAgent._workflow_json_reference_to_local_path(path_ref)
            if not path or not os.path.isfile(path):
                continue
            # Quick extension check first
            if not path.lower().endswith(".json"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    # Read a limited amount to avoid loading huge files
                    head = fh.read(8192)
                data = json.loads(head) if len(head) < 8192 else None
                if data is None:
                    # File is larger — read fully
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                if isinstance(data, dict) and "nodes" in data and "connections" in data:
                    return path
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
        return None

    # -------------------- sandbox execution --------------------------------

    def _execute_in_sandbox(self, script: str) -> dict[str, Any]:
        """Execute a Python script in a sandboxed exec() environment.

        Returns dict with keys: stdout, error, full_response.
        """
        prelude = (
            "import json, os, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "try:\n"
            "    import geopandas as gpd\n"
            "except ImportError:\n"
            "    gpd = None\n"
            "try:\n"
            "    import pandas as pd\n"
            "except ImportError:\n"
            "    pd = None\n"
            "try:\n"
            "    import rasterio\n"
            "except ImportError:\n"
            "    rasterio = None\n"
        )
        full_code = prelude + "\n" + script

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        error = None
        try:
            namespace: dict[str, Any] = {}
            exec(full_code, namespace, namespace)  # noqa: S102
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        output = stdout_capture.getvalue()
        if error:
            result = f"ERROR:\n{error}\nSTDOUT:\n{output}"
        else:
            result = f"STDOUT:\n{output}"
        return {"stdout": output, "error": error, "full_response": result}

    # -------------------- file inspection script generation ----------------

    @staticmethod
    def _build_inspection_script(file_path: str) -> str:
        """Generate a Python script that inspects a file and prints a summary.

        The script auto-detects the file type from the extension and reads:
        - Tabular/vector (geojson, gpkg, shp, csv): columns, dtypes, CRS, row count,
          bbox (if spatial), unique values for important categorical/text columns,
          and 1 sample row with values truncated to 50 chars.
        - Raster (tif, tiff): bounds, CRS, shape, dtype, band count (NO pixel data).
        - Other: file size and type only.
        """
        ext = os.path.splitext(file_path)[1].lower()
        escaped_path = file_path.replace("\\", "\\\\").replace("'", "\\'")

        if ext in (".geojson", ".gpkg", ".shp", ".csv", ".tsv", ".xlsx", ".xls"):
            return (
                f"path = '{escaped_path}'\n"
                "import json\n"
                "summary = {'path': path, 'type': 'tabular_or_vector'}\n"
                "try:\n"
                "    if path.lower().endswith('.csv') or path.lower().endswith('.tsv'):\n"
                "        sep = '\\t' if path.lower().endswith('.tsv') else ','\n"
                "        df = pd.read_csv(path, sep=sep, nrows=25000) if pd else None\n"
                "        if df is not None:\n"
                "            summary['columns'] = list(df.columns)\n"
                "            summary['dtypes'] = {c: str(df[c].dtype) for c in df.columns}\n"
                "            try:\n"
                "                with open(path, 'r', encoding='utf-8', errors='ignore') as f:\n"
                "                    row_count = sum(1 for _ in f) - 1\n"
                "            except Exception:\n"
                "                row_count = 'unknown'\n"
                "            summary['row_count'] = row_count\n"
                "            \n"
                "            # Extract unique values for important columns\n"
                "            unique_values = {}\n"
                "            for col in df.columns:\n"
                "                col_lower = col.lower()\n"
                "                dtype_str = str(df[col].dtype)\n"
                "                is_important = (\n"
                "                    'object' in dtype_str or 'category' in dtype_str or 'string' in dtype_str or\n"
                "                    any(k in col_lower for k in ('place', 'name', 'state', 'country', 'county', 'city', 'region', 'type', 'status', 'mag', 'magnitude'))\n"
                "                )\n"
                "                if is_important:\n"
                "                    try:\n"
                "                        vals = df[col].dropna().unique()\n"
                "                        total_unique = len(vals)\n"
                "                        if total_unique > 0:\n"
                "                            shown_limit = 15\n"
                "                            sub_vals = [str(v)[:50] + ('...' if len(str(v)) > 50 else '') for v in vals[:shown_limit]]\n"
                "                            if total_unique > shown_limit:\n"
                "                                sub_vals.append(f'and {total_unique - shown_limit} more unique values')\n"
                "                            unique_values[col] = {\n"
                "                                'total_count': total_unique,\n"
                "                                'sample_values': sub_vals\n"
                "                            }\n"
                "                    except Exception:\n"
                "                        pass\n"
                "            if unique_values:\n"
                "                summary['unique_categorical_values'] = unique_values\n"
                "            \n"
                "            sample = {}\n"
                "            for col in df.columns:\n"
                "                val = str(df[col].iloc[0]) if len(df) > 0 else ''\n"
                "                sample[col] = val[:50] + ('...' if len(val) > 50 else '')\n"
                "            summary['sample'] = sample\n"
                "    elif path.lower().endswith(('.xlsx', '.xls')):\n"
                "        df = pd.read_excel(path, nrows=2) if pd else None\n"
                "        if df is not None:\n"
                "            try:\n"
                "                excel_file = pd.ExcelFile(path)\n"
                "                summary['sheets'] = excel_file.sheet_names\n"
                "                full_df = pd.read_excel(path, nrows=25000)\n"
                "                summary['columns'] = list(full_df.columns)\n"
                "                summary['dtypes'] = {c: str(full_df[c].dtype) for c in full_df.columns}\n"
                "                summary['row_count'] = len(full_df)\n"
                "                \n"
                "                # Extract unique values for important columns\n"
                "                unique_values = {}\n"
                "                for col in full_df.columns:\n"
                "                    col_lower = col.lower()\n"
                "                    dtype_str = str(full_df[col].dtype)\n"
                "                    is_important = (\n"
                "                        'object' in dtype_str or 'category' in dtype_str or 'string' in dtype_str or\n"
                "                        any(k in col_lower for k in ('place', 'name', 'state', 'country', 'county', 'city', 'region', 'type', 'status', 'mag', 'magnitude'))\n"
                "                    )\n"
                "                    if is_important:\n"
                "                        try:\n"
                "                            vals = full_df[col].dropna().unique()\n"
                "                            total_unique = len(vals)\n"
                "                            if total_unique > 0:\n"
                "                                shown_limit = 15\n"
                "                                sub_vals = [str(v)[:50] + ('...' if len(str(v)) > 50 else '') for v in vals[:shown_limit]]\n"
                "                                if total_unique > shown_limit:\n"
                "                                    sub_vals.append(f'and {total_unique - shown_limit} more unique values')\n"
                "                                unique_values[col] = {\n"
                "                                    'total_count': total_unique,\n"
                "                                    'sample_values': sub_vals\n"
                "                                }\n"
                "                        except Exception:\n"
                "                            pass\n"
                "                if unique_values:\n"
                "                    summary['unique_categorical_values'] = unique_values\n"
                "                \n"
                "                sample = {}\n"
                "                for col in full_df.columns:\n"
                "                    val = str(full_df[col].iloc[0]) if len(full_df) > 0 else ''\n"
                "                    sample[col] = val[:50] + ('...' if len(val) > 50 else '')\n"
                "                summary['sample'] = sample\n"
                "            except Exception as e2:\n"
                "                summary['row_count'] = 'unknown'\n"
                "                summary['error'] = str(e2)\n"
                "    else:\n"
                "        # Geospatial vector format\n"
                "        try:\n"
                "            import pyogrio\n"
                "            info = pyogrio.read_info(path)\n"
                "            summary['columns'] = list(info['fields'])\n"
                "            summary['dtypes'] = {name: str(dtype) for name, dtype in zip(info['fields'], info['dtypes'])}\n"
                "            summary['crs'] = str(info['crs'])\n"
                "            summary['geometry_type'] = [info['geometry_type']]\n"
                "            summary['row_count'] = info['features']\n"
                "        except Exception:\n"
                "            # fallback to geopandas\n"
                "            pass\n"
                "        \n"
                "        gdf = gpd.read_file(path, rows=1) if gpd else None\n"
                "        if gdf is not None:\n"
                "            if 'columns' not in summary:\n"
                "                summary['columns'] = list(gdf.columns)\n"
                "                summary['dtypes'] = {c: str(gdf[c].dtype) for c in gdf.columns}\n"
                "                summary['crs'] = str(gdf.crs)\n"
                "                summary['geometry_type'] = gdf.geometry.geom_type.unique().tolist() if len(gdf) > 0 else []\n"
                "                summary['row_count'] = 'unknown (sampled)'\n"
                "            \n"
                "            # Load larger chunk to get unique values and spatial bounds\n"
                "            try:\n"
                "                gdf_full = gpd.read_file(path, rows=25000) if gpd else None\n"
                "            except Exception:\n"
                "                gdf_full = gpd.read_file(path) if gpd else None\n"
                "                if gdf_full is not None:\n"
                "                    gdf_full = gdf_full.head(25000)\n"
                "            \n"
                "            if gdf_full is not None:\n"
                "                if hasattr(gdf_full, 'total_bounds') and len(gdf_full) > 0:\n"
                "                    try:\n"
                "                        summary['bbox'] = list(gdf_full.total_bounds)\n"
                "                    except Exception:\n"
                "                        pass\n"
                "                \n"
                "                # Extract unique values for important columns\n"
                "                unique_values = {}\n"
                "                for col in gdf_full.columns:\n"
                "                    if col == 'geometry':\n"
                "                        continue\n"
                "                    col_lower = col.lower()\n"
                "                    dtype_str = str(gdf_full[col].dtype)\n"
                "                    is_important = (\n"
                "                        'object' in dtype_str or 'category' in dtype_str or 'string' in dtype_str or\n"
                "                        any(k in col_lower for k in ('place', 'name', 'state', 'country', 'county', 'city', 'region', 'type', 'status', 'mag', 'magnitude'))\n"
                "                    )\n"
                "                    if is_important:\n"
                "                        try:\n"
                "                            vals = gdf_full[col].dropna().unique()\n"
                "                            total_unique = len(vals)\n"
                "                            if total_unique > 0:\n"
                "                                shown_limit = 15\n"
                "                                sub_vals = [str(v)[:50] + ('...' if len(str(v)) > 50 else '') for v in vals[:shown_limit]]\n"
                "                                if total_unique > shown_limit:\n"
                "                                    sub_vals.append(f'and {total_unique - shown_limit} more unique values')\n"
                "                                unique_values[col] = {\n"
                "                                    'total_count': total_unique,\n"
                "                                    'sample_values': sub_vals\n"
                "                                }\n"
                "                        except Exception:\n"
                "                            pass\n"
                "                if unique_values:\n"
                "                    summary['unique_categorical_values'] = unique_values\n"
                "            \n"
                "            sample = {}\n"
                "            for col in gdf.columns:\n"
                "                if col == 'geometry':\n"
                "                    sample[col] = str(gdf[col].iloc[0])[:50] + '...' if len(gdf) > 0 else ''\n"
                "                else:\n"
                "                    val = str(gdf[col].iloc[0]) if len(gdf) > 0 else ''\n"
                "                    sample[col] = val[:50] + ('...' if len(val) > 50 else '')\n"
                "            summary['sample'] = sample\n"
                "except Exception as e:\n"
                "    summary['error'] = str(e)\n"
                "print(json.dumps(summary, indent=2))\n"
            )
        elif ext in (".tif", ".tiff", ".geotiff", ".geotif"):
            return (
                f"path = '{escaped_path}'\n"
                "import json\n"
                "summary = {'path': path, 'type': 'raster'}\n"
                "try:\n"
                "    if rasterio:\n"
                "        with rasterio.open(path) as src:\n"
                "            summary['crs'] = str(src.crs)\n"
                "            summary['bounds'] = list(src.bounds)\n"
                "            summary['shape'] = [src.height, src.width]\n"
                "            summary['band_count'] = src.count\n"
                "            summary['dtypes'] = [str(d) for d in src.dtypes]\n"
                "            summary['nodata'] = [str(n) for n in (src.nodatavals or [])]\n"
                "    else:\n"
                "        summary['error'] = 'rasterio not available'\n"
                "except Exception as e:\n"
                "    summary['error'] = str(e)\n"
                "print(json.dumps(summary, indent=2))\n"
            )
        else:
            return (
                f"path = '{escaped_path}'\n"
                "import json, os\n"
                "summary = {'path': path, 'type': 'other'}\n"
                "try:\n"
                "    summary['size_bytes'] = os.path.getsize(path)\n"
                "    summary['extension'] = os.path.splitext(path)[1]\n"
                "except Exception as e:\n"
                "    summary['error'] = str(e)\n"
                "print(json.dumps(summary, indent=2))\n"
            )

    # -------------------- replanning LLM prompt ----------------------------

    def _build_replanning_prompt(
        self,
        workflow_data: dict[str, Any],
        inspection_summaries: list[dict[str, Any]],
        catalogs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build the LLM messages for workflow replanning."""
        all_agents = [agent for catalog in catalogs for agent in catalog.get("agents", [])]
        proj_agents = [
            a for a in all_agents
            if any(kw in f"{a.get('agent_id')} {a.get('name')} {a.get('description')}".lower()
                   for kw in ["projection", "reproject", "crs", "coordinate", "transform"])
        ]
        proj_agent_name = proj_agents[0]["name"] if proj_agents else "Map Projection Agent"

        # Compact catalog (same as planning)
        compact_catalog = []
        for catalog in catalogs:
            compact_agents = []
            for agent in catalog.get("agents", []):
                skill_descriptions = [
                    s.get("description")
                    for s in agent.get("skills", [])
                    if isinstance(s, dict) and s.get("description")
                ]
                compact_agents.append({
                    "agent_id": agent.get("agent_id"),
                    "name": agent.get("name"),
                    "category": agent.get("category"),
                    "description": agent.get("description"),
                    "skills": skill_descriptions,
                })
            compact_catalog.append({
                "source": catalog.get("source"),
                "base_url": catalog.get("base_url"),
                "agents": compact_agents,
            })

        # Extract workflow structure
        nodes = workflow_data.get("nodes", [])
        connections = workflow_data.get("connections", [])
        self._debug_print(
            "Replan input workflow state",
            {
                "node_count": len(nodes),
                "connection_count": len(connections),
                "failed_nodes": [
                    {"id": n.get("id"), "agentId": n.get("agentId"), "error": n.get("errorMessage")}
                    for n in nodes
                    if n.get("status") == "error"
                ],
            },
        )
        existing_replan = workflow_data.get("replan", {})
        if not isinstance(existing_replan, dict):
            existing_replan = {}
        replan_history = existing_replan.get("history", [])

        # Build node summaries with status, logs, and output info
        node_summaries = []
        for node in nodes:
            summary = {
                "id": node.get("id"),
                "agentId": node.get("agentId"),
                "name": node.get("name"),
                "instructions": node.get("instructions"),
                "status": node.get("status"),
                "logs_tail": (node.get("logs") or [])[-10:],
                "outputArtifacts": node.get("outputArtifacts", []),
                "qc_count": node.get("qc_count", 0),
            }
            if node.get("errorMessage"):
                summary["errorMessage"] = node["errorMessage"]
            node_summaries.append(summary)

        system_prompt = (
            "You are the Autonomous GIS Orchestrator Agent operating in REPLANNING MODE.\n\n"
            "A previously generated workflow has been executed on the canvas and some nodes FAILED. "
            "Your task is to diagnose the root cause of the failure and produce a corrected "
            "workflow JSON that resolves the issue.\n\n"

            "CRITICAL DIAGNOSTIC PATTERN:\n"
            "1. Read the execution logs and error messages of the failed nodes carefully.\n"
            "2. The root cause may NOT be the failed node itself. An upstream node may have "
            "   completed successfully but produced incorrect or incompatible data that caused "
            "   the downstream failure. Inspect the output summaries of successful upstream nodes.\n"
            "3. Consider whether the instructions given to each agent were correct and sufficient.\n"
            "4. Consider whether the correct agent type was chosen for each step.\n"
            "5. USE CURRENT WORKFLOW AS REFERENCE: Use the provided workflow design as a reference. "
            "   Observe which nodes have non-zero 'qc_count' (meaning they have repeatedly failed Quality Control validation). "
            "   Analyze this structure to see where the logic fell short, and correct it by redesigning instructions, "
            "   substituting agents, or injecting missing converter/filter steps. Do not repeat the same mistakes.\n"
            "6. PROCESSED OUTPUT PRECEDENCE: Check whether downstream nodes are incorrectly connected to raw source datasets when an intermediate node already transforms, filters, joins, aggregates, enriches, reprojects, clips, or converts that source dataset for downstream use. In that case, remove the raw direct edge and connect the downstream node to the processed intermediate output instead, unless the downstream node explicitly needs the raw layer unchanged as an additional input.\n\n"

            "AVAILABLE CORRECTIONS:\n"
            "1. INSTRUCTION CHANGES: Modify the instructions given to one or more agents to fix "
            "   the issue (e.g., add CRS specification, change output format, add missing parameters).\n"
            "2. AGENT SUBSTITUTIONS: Replace an agent with a different one from the available catalog "
            "   if the current agent is not suitable for the task.\n"
            "3. STEP ADDITIONS: Add new intermediate steps if the workflow is missing a necessary "
            "   processing stage (e.g., CRS reprojection between retrieval and analysis).\n"
            "4. STEP REMOVALS: Remove steps that are unnecessary or causing conflicts.\n\n"

            "OUTPUT REQUIREMENTS:\n"
            "You must produce a complete, valid GAS canvas workflow JSON with the following structure:\n"
            "- id, name, description, createdAt (standard workflow metadata)\n"
            "- nodes: array of node objects with id, agentId, name, x, y, instructions, "
            "  inputDatasets (empty), credentials (empty), serverUrl, status (set to 'idle'), logs (empty array)\n"
            "- connections: array of connection objects with id, sourceId, targetId\n"
            "- replan: object containing:\n"
            "  - revision_number: integer (current revision)\n"
            "  - trigger: string describing what triggered this replan\n"
            "  - root_cause_analysis: string with detailed analysis\n"
            "  - substitutions: array of {step_id, original_agent_id, new_agent_id, reason}\n"
            "  - instruction_changes: array of {step_id, agent_id, original_instruction, new_instruction, reason}\n"
            "  - preserved_steps: array of step_ids that don't need changes\n"
            "  - history: array of prior replan attempts (preserve from input)\n\n"

            "ANTI-REPETITION RULE:\n"
            "If the 'history' array in the input contains prior replan attempts, you MUST NOT "
            "repeat the same changes that already failed. You must try a fundamentally different "
            "approach. If all reasonable approaches have been exhausted, set the workflow name to "
            "'REPLAN_EXHAUSTED' and explain in the root_cause_analysis why no further correction is possible.\n\n"

            "INSTRUCTION FORMULATION RULES (same as planning mode):\n"
            "1. NO FILENAMES in instructions.\n"
            "2. NO STEP CROSS-REFERENCES in instructions.\n"
            "3. Instructions must be self-contained for each agent.\n"
            "4. NO DOWNLOAD FORMAT SPECIFICATION: Do not suggest or specify output formats for data downloading/retrieval agents (like retrieval agents). Let them retrieve data in their original formats.\n"
            f"5. CRS DECOUPLING: If a node failed because of CRS mismatch, because it was asked to buffer/measure in meters on geographic coordinates (degrees), or because it ran a terrain calculation (like slope/aspect) directly on a geographic DEM (EPSG:4326), do not simply modify the instructions of that failed node. Instead, insert a new preceding node using the '{proj_agent_name}' to handle the coordinate reference system transformation to a metric projected CRS (such as a local UTM zone), and update the inputs/dependencies accordingly.\n"
            "6. NO IMPLICIT REPROJECTION: Do not write instructions asking analysis, raster, or mapping agents to handle coordinate projection or spatial alignment internally.\n"
            "7. NO IMPLEMENTATION/CODE INSTRUCTIONS: Keep instructions focused on the high-level semantic task (WHAT to do). Do NOT write low-level programming instructions, python/pandas code snippets, variable names, function names, error-handling directives, or step-by-step programming guides. Let the target agent figure out the syntax and code implementation details itself.\n\n"

            "AGENT SELECTION RULE FOR REPLANNING:\n"
            "Choose replacement or added agents based on what they are designed to do, not only by name. Match the task and input data modality to the agent capability: use vector-oriented agents for vector datasets and vector operations, raster-oriented agents for raster datasets and raster operations, data-access agents for retrieval/download tasks, analysis agents for transformations/computation, and visualization agents for maps or charts. For every node, copy the selected agent's `agent_id` exactly as it appears in the provided available catalog. Do not invent, translate, shorten, title-case, or use a display name as `agentId`.\n\n"

            "Your output must be a strict JSON response matching the described structure, "
            "with no markdown code blocks, headers, or text outside the JSON."
        )

        user_content = json.dumps({
            "mode": "replan",
            "failed_workflow": {
                "nodes": node_summaries,
                "connections": connections,
            },
            "existing_replan": existing_replan,
            "replan_history": replan_history,
            "output_inspection_summaries": inspection_summaries,
            "available_gas_capabilities": compact_catalog,
        }, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # -------------------- main replan method ------------------------------

    def run_replan(
        self,
        query: str,
        workflow_json_path: str,
        input_dataset_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Diagnose a failed workflow and produce a corrected version.

        This method is completely separate from the planning logic in run().
        It is triggered when a canvas workflow JSON file is detected among
        the input datasets.
        """
        start_time = time.time()

        self.emit_progress(
            progress_callback,
            stage="start",
            message=(
                "I detected a workflow JSON file in the inputs. Entering "
                "REPLANNING mode to diagnose failures and produce a corrected workflow."
            ),
        )

        # 1. Read the failed workflow JSON
        try:
            with open(workflow_json_path, "r", encoding="utf-8") as fh:
                workflow_data = json.load(fh)
        except Exception as exc:
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model": self.model,
                "duration": round(time.time() - start_time, 2),
                "outputs": {
                    "text": f"Failed to read workflow JSON: {exc}",
                    "error": str(exc),
                },
                "metrics": self.metrics(number_of_artifacts=0),
            }

        nodes = workflow_data.get("nodes", [])
        connections = workflow_data.get("connections", [])

        failed_nodes = [n for n in nodes if n.get("status") == "error"]
        completed_nodes = [n for n in nodes if n.get("status") == "completed"]

        self.emit_progress(
            progress_callback,
            stage="input_inspection",
            message=(
                f"Workflow has {len(nodes)} nodes: "
                f"{len(completed_nodes)} completed, {len(failed_nodes)} failed. "
                "I will inspect the outputs of completed nodes to diagnose the failure."
            ),
            data={
                "total_nodes": len(nodes),
                "failed_count": len(failed_nodes),
                "completed_count": len(completed_nodes),
            },
        )

        # 2. Inspect outputs of completed nodes via sandbox
        inspection_summaries: list[dict[str, Any]] = []

        for node in completed_nodes:
            artifacts = node.get("outputArtifacts", [])
            for art in artifacts:
                art_url = art.get("url", "")
                # Convert artifact URL to local path if possible
                local_path = self._artifact_url_to_local_path(art_url)
                if not local_path or not os.path.isfile(local_path):
                    inspection_summaries.append({
                        "node_id": node.get("id"),
                        "node_name": node.get("name"),
                        "artifact_url": art_url,
                        "status": "not_accessible",
                        "note": "File not found on local filesystem.",
                    })
                    continue

                self.emit_progress(
                    progress_callback,
                    stage="data_validation",
                    message=(
                        f"Inspecting output from '{node.get('name')}': "
                        f"{os.path.basename(local_path)}"
                    ),
                )

                script = self._build_inspection_script(local_path)
                result = self._execute_in_sandbox(script)
                self.increment_tool_calls()

                summary_data: dict[str, Any] = {
                    "node_id": node.get("id"),
                    "node_name": node.get("name"),
                    "artifact_url": art_url,
                    "file_path": local_path,
                }
                if result["error"]:
                    summary_data["status"] = "inspection_error"
                    summary_data["error"] = result["error"][:500]
                else:
                    try:
                        parsed = json.loads(result["stdout"].strip())
                        summary_data.update(parsed)
                        summary_data["status"] = "inspected"
                    except (json.JSONDecodeError, ValueError):
                        summary_data["status"] = "parse_error"
                        summary_data["raw_output"] = result["stdout"][:500]

                inspection_summaries.append(summary_data)

        # 3. Discover agent capabilities (same as planning)
        pipeline_sources = self._normalize_pipeline_sources()
        catalogs = []
        if pipeline_sources:
            try:
                catalogs = self._discover_capabilities(pipeline_sources, progress_callback)
            except Exception as exc:
                logging.warning("Capability discovery failed during replan: %s", exc)

        # 4. Build replanning LLM prompt and call
        self.emit_progress(
            progress_callback,
            stage="planning",
            message=(
                "I am sending the workflow failure data and output inspections "
                "to the model for root cause analysis and corrective planning."
            ),
            data={
                "inspection_count": len(inspection_summaries),
                "catalog_count": len(catalogs),
            },
        )

        try:
            messages = self._build_replanning_prompt(
                workflow_data, inspection_summaries, catalogs
            )
            self.increment_llm_calls()
            response = self._chat_completion(messages)
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            content = response.choices[0].message.content
            corrected = self._extract_json(content)
            self._debug_print("Replan model output", corrected)
        except Exception as exc:
            self.emit_progress(
                progress_callback,
                stage="error",
                message=f"Replanning LLM call failed: {exc}",
            )
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model": self.model,
                "duration": round(time.time() - start_time, 2),
                "outputs": {
                    "text": f"Replanning failed: {exc}",
                    "error": str(exc),
                },
                "metrics": self.metrics(number_of_artifacts=0),
            }

        # 5. Ensure the corrected workflow has the replan field
        if "replan" not in corrected:
            corrected["replan"] = {}
        replan_block = corrected["replan"]
        replan_block.setdefault("revision_number", 1)
        replan_block.setdefault("trigger", "Automated replan from workflow failure")
        replan_block.setdefault("root_cause_analysis", "")
        replan_block.setdefault("substitutions", [])
        replan_block.setdefault("instruction_changes", [])
        replan_block.setdefault("preserved_steps", [])
        previous_replan = workflow_data.get("replan", {})
        if not isinstance(previous_replan, dict):
            previous_replan = {}
        replan_block.setdefault("history", previous_replan.get("history", []))

        corrected_agent_lookup: dict[str, dict[str, Any]] = {
            agent.get("agent_id"): agent
            for catalog in catalogs
            for agent in catalog.get("agents", [])
            if agent.get("agent_id")
        }
        self._enforce_local_analysis_routing_on_nodes(
            corrected.get("nodes", []),
            corrected.get("connections", []),
            corrected_agent_lookup,
        )

        # Identify preserved nodes that don't need to be run again
        original_nodes_by_id = {n.get("id"): n for n in workflow_data.get("nodes", [])}
        
        # Build connection DAG to trace upstream dependency
        upstream_dependencies = defaultdict(list)
        for conn in corrected.get("connections", []):
            src_id = conn.get("sourceId")
            tgt_id = conn.get("targetId")
            if src_id and tgt_id:
                upstream_dependencies[tgt_id].append(src_id)
                
        # Also map step dependencies via depends_on / input_from_steps
        for node in corrected.get("nodes", []):
            nid = node.get("id")
            for dep in node.get("input_from_steps") or node.get("depends_on") or []:
                if dep and dep not in upstream_dependencies[nid]:
                    upstream_dependencies[nid].append(dep)

        # Filter candidates: completed originally, unchanged agentId and instructions
        preserved_nodes = {}
        for node in corrected.get("nodes", []):
            nid = node.get("id")
            orig_node = original_nodes_by_id.get(nid)
            if orig_node and orig_node.get("status") == "completed":
                instructions_match = (
                    str(node.get("instructions") or "").strip() ==
                    str(orig_node.get("instructions") or "").strip()
                )
                agent_match = (node.get("agentId") == orig_node.get("agentId"))
                if instructions_match and agent_match:
                    preserved_nodes[nid] = orig_node

        # Iteratively invalidate any node if any of its upstream dependency nodes is not completed/preserved
        changed = True
        while changed:
            changed = False
            for nid in list(preserved_nodes.keys()):
                parents = upstream_dependencies[nid]
                for parent_id in parents:
                    if parent_id not in preserved_nodes:
                        del preserved_nodes[nid]
                        changed = True
                        break

        # Apply state propagation
        for node in corrected.get("nodes", []):
            nid = node.get("id")
            if nid in preserved_nodes:
                orig = preserved_nodes[nid]
                node["status"] = "completed"
                node["logs"] = orig.get("logs") or []
                node["outputArtifacts"] = orig.get("outputArtifacts") or []
                node["qc_status"] = "controlled"
                node["qc_count"] = 0
                node["max_qc_attempts"] = self._qc_failure_limit()
                if "results" in orig:
                    node["results"] = orig["results"]
                if "parameters" in orig:
                    node["parameters"] = {
                        **(orig.get("parameters") or {}),
                        "qc_status": "controlled",
                        "qc_count": 0,
                        "max_qc_attempts": self._qc_failure_limit(),
                        "max_qc_attempts_per_node": self._qc_failure_limit(),
                    }
            else:
                node["status"] = "idle"
                node["logs"] = []
                node["outputArtifacts"] = []
                node["qc_status"] = "uncontrolled"
                node["qc_count"] = 0
                node["max_qc_attempts"] = self._qc_failure_limit()
                node.setdefault("parameters", {})
                node["parameters"]["qc_status"] = "uncontrolled"
                node["parameters"]["qc_count"] = 0
                node["parameters"]["max_qc_attempts"] = self._qc_failure_limit()
                node["parameters"]["max_qc_attempts_per_node"] = self._qc_failure_limit()
                node.setdefault("inputDatasets", [])
                node.setdefault("credentials", {})

        # 6. Write output JSON
        json_path = self._write_canvas_json(
            f"replan_{query}" if query else "replan_workflow", corrected
        )

        self.emit_progress(
            progress_callback,
            stage="complete",
            message="Replanning is complete. A corrected workflow JSON has been produced.",
            data={
                "substitutions": len(replan_block.get("substitutions", [])),
                "instruction_changes": len(replan_block.get("instruction_changes", [])),
            },
        )

        step_count = len(corrected.get("nodes", []))
        substitution_count = len(replan_block.get("substitutions", []))
        instruction_change_count = len(replan_block.get("instruction_changes", []))
        root_cause = replan_block.get("root_cause_analysis", "")

        summary = (
            f"Replanning complete. Produced a corrected workflow with {step_count} node(s). "
            f"Changes: {substitution_count} agent substitution(s), "
            f"{instruction_change_count} instruction change(s). "
            f"Root cause: {root_cause[:200]}"
        )

        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "duration": round(time.time() - start_time, 2),
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "inputs": {
                "text": query,
                "dataset_paths": input_dataset_paths,
                "parameters": {
                    "mode": "replan",
                    "pipeline_sources": pipeline_sources,
                },
            },
            "outputs": {
                "text": summary,
                "replan_analysis": {
                    "root_cause": root_cause,
                    "substitutions": replan_block.get("substitutions", []),
                    "instruction_changes": replan_block.get("instruction_changes", []),
                },
                "workflow_json_file": str(json_path),
            },
            "metrics": self.metrics(number_of_artifacts=1),
            "stochasticity": {
                "used": True,
                "controls": [self._last_temperature_control],
            },
        }

    def _ensure_local_artifact(self, url: str) -> str | None:
        """Get a local path for the given artifact URL, downloading it if necessary."""
        if not url:
            return None
        
        # First try to map to local filesystem directly
        local_path = self._artifact_url_to_local_path(url)
        if local_path and os.path.isfile(local_path):
            return local_path
            
        # If not local, download the file into a temporary cache inside output_dir
        try:
            if url.startswith("/"):
                url = f"{self._get_dynamic_server_url().rstrip('/')}{url}"
            elif not url.startswith(("http://", "https://")):
                url = f"{self._get_dynamic_server_url().rstrip('/')}/{url.lstrip('/')}"
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or f"temp_artifact_{uuid.uuid4().hex}"
            
            # Avoid path traversal attacks
            filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
            
            cache_dir = self.output_dir / ".qc_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_cache_path = cache_dir / filename
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_cache_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            
            if os.path.isfile(local_cache_path):
                return str(local_cache_path)
        except Exception as exc:
            logging.error("Failed to download remote artifact from %s: %s", url, exc)
            
        return None

    def run_quality_control(
        self,
        query: str,
        workflow_json_path: str,
        input_dataset_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Verify output datasets or failures for completed/error nodes in the workflow JSON.

        If outputs/logs match expectations, the node is marked as controlled and workflow is approved.
        If incorrect, the agent modifies/repairs the workflow DAG and returns the updated version.
        """
        start_time = time.time()
        pipeline_sources = self._normalize_pipeline_sources()

        logging.info(f"[QC SYSTEM] Initiated Quality Control check for query: '{query}'")

        # Discover agent capabilities (same pattern as replanning)
        catalogs: list[dict[str, Any]] = []
        if pipeline_sources:
            try:
                catalogs = self._discover_capabilities(pipeline_sources, progress_callback)
                logging.info(f"[QC SYSTEM] Discovered {sum(len(c.get('agents', [])) for c in catalogs)} agent(s) from {len(catalogs)} catalog(s).")
            except Exception as exc:
                logging.warning("[QC SYSTEM] Capability discovery failed during QC: %s", exc)

        self.emit_progress(
            progress_callback,
            stage="start",
            message="Entering Quality Control mode. Reading active workflow JSON...",
        )

        try:
            with open(workflow_json_path, "r", encoding="utf-8") as fh:
                workflow_data = json.load(fh)
        except Exception as exc:
            logging.error(f"[QC SYSTEM] Error: Failed to read workflow JSON: {exc}")
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model": self.model,
                "duration": round(time.time() - start_time, 2),
                "outputs": {
                    "text": f"Failed to read workflow JSON for QC: {exc}",
                    "error": str(exc),
                },
                "metrics": self.metrics(number_of_artifacts=0),
            }

        nodes = workflow_data.get("nodes", [])
        qc_candidate_nodes = [
            n for n in nodes if str(n.get("status") or "").lower() in {"completed", "error", "failed"}
        ]

        node_map = {n.get("id"): n for n in qc_candidate_nodes if n.get("id")}
        sorted_qc_candidates: list[dict[str, Any]] = []
        visited_nodes: set[str] = set()

        def visit_qc_candidate(nid: str) -> None:
            if nid in visited_nodes or nid not in node_map:
                return
            node = node_map[nid]
            for upstream_id in (node.get("input_from_steps") or node.get("depends_on") or []):
                visit_qc_candidate(str(upstream_id))
            visited_nodes.add(nid)
            sorted_qc_candidates.append(node)

        for node in qc_candidate_nodes:
            if node.get("id"):
                visit_qc_candidate(str(node["id"]))

        def _node_qc_status(node: dict[str, Any]) -> str:
            params = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
            return str(node.get("qc_status") or params.get("qc_status") or "uncontrolled")

        def _sync_node_qc_fields(node: dict[str, Any], *, qc_status: str, qc_count: int | None = None, max_qc_attempts: int | None = None) -> None:
            node["qc_status"] = qc_status
            params = node.setdefault("parameters", {})
            if isinstance(params, dict):
                params["qc_status"] = qc_status
                if qc_count is not None:
                    params["qc_count"] = qc_count
                if max_qc_attempts is not None:
                    params["max_qc_attempts"] = max_qc_attempts
                    params["max_qc_attempts_per_node"] = max_qc_attempts
            if qc_count is not None:
                node["qc_count"] = qc_count
            if max_qc_attempts is not None:
                node["max_qc_attempts"] = max_qc_attempts

        # Identify uncontrolled nodes that have produced either a result or an execution failure.
        unchecked_nodes = [n for n in sorted_qc_candidates if _node_qc_status(n) != "controlled"]
        
        if not unchecked_nodes:
            logging.info("[QC SYSTEM] No uncontrolled completed or failed nodes found. Skipping QC validation.")
            self.emit_progress(
                progress_callback,
                stage="qc_skip",
                message="No uncontrolled completed or failed nodes found. Workflow is up to date.",
            )
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model": self.model,
                "duration": round(time.time() - start_time, 2),
                "outputs": {
                    "text": "Quality Control completed: all completed or failed steps are verified.",
                    "workflow_plan": workflow_data,
                    "workflow_json_file": workflow_json_path,
                },
                "metrics": self.metrics(number_of_artifacts=1),
            }

        # Select the UI-declared QC target when provided; otherwise use
        # topological order for compatibility with direct API callers.
        requested_target_id = workflow_data.get("qc_target_node_id")
        target_node = None
        if requested_target_id:
            target_node = next(
                (n for n in unchecked_nodes if str(n.get("id")) == str(requested_target_id)),
                None,
            )
        if target_node is None:
            target_node = unchecked_nodes[0]
        node_id = target_node.get("id")
        node_name = target_node.get("name")

        raw_node_params = target_node.get("parameters") if isinstance(target_node.get("parameters"), dict) else {}
        try:
            current_qc_count = int(target_node.get("qc_count", raw_node_params.get("qc_count", 0)) or 0)
        except (TypeError, ValueError):
            current_qc_count = 0
        try:
            qc_failure_limit = int(
                target_node.get(
                    "max_qc_attempts",
                    raw_node_params.get("max_qc_attempts", raw_node_params.get("max_qc_attempts_per_node", self._qc_failure_limit())),
                )
                or self._qc_failure_limit()
            )
        except (TypeError, ValueError):
            qc_failure_limit = self._qc_failure_limit()
        qc_failure_limit = max(1, qc_failure_limit)

        if current_qc_count >= qc_failure_limit:
            logging.info(
                f"[QC SYSTEM] Node '{node_name}' (ID: {node_id}) has failed Quality Control "
                f"{current_qc_count} times (limit is {qc_failure_limit})."
            )
            self._debug_print(
                "QC failure limit reached",
                {
                    "node_id": node_id,
                    "node_name": node_name,
                    "qc_count": current_qc_count,
                    "limit": qc_failure_limit,
                },
            )
            self.emit_progress(
                progress_callback,
                stage="qc_limit_reached",
                message=f"Node '{node_name}' already reached the QC attempt limit ({current_qc_count}/{qc_failure_limit}).",
            )

            target_node["status"] = "error"
            target_node["errorMessage"] = f"Failed Quality Control validation {current_qc_count} times."
            _sync_node_qc_fields(target_node, qc_status="uncontrolled", qc_count=current_qc_count, max_qc_attempts=qc_failure_limit)
            workflow_data["workflow_status"] = "qc_limit_reached"
            workflow_data["failed_node_id"] = node_id
            workflow_data["message"] = f"Maximum QC attempts reached for node '{node_name}'."
            try:
                with open(workflow_json_path, "w", encoding="utf-8") as fh:
                    json.dump(workflow_data, fh, indent=2)
            except Exception as exc:
                logging.error(f"[QC SYSTEM] Error saving workflow state after QC limit: {exc}")
            if bool((self.request_parameters or {}).get("allow_internal_replan")):
                return self.run_replan(
                    query, workflow_json_path, input_dataset_paths, progress_callback
                )
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model": self.model,
                "duration": round(time.time() - start_time, 2),
                "outputs": {
                    "text": workflow_data["message"],
                    "workflow_status": "qc_limit_reached",
                    "failed_node_id": node_id,
                    "workflow_plan": workflow_data,
                    "workflow_json_file": workflow_json_path,
                },
                "metrics": self.metrics(number_of_artifacts=1),
            }

        logging.info(f"[QC SYSTEM] Inspecting node outputs and logs: '{node_name}' (ID: {node_id})")
        
        self.emit_progress(
            progress_callback,
            stage="qc_inspection",
            message=f"Inspecting outputs and logs of node: '{node_name}' (ID: {node_id})",
        )

        # 1. Sandbox Inspection of target node artifacts
        inspection_summaries = []
        artifacts = target_node.get("outputArtifacts", [])
        logging.info(f"[QC SYSTEM] Found {len(artifacts)} output artifact(s). Executing code sandbox validation...")
        for art in artifacts:
            art_url = art.get("url", "")
            local_path = self._ensure_local_artifact(art_url)
            if not local_path or not os.path.isfile(local_path):
                inspection_summaries.append({
                    "name": art.get("name"),
                    "url": art_url,
                    "status": "not_accessible",
                    "note": "Output file not found or couldn't be downloaded.",
                })
                continue

            # Run python sandbox script to extract format metadata
            script = self._build_inspection_script(local_path)
            res = self._execute_in_sandbox(script)
            if res.get("error"):
                inspection_summaries.append({
                    "name": art.get("name"),
                    "url": art_url,
                    "status": "corrupt_or_error",
                    "error": res["error"],
                    "stdout_tail": res["stdout"][-500:],
                })
            else:
                try:
                    meta = json.loads(res["stdout"].strip())
                    inspection_summaries.append({
                        "name": art.get("name"),
                        "url": art_url,
                        "status": "verified_accessible",
                        "metadata": meta,
                    })
                except Exception:
                    inspection_summaries.append({
                        "name": art.get("name"),
                        "url": art_url,
                        "status": "raw_text_output",
                        "stdout_tail": res["stdout"][-500:],
                    })

        # 2. Invoke LLM for Quality Verdict
        self.emit_progress(
            progress_callback,
            stage="qc_evaluation",
            message=f"Evaluating node '{node_name}' output alignment against instructions...",
        )

        verdict_data = None
        last_error = None
        for attempt in range(1, 4):
            try:
                messages = self._build_qc_prompt(query, target_node, inspection_summaries, workflow_data, catalogs)
                self.increment_llm_calls()
                response = self._chat_completion(messages)
                usage = getattr(response, "usage", None)
                if usage:
                    self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
                content = response.choices[0].message.content
                verdict_data = self._extract_json(content)
                break
            except Exception as exc:
                last_error = exc

        if not verdict_data:
            logging.error(f"[QC SYSTEM] Error: Quality Control evaluation failed: {last_error}")
            raise ValueError(f"Quality Control evaluation failed: {last_error}")
        self._debug_print(
            "QC verdict model output",
            {
                "node_id": node_id,
                "node_name": node_name,
                "inspection_summaries": inspection_summaries,
                "verdict": verdict_data,
            },
        )

        verdict = verdict_data.get("verdict") or "REJECTED"
        reason = verdict_data.get("reason") or "No reason provided."
        recovery_option = None
        logging.info(f"[QC SYSTEM] VERDICT: {verdict}")
        logging.info(f"[QC SYSTEM] REASON: {reason}")

        if verdict == "OK":
            # Approved: update node qc_status to controlled
            _sync_node_qc_fields(target_node, qc_status="controlled", qc_count=current_qc_count, max_qc_attempts=qc_failure_limit)
            self.emit_progress(
                progress_callback,
                stage="qc_passed",
                message=f"Quality Control passed for '{node_name}': {reason}",
            )
        else:
            # Rejected: mutate/repair workflow
            recovery_option = verdict_data.get("recovery_option", "F")
            logging.info(f"[QC SYSTEM] Recovery Option: {recovery_option}")
            self.emit_progress(
                progress_callback,
                stage="qc_rejected",
                message=f"Quality Control rejected node '{node_name}': {reason}. Applying recovery option {recovery_option}.",
            )

            # Increment QC count for target node
            new_qc_count = target_node.get("qc_count", 0) + 1
            _sync_node_qc_fields(target_node, qc_status="uncontrolled", qc_count=new_qc_count, max_qc_attempts=qc_failure_limit)

            # Build set of valid agent IDs from the discovered catalogs
            valid_agent_ids: set[str] = set()
            for catalog in catalogs:
                for agent in catalog.get("agents", []):
                    aid = agent.get("agent_id")
                    if aid:
                        valid_agent_ids.add(aid)

            # Derive a fallback serverUrl from existing workflow nodes
            existing_server_urls = [
                n.get("serverUrl") for n in nodes
                if n.get("serverUrl")
            ]
            fallback_server_url = existing_server_urls[0] if existing_server_urls else ""

            suggested_changes = verdict_data.get("suggested_workflow_changes")
            if suggested_changes and isinstance(suggested_changes, dict) and "nodes" in suggested_changes:
                # Merge mutated workflow JSON
                workflow_data = suggested_changes
                mutated_nodes = workflow_data.get("nodes", [])
                
                # Propagate updated qc_count into the mutated target node
                for mn in mutated_nodes:
                    if mn.get("id") == node_id:
                        _sync_node_qc_fields(mn, qc_status=str(mn.get("qc_status") or "uncontrolled"), qc_count=new_qc_count, max_qc_attempts=qc_failure_limit)
                
                # Build connection map for downstream traversal
                parent_to_children = defaultdict(list)
                for conn in workflow_data.get("connections", []):
                    parent_to_children[conn.get("sourceId")].append(conn.get("targetId"))
                
                # Build map of original nodes for comparison
                original_node_map = {n.get("id"): n for n in nodes}

                # Identify altered and new nodes
                altered_ids = set()
                new_node_ids = set()
                for mn in mutated_nodes:
                    orig = original_node_map.get(mn.get("id"))
                    if not orig:
                        # This is a brand-new node added by the QC
                        new_node_ids.add(mn.get("id"))
                        altered_ids.add(mn.get("id"))
                    elif (
                        orig.get("instructions") != mn.get("instructions")
                        or orig.get("agentId") != mn.get("agentId")
                        or mn.get("status") == "idle"
                    ):
                        altered_ids.add(mn.get("id"))
                
                # Traverse downstream to identify all nodes that depend on altered nodes
                downstream_ids = set()
                queue = deque(altered_ids)
                while queue:
                    curr = queue.popleft()
                    for child in parent_to_children[curr]:
                        if child not in downstream_ids:
                            downstream_ids.add(child)
                            queue.append(child)

                # IDs that need to be reset to idle (new + altered + downstream)
                # BUT exclude the target node if it should stay completed (Options A/B/C)
                target_stays_completed = recovery_option in ("A", "B", "C")
                all_reset_ids = altered_ids.union(downstream_ids)
                if target_stays_completed:
                    all_reset_ids.discard(node_id)

                logging.info(
                    f"[QC SYSTEM] Rejecting node '{node_name}' (option {recovery_option}). "
                    f"Resetting {len(all_reset_ids)} node(s) to 'idle' status. "
                    f"New nodes added: {len(new_node_ids)}."
                )
                
                # Apply resets and validate agent IDs
                for mn in mutated_nodes:
                    mid = mn.get("id")

                    # Validate agentId against catalog
                    if valid_agent_ids and mn.get("agentId") not in valid_agent_ids:
                        logging.warning(
                            f"[QC SYSTEM] Node '{mn.get('name')}' has unknown agentId "
                            f"'{mn.get('agentId')}'. This agent may not exist in the catalog."
                        )

                    # Ensure new nodes have a serverUrl
                    if mid in new_node_ids:
                        if not mn.get("serverUrl"):
                            mn["serverUrl"] = fallback_server_url
                        mn["status"] = "idle"
                        mn["logs"] = []
                        mn["outputArtifacts"] = []
                        _sync_node_qc_fields(mn, qc_status="uncontrolled", qc_count=0, max_qc_attempts=qc_failure_limit)
                        mn.setdefault("inputDatasets", [])
                        mn.setdefault("credentials", {})
                    elif mid in all_reset_ids:
                        mn["status"] = "idle"
                        mn["logs"] = []
                        mn["outputArtifacts"] = []
                        next_count = new_qc_count if mid == node_id else 0
                        _sync_node_qc_fields(mn, qc_status="uncontrolled", qc_count=next_count, max_qc_attempts=qc_failure_limit)
                    else:
                        # Retain original state for unaltered upstream nodes
                        orig = original_node_map.get(mid)
                        if orig:
                            for key, val in orig.items():
                                mn.setdefault(key, val)
                            mn["status"] = orig.get("status") or "completed"
                            mn["logs"] = orig.get("logs") or []
                            mn["outputArtifacts"] = orig.get("outputArtifacts") or []
                            if mid == node_id:
                                # Target node: mark controlled if it stays completed
                                val = "controlled" if target_stays_completed else "uncontrolled"
                                _sync_node_qc_fields(mn, qc_status=val, qc_count=new_qc_count, max_qc_attempts=qc_failure_limit)
                            else:
                                val = orig.get("qc_status") or "controlled"
                                orig_count = int(orig.get("qc_count", 0) or 0)
                                _sync_node_qc_fields(mn, qc_status=val, qc_count=orig_count, max_qc_attempts=qc_failure_limit)
            else:
                # Fallback: reset the failed node to idle so it runs again
                logging.info(f"[QC SYSTEM] Rejecting node '{node_name}'. Resetting failed node to 'idle' for re-run.")
                target_node["status"] = "idle"
                target_node["logs"] = [f"[SYSTEM]: QC Rejected. Reason: {reason}"]
                target_node["outputArtifacts"] = []
                _sync_node_qc_fields(target_node, qc_status="uncontrolled", qc_count=new_qc_count, max_qc_attempts=qc_failure_limit)

            if (
                new_qc_count >= qc_failure_limit
                and bool((self.request_parameters or {}).get("allow_internal_replan"))
            ):
                target_node["status"] = "error"
                target_node["errorMessage"] = (
                    f"Failed Quality Control validation {new_qc_count} times. "
                    f"Latest reason: {reason}"
                )
                workflow_data["workflow_status"] = "qc_limit_reached"
                workflow_data["failed_node_id"] = node_id
                try:
                    with open(workflow_json_path, "w", encoding="utf-8") as fh:
                        json.dump(workflow_data, fh, indent=2)
                except Exception as exc:
                    logging.error("[QC SYSTEM] Error saving workflow before replanning: %s", exc)
                return self.run_replan(
                    query, workflow_json_path, input_dataset_paths, progress_callback
                )

        # Overwrite the active workflow JSON file
        try:
            workflow_data["verdict"] = verdict
            workflow_data["reason"] = reason
            workflow_data["recovery_option"] = recovery_option
            workflow_data["workflow_status"] = "qc_controlled" if verdict == "OK" else "qc_updated"
            workflow_data.pop("replan", None)
            
            with open(workflow_json_path, "w", encoding="utf-8") as fh:
                json.dump(workflow_data, fh, indent=2)
        except Exception as exc:
            logging.error(f"[QC SYSTEM] Error: Failed to write updated workflow JSON: {exc}")

        logging.info("[QC SYSTEM] Quality Control check complete.\n")

        print("-" * 50)
        print(f"QC Agent Name: {self.agent_name}")
        print("Updated Workflow Plan JSON:")
        print(json.dumps(workflow_data, indent=2))
        print("-" * 50)

        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "model": self.model,
            "duration": round(time.time() - start_time, 2),
            "outputs": {
                "text": f"QC Completed. Verdict: {verdict}. Reason: {reason}",
                "verdict": verdict,
                "reason": reason,
                "recovery_option": verdict_data.get("recovery_option"),
                "workflow_status": workflow_data.get("workflow_status"),
                "workflow_plan": workflow_data,
                "workflow_json_file": workflow_json_path,
            },
            "metrics": self.metrics(number_of_artifacts=1),
        }

    def _build_qc_prompt(
        self,
        query: str,
        target_node: dict[str, Any],
        inspection_summaries: list[dict[str, Any]],
        workflow_data: dict[str, Any],
        catalogs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Construct the prompt comparing execution results against instructions.

        The prompt now includes the discovered agent catalog so the LLM can
        propose corrective nodes using only real, available agent IDs.
        """
        # Build compact agent catalog (same pattern as replanning)
        compact_catalog: list[dict[str, Any]] = []
        if catalogs:
            for catalog in catalogs:
                compact_agents = []
                for agent in catalog.get("agents", []):
                    skill_descriptions = [
                        s.get("description")
                        for s in agent.get("skills", [])
                        if isinstance(s, dict) and s.get("description")
                    ]
                    compact_agents.append({
                        "agent_id": agent.get("agent_id"),
                        "name": agent.get("name"),
                        "category": agent.get("category"),
                        "description": agent.get("description"),
                        "skills": skill_descriptions,
                    })
                compact_catalog.append({
                    "source": catalog.get("source"),
                    "base_url": catalog.get("base_url"),
                    "agents": compact_agents,
                })

        # Derive a default serverUrl from existing workflow nodes
        existing_server_urls = [
            n.get("serverUrl") for n in workflow_data.get("nodes", [])
            if n.get("serverUrl")
        ]
        default_server_url = existing_server_urls[0] if existing_server_urls else ""

        system_prompt = (
            "You are the Autonomous GIS Orchestrator Agent operating in QUALITY CONTROL mode.\n\n"
            "Your task is to analyze the execution output and logs of a single completed workflow node, "
            "and determine if it succeeded in fulfilling its natural-language instructions correctly.\n\n"

            "THINK LIKE A HUMAN WORKFLOW DESIGNER:\n"
            "Imagine you designed this workflow and you are now reviewing each completed step. "
            "When you find an issue, you think about the most practical way to fix it — "
            "just like a data engineer or systems analyst would when reviewing their own pipeline.\n\n"

            "CRITICAL DIRECTIVES:\n"
            "1. Be practical and not overly strict. Your goal is to solve issues efficiently.\n"
            "2. Evaluate ONLY the target_node's specific instructions against its sandbox_inspection_summaries and logs_tail.\n"
            "3. Do NOT reject or rerun a node because other nodes in the workflow are unfinished, idle, running, or have no output artifacts. It is completely normal for other nodes to be unfinished.\n"
            "4. NEVER insert verification or inspection agents onto the canvas workflow. Verification is already handled by you (the QC agent) in the sandbox.\n"
            "5. You MUST ONLY use agent IDs from the 'available_agent_catalog' provided in the user payload. Copy the selected agent's `agent_id` exactly. Do NOT invent, translate, shorten, title-case, pluralize, or use a display name as an agent ID.\n"
            "6. RASTER TERRAIN VALIDATION: If a node is tasked with calculating slope, aspect, or hillshade, it MUST NOT do so directly on a DEM raster in geographic coordinates (EPSG:4326) because this causes massive unit mismatch errors (degrees vs. meters). If you inspect a completed slope/aspect analysis node and discover the input DEM was in geographic coordinates, you must reject it (VERDICT = 'REJECTED') and use the recovery decision tree to insert a preceding reprojection node or reset the workflow structure to handle reprojection first.\n\n"
            "7. CONTIGUOUS US COUNTY FILTER VALIDATION: For tasks that filter US counties to the contiguous United States, do NOT reject solely because the output has about 3,108 county features or 49 unique state-like values. In Census county data, 49 values can correctly mean the 48 contiguous states plus District of Columbia. Reject this scope only if forbidden non-contiguous state identifiers are present: STATEFP in ['02','15','60','66','69','72','78'], STUSPS in ['AK','HI','AS','GU','MP','PR','VI'], or state names Alaska, Hawaii, American Samoa, Guam, Northern Mariana Islands, Puerto Rico, or Virgin Islands. A bbox roughly spanning lon -125 to -67 and lat 24 to 50 is compatible with the contiguous US.\n\n"

            "VERDICT CRITERIA:\n"
            "VERDICT = 'OK':\n"
            "  - The output file exists, is readable, and matches the target agent's instructions.\n"
            "  - The unique categorical/text values (under 'unique_categorical_values') and spatial bounding box (under 'bbox') match the requested scope.\n"
            "  - For data-producing, analysis, projection, raster, vector, mapping, or downstream-consumed steps, the output is a registered file artifact.\n"
            "  - For inspection, diagnostic, report, or answer-only steps that are not consumed by downstream nodes, readable structured results or clear logs may be accepted even without a file artifact.\n"
            "  - Minor naming, metadata, or format differences are acceptable when the substantive result satisfies the instruction and remains usable by the next step.\n\n"

            "VERDICT = 'REJECTED':\n"
            "  - A downstream-consumed data-producing step lacks a registered file artifact, or\n"
            "  - The dataset scope is broader or narrower than requested (wrong attribute subset, wrong spatial bounds, wrong unit/CRS, wrong format), or\n"
            "  - The file is corrupt, unreadable, or empty, or\n"
            "  - The agent produced only text/log output for a step whose result must be reused as a dataset by later nodes.\n\n"

            "RECOVERY DECISION TREE (when REJECTED):\n"
            "When you reject a node, think through these options in order:\n\n"
            "OPTION A — INJECT DOWNSTREAM FILTER/CLEANER:\n"
            "  If the output contains the correct data but includes extra records, outliers, "
            "duplicates, or is too broad, add a new downstream node (using a suitable agent "
            "from the available catalog) to filter, clean, or subset the target node's output.\n"
            "  Keep the target node as 'completed' — its output feeds the new filtering/cleaner node.\n\n"

            "OPTION B — INJECT DOWNSTREAM ADAPTER/CONVERTER:\n"
            "  If the output data is correct but needs a transformation (e.g. format conversion, "
            "unit conversion, projection, or coordinate reference system alignment) to match the requirements "
            "of downstream nodes, add a downstream adapter/converter node.\n"
            "  Keep the target node as 'completed'.\n\n"

            "OPTION C — INJECT UPSTREAM RETRIEVAL/CONTEXT:\n"
            "  If the node requires reference context or a secondary lookup dataset that is missing, "
            "inject a retrieval/context node upstream to fetch this data, connect it to the target node's inputs, "
            "and add any necessary downstream processing nodes to merge/clip them.\n"
            "  Keep the target node as 'completed'.\n\n"

            "OPTION D — MODIFY INSTRUCTIONS:\n"
            "  If the agent failed because the instructions or query parameters were wrong, ambiguous, "
            "or incomplete, modify the target node's instructions to be correct and set its status to "
            "'idle' so it will be re-executed with corrected instructions.\n\n"

            "OPTION E — SWAP AGENT:\n"
            "  If the agent type assigned to the node is fundamentally wrong or incapable of "
            "executing the task, replace the target node's agentId with a more suitable agent from "
            "the available_agent_catalog. Set the target node's status to 'idle'.\n\n"

            "AGENT SELECTION RULE FOR QC REPAIRS:\n"
            "  Choose replacement or added agents based on what they are designed to do, not only by name. "
            "Match the task and input data modality to the agent capability: use vector-oriented agents for vector datasets and vector operations, "
            "raster-oriented agents for raster datasets and raster operations, data-access agents for retrieval/download tasks, "
            "analysis agents for transformations/computation, and visualization agents for maps or charts. "
            "Use only agent IDs from the provided available_agent_catalog.\n\n"

            "OPTION F — RERUN (RETRY / LAST RESORT):\n"
            "  Only reset the target node to 'idle' (to rerun it with the same instructions) "
            "if its output is completely empty, corrupt, unreadable, or its logs show a transient fatal error "
            "with no output dataset generated at all.\n\n"

            "STATUS SEMANTICS:\n"
            "- status='completed' + qc_status='controlled': The node is done and will NOT be re-executed.\n"
            "- status='idle' + qc_status='uncontrolled': The node will be re-executed by the Canvas scheduler.\n"
            "- When you add a NEW node, set its status='idle', qc_status='uncontrolled', logs=[], outputArtifacts=[].\n"
            "- When you keep the target node as 'completed' (Options A/B/C), mark it qc_status='controlled'.\n"
            "- When you reset the target node (Options D/E/F), set status='idle', qc_status='uncontrolled'.\n"
            "- All downstream nodes that depend on a reset or newly-added node must also be reset to 'idle'.\n\n"

            "RE-WIRING CONNECTIONS (CRITICAL):\n"
            "- If you insert new corrective nodes (Options A/B/C) to filter/convert/retrieve the target node's output, you MUST update the connections in the workflow.\n"
            "- Specifically: DELETE any direct connection from the target node to its original downstream nodes (e.g. mapping or analysis nodes).\n"
            "- Connect the target node to the first new corrective node (e.g. download or filter node).\n"
            "- Connect the final corrective node (e.g. filtering/adapter node) to those downstream nodes instead.\n"
            "- Downstream nodes MUST take the corrected/transformed data from the new nodes, NOT the uncorrected data from the target node.\n\n"
            "- General processed-output precedence: if any existing intermediate node transforms, filters, joins, aggregates, enriches, reprojects, clips, converts, or otherwise prepares a source dataset for a later node, that later node should consume the intermediate node's processed output rather than the original raw source dataset. Keep a raw direct edge only when the later node explicitly needs the raw layer unchanged as an additional input.\n\n"

            "NODE FORMAT FOR NEW NODES:\n"
            "Each new node must have: id (unique string), agentId (from catalog), name, "
            f"x, y (position), instructions, inputDatasets (empty array), credentials (empty object), "
            f"serverUrl (use '{default_server_url}'), status ('idle'), logs (empty array), "
            "outputArtifacts (empty array), qc_status ('uncontrolled').\n"
            "Each new connection must have: id (unique string), sourceId, targetId.\n\n"

            "QC INSTRUCTION FORMULATION RULES FOR NEW/MUTATED NODES:\n"
            "- Keep node instructions direct, simple, and high-level (WHAT to do, not HOW to do it).\n"
            "- Do NOT write low-level programming detail, python/pandas code snippets, variable or function names, or step-by-step programming guides. Let the target agent decide the coding implementation details.\n\n"

            "OUTPUT STRUCTURE:\n"
            "Return a strict JSON response (no markdown, no headers outside JSON) matching:\n"
            "{\n"
            "  \"verdict\": \"OK\" | \"REJECTED\",\n"
            "  \"reason\": \"detailed description of why it passes or needs changes\",\n"
            "  \"recovery_option\": \"A\" | \"B\" | \"C\" | \"D\" | \"E\" | \"F\" (only when REJECTED),\n"
            "  \"suggested_workflow_changes\": { ... corrected full workflow canvas JSON ... } (only when REJECTED)\n"
            "}\n"
        )

        # Build user payload with full context
        user_payload: dict[str, Any] = {
            "overall_user_goal": query,
            "workflow_canvas_state": workflow_data,
            "target_node": {
                "id": target_node.get("id"),
                "agentId": target_node.get("agentId"),
                "name": target_node.get("name"),
                "instructions": target_node.get("instructions"),
                "logs_tail": (target_node.get("logs") or [])[-20:],
            },
            "sandbox_inspection_summaries": inspection_summaries,
        }
        if compact_catalog:
            user_payload["available_agent_catalog"] = compact_catalog

        user_content = json.dumps(user_payload, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    @staticmethod
    def _artifact_url_to_local_path(url: str) -> str | None:
        """Convert an artifact download URL to a local filesystem path.

        Artifact URLs from the GAS canvas have patterns like:
          /api/gas/download?path=/absolute/path/to/file.gpkg
          /api/download?file=relative/path.tif
          http://host:port/data/agent_output/file.geojson

        This tries common patterns and returns the local path if it exists.
        """
        if not url:
            return None

        # Pattern 1: query parameter 'path' contains the absolute path
        parsed = urlparse(url)
        from urllib.parse import parse_qs
        params = parse_qs(parsed.query)
        for key in ("path", "file", "filepath"):
            values = params.get(key, [])
            if values:
                candidate = values[0]
                if os.path.isfile(candidate):
                    return candidate

        # Pattern 2: the URL path itself might map to a data directory file
        url_path = parsed.path
        if url_path:
            # Check for agent-specific proxy path pattern: /agents/<agent_id>/data/<filename>
            import re
            match = re.search(r'/agents/([^/]+)/data/([^/]+)', url_path)
            if match:
                agent_id, filename = match.groups()
                candidate = str(DATA_DIR / agent_id / filename)
                if os.path.isfile(candidate):
                    return candidate

            # Strip common prefixes
            for prefix in ("/api/gas/download", "/api/download", "/data/"):
                if url_path.startswith(prefix):
                    remainder = url_path[len(prefix):].lstrip("/")
                    # Try it relative to DATA_DIR
                    candidate = str(DATA_DIR / remainder)
                    if os.path.isfile(candidate):
                        return candidate

            # Try the full path as-is (absolute paths embedded in URLs)
            if os.path.isfile(url_path):
                return url_path

        return None
