"""Frozen feature profiles and orchestrator-local configuration defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gas_server.core.config import DATA_DIR


PlannerMode = Literal["generic", "gis_aware"]
QCMode = Literal["none", "generic", "gis_aware"]
ReplannerMode = Literal["none", "generic", "gis_aware"]


@dataclass(frozen=True)
class OrchestratorFeatures:
    """Capabilities supplied to the runtime instead of raw condition IDs."""

    profile_name: str
    planner_mode: PlannerMode
    qc_mode: QCMode
    replanner_mode: ReplannerMode
    technical_invariants: bool = True
    local_recovery: bool = True
    structural_patching: bool = True
    gis_finding_routing: bool = True


_FEATURE_PROFILES: dict[str, OrchestratorFeatures] = {
    "C1": OrchestratorFeatures(
        profile_name="C1",
        planner_mode="generic",
        qc_mode="generic",
        replanner_mode="generic",
        gis_finding_routing=False,
    ),
    "C2": OrchestratorFeatures(
        profile_name="C2",
        planner_mode="generic",
        qc_mode="gis_aware",
        replanner_mode="gis_aware",
    ),
    "C3": OrchestratorFeatures(
        profile_name="C3",
        planner_mode="gis_aware",
        qc_mode="gis_aware",
        replanner_mode="none",
        structural_patching=False,
    ),
    "C4": OrchestratorFeatures(
        profile_name="C4",
        planner_mode="gis_aware",
        qc_mode="none",
        replanner_mode="gis_aware",
        gis_finding_routing=False,
    ),
    "C5": OrchestratorFeatures(
        profile_name="C5",
        planner_mode="gis_aware",
        qc_mode="gis_aware",
        replanner_mode="gis_aware",
    ),
}


def features_for_condition(condition: str) -> OrchestratorFeatures:
    """Return a frozen feature profile and reject unknown conditions."""

    try:
        return _FEATURE_PROFILES[condition.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown orchestrator condition: {condition!r}") from exc


DEFAULT_PUBLIC_MODE = "plan"
STATE_SCHEMA_VERSION = "1.0.0"
STORE_PATH = Path(
    os.getenv(
        "GAS_ORCHESTRATOR_STORE_PATH",
        str(DATA_DIR / "orchestrator" / "workflows.sqlite3"),
    )
)
STORE_RETENTION_DAYS = max(1, int(os.getenv("GAS_ORCHESTRATOR_RETENTION_DAYS", "30")))
EXECUTION_LEASE_SECONDS = max(5, int(os.getenv("GAS_ORCHESTRATOR_LEASE_SECONDS", "120")))
