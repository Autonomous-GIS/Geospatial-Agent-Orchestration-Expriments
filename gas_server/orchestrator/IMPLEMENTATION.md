# Orchestrator implementation baseline

This package implements the frozen specification in
`CONSOLIDATED_GIS_ORCHESTRATOR_REFACTOR_PLAN.md` version 1.0.0.

Implementation rules:

- Do not modify `GeoAgent` or its lifecycle.
- Preserve public planning behavior until the explicit mode adapter is added.
- Keep one production execution loop in `core/controller.py`.
- Keep benchmark and oracle imports outside this package.
- Resolve C1-C5 behavior only through `config.py` feature profiles.
- Treat canvas JSON as a projection, not authoritative state.
- Advance only after the tests for the current phase gate pass.

Initial configuration defaults:

- Store: SQLite at `Data/orchestrator/workflows.sqlite3`, overridable with
  `GAS_ORCHESTRATOR_STORE_PATH`.
- Retention metadata: 30 days, with deletion invoked explicitly outside workflow
  execution.
- Public default mode: `plan`.
- Development transport: local GAS `ServiceRegistry`.
- State schema version: 1.0.0.

Model/provider settings, evaluation budgets, prompt hashes, capability health
policy, and the final development fixture freeze remain configuration decisions
to be fixed before their corresponding phases.
