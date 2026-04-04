# Orchestration And Agents Upgrade Prompt

Upgrade AI-Factory's orchestration and agent runtime so task routing, retries, circuit breakers, telemetry, and lineage behave like a reliable control plane instead of best-effort glue.

## Primary Goal

Strengthen the system that coordinates work across data, training, evaluation, monitoring, deployment, and optimization agents.

## Read This First

- `prompts/shared-repo-context.md`
- `AGENTS.md`
- `ai_factory/core/orchestration/agents.py`
- `ai_factory/core/orchestration/models.py`
- `ai_factory/core/orchestration/service.py`
- `ai_factory/core/orchestration/sqlite.py`
- `ai_factory/orchestration/distributed.py`
- `ai_factory/core/lineage/`
- `tests/test_orchestration_foundation.py`
- `tests/test_distributed_orchestration.py`
- `tests/test_orchestration_api.py`

## Scope

- Agent registration and capability declarations
- Retry policy behavior, backoff, dead-letter handling, and circuit states
- Event emission and lineage completeness
- Monitoring summaries, stalled-task recovery, and task leasing
- UI/API projections of orchestration health

## Definition Of Done

- Agent lifecycle behavior is explicit and testable.
- Failure and recovery paths are first-class, not accidental.
- Lineage and monitoring outputs help operators understand what happened.
