# Shared Repo Context Prompt

You are working inside `AI-Factory`, a platform that spans dataset generation, training, evaluation, inference, orchestration, a web dashboard, and a Rust engine named Titan.

## Mission

Make improvements that reduce friction for real users and future agents. Favor coherent systems over isolated patches.

## Read This First

- `AGENTS.md`
- `README.md`
- `docs/architecture.md`
- `docs/v1-operational-contract.md`
- `ai_factory/core/orchestration/agents.py`
- `ai_factory/core/orchestration/service.py`

## Architectural Truths

- `ai_factory/core/` is foundational and must not depend on subsystem business logic.
- `inference/app/` is the FastAPI/API surface that powers the web frontend and tests.
- `frontend/` is the Next.js product shell and should reflect real backend capabilities rather than fake demo state.
- `training/`, `evaluation/`, and `data/` should share canonical schemas and artifacts rather than invent ad hoc formats.
- `ai_factory_titan/` is the Rust systems layer and should grow toward a real local inference/runtime core.

## General Quality Bar

- Improve the typed contract, implementation, and tests together.
- Prefer removing duplicated logic over adding a third copy.
- Keep observability intact: status, telemetry, degraded modes, and failure surfaces matter.
- Treat stale docs, dead configs, and unreferenced assets as debt to be audited, not blindly deleted.

## Output Expectations

When you finish a task, summarize:

- what changed
- what was validated
- what remains risky or intentionally deferred
