# Backend Platform Upgrade Prompt

Expand, harden, and optimize the AI-Factory Python backend. Your goal is to construct a FastAPI and service layer that is resilient, strictly typed, highly observable, and engineered to support partial failures gracefully. 

## Primary Goal

Make the backend feel deliberate and rock-solid. Ensure well-typed Pydantic contracts, intelligent route structures, observable health metrics, and seamless consumption by both the Next.js frontend and autonomous agents.

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `inference/app/main.py` (App lifecycle and middleware)
- `inference/app/config.py` (Environment and singleton management)
- `inference/app/dependencies.py` (Dependency injection graph)
- `inference/app/services/` (Business logic layer)
- `inference/app/routers/` (HTTP controller layer)
- `ai_factory/titan.py` (Rust engine bridge)
- `ai_factory/core/orchestration/service.py` (Control plane)
- `tests/test_*api*.py` (API Contract tests)

## Scope & Execution Directives

1. **Strict Routing & Dependency Injection**:
   - Eliminate duplicate router registrations.
   - Refactor hardcoded dependencies to use `fastapi.Depends()` strictly, enabling seamless mocking during tests.
2. **Service Boundaries & Pydantic Validation**:
   - Separate HTTP parsing (routers) from business logic (services).
   - Upgrade schemas to Pydantic V2 standards. Ensure `ConfigDict(strict=True)` is used where appropriate to prevent silent data coercion.
3. **Observability & Mission Control**:
   - Tightly integrate Titan telemetry, active orchestration runs, and instance metrics into a unified `/health` or `/metrics` (Prometheus) endpoint.
4. **Graceful Degradation & Circuit Breaking**:
   - If Titan is unavailable, the API must not crash. It should return a `503 Service Unavailable` with a structured error payload detailing the failure.
   - Implement explicit error surfaces that help the frontend display "Degraded Mode" UI states.
5. **Async Safety**:
   - Ensure all I/O bound operations (DB, HTTP clients) are properly awaited. Avoid blocking the main asyncio event loop.

## High-Value Targets

- Consolidate Titan and mission-control payload consistency across all endpoints.
- Eradicate schema mismatches between `inference/app/schemas.py` and `frontend/lib/api.ts`.
- Standardize fault reporting structure (e.g., standardizing on RFC 7807 Problem Details for HTTP APIs).

## Definition Of Done

- The backend routing tree is pristine, with zero accidental duplication or alias drift.
- `mypy --strict` passes completely on `inference/app`.
- Broken external dependencies result in explicit 503s or degraded states, NEVER 500 Internal Server Errors.
- Complete API test coverage for failure modes, not just happy paths.