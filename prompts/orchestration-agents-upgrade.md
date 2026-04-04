# Orchestration And Agents Upgrade Prompt

Upgrade AI-Factory's orchestration and agent control plane. Your objective is to transition task routing, event emission, agent orchestration, and lineage tracking from a basic synchronous workflow to an asynchronous, highly-resilient, production-ready system.

## Primary Goal

Build a robust control plane that reliably coordinates multi-stage distributed work across dataset synthesis, model training, evaluation runs, and continuous monitoring agents. Assume that networks will partition and worker nodes will crash.

## Read This First (Mandatory Ingestion)

- `prompts/shared-repo-context.md`
- `AGENTS.md` (Agent system laws)
- `ai_factory/core/orchestration/agents.py` (Agent interface definitions)
- `ai_factory/core/orchestration/models.py` (Orchestration schemas)
- `ai_factory/core/orchestration/service.py` (Core control logic)
- `ai_factory/core/orchestration/sqlite.py` (State persistence layer)
- `ai_factory/orchestration/distributed.py` (Multi-node dispatcher)
- `ai_factory/core/lineage/` (Data provenance tracking)
- `tests/test_orchestration_foundation.py`
- `tests/test_distributed_orchestration.py`
- `tests/test_orchestration_api.py`

## Scope & Execution Directives

1. **Resiliency & Retry Semantics**:
   - Implement exponential backoff for external API calls and task execution.
   - Establish formal "Circuit Breaker" states for failing sub-systems.
   - Define "Dead Letter Queues" for irreparably failed tasks so they can be inspected later via the API.
2. **State Machines & Idempotency**:
   - Agents and Orchestration Tasks must operate as strict state machines (Pending -> Running -> Completed/Failed).
   - Ensure every mutation action is idempotent to prevent duplicate processing on retry.
3. **Telemetry & Lineage Precision**:
   - Every state transition must emit a structured event that is persisted to the database.
   - Ensure data lineage correctly links source datasets, applied configurations, resulting weights, and final evaluation scores.
4. **Stalled Task Recovery**:
   - Implement mechanisms (e.g., heartbeat leasing) to detect when a worker node crashes mid-task and safely re-queue the work.

## Definition Of Done

- The agent execution lifecycle is completely deterministic, explicitly modeled, and fully testable.
- Failure scenarios (network drops, worker crashes) result in clean recovery or stateful failure tracking, not system lockup.
- Telemetry events and data lineage provide an unambiguous audit trail of "What happened, when, and why".
- The frontend orchestration views are fully powered by these reliable control-plane APIs.