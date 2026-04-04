# AI-Factory Specialized Agent Prompts

This folder contains the highly specific, high-signal context prompts designed to guide autonomous agents in upgrading different subsystems of the **AI-Factory Platform**. By providing narrow, domain-focused starting context, agents avoid getting lost in the broader repository and can immediately begin surgical, high-quality execution.

## Agent Workflow (How To Use)

When assigned a task, you MUST adhere to the following sequence:

1. **Inhale Foundation:** Start by reading `shared-repo-context.md`. This establishes the non-negotiable architectural boundaries, typing constraints, and operational laws of the entire platform.
2. **Select Specialization:** Pick the single most specific prompt from the list below that aligns with your task. Do NOT read all prompts unless your task spans the entire platform.
3. **Read The Target Manifest:** Open the chosen prompt and rigorously read every file listed in the `Read This First (Mandatory Ingestion)` section. This defines your immediate operational context.
4. **Execute & Expand:** Begin execution. Expand your scope outward *only* if the target files are insufficient to complete the task.
5. **Verify:** Ensure your work meets the explicit `Definition Of Done` defined in the prompt.

## Prompt Index

- `shared-repo-context.md`: The absolute source of truth for global architecture, typed boundaries, agent workflows, and the overall quality bar.
- `web-dashboard-upgrade.md`: For upgrading the Next.js frontend, wiring UI components to real backend APIs, enforcing Zod validation, and handling UI degradation gracefully.
- `backend-platform-upgrade.md`: For expanding the FastAPI platform, tightening Pydantic V2 schemas, enforcing strict routing, dependency injection, and observable circuit breakers.
- `engine-training-upgrade.md`: For overhauling the ML training loops, QLoRA workflows, hardware optimizations, and weight packaging (bridging to Titan).
- `docs-evals-datasets-upgrade.md`: For curating platform documentation, enforcing dataset lineage/hashing, and hardening deterministic LLM evaluation pipelines.
- `orchestration-agents-upgrade.md`: For building the robust, asynchronous control-plane that manages retries, state machines, dead letter queues, and distributed agent telemetry.
- `aifactory-titan.md`: For evolving the Rust-based Titan engine into a serious local inference core with explicit quantization layouts and C++ kernel bridging.

## Universal Directives

- **Preserve the `ai_factory.core` boundary:** Core code must stay foundational and subsystem-agnostic.
- **Atomic Quality:** Prefer changes that improve typed contracts, test coverage, and system observability simultaneously. Do not commit untested logic.
- **Responsible Pruning:** Do not blindly delete artifacts, datasets, or notebooks. Only remove stale files after rigorous dependency analysis.
- **When in Doubt:** Consult `AGENTS.md`, `README.md`, and `docs/architecture.md` first.