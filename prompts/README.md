# AI-Factory Specialized Agent Prompts

This folder gives specialized agents a narrow, high-signal starting context so they do not need to manually inspect the entire repository before making progress.

## How To Use

1. Start with `shared-repo-context.md`.
2. Pick the most specific prompt for the task.
3. Read only the files listed in `Read This First` before editing.
4. Expand outward only if the prompt's target files are insufficient.

## Prompt Index

- `shared-repo-context.md`: global architecture, boundaries, workflow, and quality bar.
- `web-dashboard-upgrade.md`: upgrade the Next.js dashboard and route UX; every action should feel polished and wired.
- `backend-platform-upgrade.md`: expand and harden the FastAPI/backend platform so it matches frontend needs cleanly.
- `engine-training-upgrade.md`: upgrade training, engine-adjacent scripts, and performance-sensitive ML workflows.
- `docs-evals-datasets-upgrade.md`: tighten docs, evaluation assets, datasets, and research hygiene.
- `orchestration-agents-upgrade.md`: improve orchestration, control-plane behaviors, retries, circuit breakers, and agent reliability.
- `aifactory-titan.md`: focused Titan prompt for making the Rust engine more `llama.cpp`-like and integrating C++ acceleration safely.

## Conventions

- Preserve the `ai_factory.core` boundary. Core code must stay foundational and subsystem-agnostic.
- Prefer changes that improve typed contracts, tests, and observability together.
- Do not delete artifacts, datasets, notebooks, or docs without checking whether code, docs, or tests still reference them.
- When in doubt, follow `AGENTS.md`, `README.md`, and `docs/architecture.md` first.
