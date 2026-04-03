# System Architecture

AI-Factory is organized as a shared-core platform rather than a loose stack of scripts. The same canonical schema, answer utilities, artifact layout, and metadata conventions are reused by data preparation, training, inference, evaluation, notebooks, and the frontend.

## New Core Modules (v2)

### `ai_factory/core/async_utils.py`
Async utilities for managing concurrent operations, rate limiting, and backpressure handling.

### `ai_factory/core/cache/`
Memory-backed caching layer for frequent lookups (dataset records, model weights, tokenizer states).

### `ai_factory/core/distributed/`
Distributed compute primitives: worker registration, job dispatch, result aggregation across cluster nodes.

### `ai_factory/core/security/`
Security abstractions: credential rotation, encrypted artifact storage, access control lists.

## Updated Modules

### `ai_factory/core/orchestration/service.py`
Enhanced orchestration service with:
- Circuit breaker patterns for fault tolerance
- Lease-based task ownership
- Retry policies with exponential backoff
- Task dependency resolution
- Real-time heartbeat monitoring

### `ai_factory/platform/monitoring/metrics.py`
Improved metrics collection with:
- System-level metrics (CPU, memory, disk, GPU)
- Training job metrics (loss, accuracy, job counts)
- Inference metrics (latency, throughput, success rate)
- Historical queries via time-series database

## Core vs. Subsystem Boundaries

The `ai_factory.core` layer is strictly foundational:
- No business logic
- No imports from `data`, `training`, `evaluation`, `inference`, `agents` subsystems
- All cross-subsystem communication flows through `FastAPI` or shared protocols

This boundary is enforced by CI checks.

## Design Principles

- Quality over scale: specialize a tractable model for calculus and olympiad reasoning rather than chasing generic coverage.
- Research velocity: keep every stage config-driven, reproducible, and introspectable through manifests and summaries.
- Product realism: the backend API, benchmark suite, and frontend are part of the system design, not a thin wrapper around a notebook.
- Extensibility: new datasets, model variants, prompts, benchmark slices, and notebook workflows should plug into the same scaffolding.

## Shared Core

The `ai_factory/core/` package provides:

- canonical `v2` schemas for dataset records, manifests, and run outputs
- stable hashing and question fingerprinting
- artifact directory helpers for `artifacts/runs/<run_id>/...` and `artifacts/models/<model_name>/...`
- answer extraction, equivalence, verifier, and reranking helpers
- JSON/JSONL/Markdown report writing and token utility helpers

This keeps logic such as final-answer parsing, `step_checks`, and run manifest structure consistent across subsystems.

## Data Architecture

The data layer is split into focused modules:

- `data/synthesis/`: custom synthetic dataset families and generation registry
- `data/adapters/`: public dataset registry definitions and normalization interfaces
- `data/quality/`: difficulty estimation, quality scoring, contamination checks, dedupe, mining
- `data/builders/`: corpus assembly, pack derivation, manifests, cards, size reports
- `data/tools/`: validation, preview, audit, export, and benchmark-pack utilities

Canonical records now include:

```json
{
  "schema_version": "v2",
  "id": "stable-question-fingerprint",
  "question": "...",
  "solution": "...",
  "final_answer": "...",
  "difficulty": "hard",
  "topic": "calculus",
  "subtopic": "integration by parts",
  "source": "custom_integral_arena",
  "dataset_split": "train",
  "step_checks": [{"kind": "substring", "value": "u = x^2"}],
  "tags": ["integrals", "verification"],
  "failure_case": false,
  "quality_score": 0.92,
  "reasoning_style": "chain_of_thought",
  "pack_id": "core_train_mix",
  "contamination": {"exact_match": false, "near_match": false},
  "lineage": {"dataset_id": "...", "dataset_family": "...", "loader": "..."},
  "generator": {"generator_family": "..."},
  "metadata": {}
}
```

The processed build emits train/eval/test splits, a manifest, cards, size reports, and derived packs for hard calculus, olympiad reasoning, verification-rich examples, failure replay, and held-out benchmarking.

## Training Architecture

Training is composed from reusable config parts:

- model config
- adapter config
- data config
- runtime config
- logging config
- packaging config

Profiles such as `baseline_qlora` or `verifier_augmented` are thin compositions over these components. The trainer stack supports LoRA/QLoRA by default, optional full-precision exports, difficulty-aware sample weights, curriculum ordering, failure-case replay, checkpoint policies, and packaged serving artifacts.

## Orchestration Architecture

The control plane sits alongside the subsystem-specific stacks rather than replacing them. A SQLite local-first control plane under `artifacts/control_plane/` is the durable source of truth for orchestration runs, tasks, attempts, events, leases, and circuit state, while the existing JSON/JSONL artifact outputs remain the canonical subsystem outputs.

Orchestration configs under `configs/*.yaml` describe:

- managed instance type: `prepare`, `train`, `finetune`, `evaluate`, `report`, `inference`, or `deploy`
- user experience level: `beginner`, `hobbyist`, or `dev`
- orchestration mode: single-node, local-parallel, cloud-parallel, or hybrid
- remote-access metadata: cloud profile resolution, SSH settings, and port-forward definitions
- sub-agent policy: bounded parallel follow-up workloads such as preprocessing, evaluation, metrics/reporting, or publish steps
- feedback loop policy: when to recommend or automatically queue the next training or deployment action
- resilience policy: retry/backoff, circuit-breaker thresholds, dead-letter behavior, and checkpoint-resume defaults
- alert policy: telemetry sink and anomaly detection toggles

Each managed instance now projects onto:

- an orchestration run
- one or more orchestration tasks
- one or more task attempts with heartbeats, leases, stdout/stderr refs, and checkpoint hints

The existing `artifacts/instances/<instance-id>/...` layout is preserved as a compatibility projection so the CLI, FastAPI instance routes, and frontend views stay thin while the runtime becomes more durable.

The built-in agent registry covers:

- data processing
- training orchestration
- evaluation and benchmarking
- monitoring and telemetry
- optimization and feedback loops
- deployment

The default orchestration templates include reusable `prepare` and `report` jobs so evaluation misses can flow directly into failure-analysis artifacts and the next training loop.

## Inference Architecture

The FastAPI layer is organized around:

- routers for health, metadata, and generation
- services for generation and metadata
- a YAML-backed model registry with lazy loading
- prompt preset and solver-mode selection
- candidate sampling, answer extraction, verification, reranking, and structured response shaping
- local cache and JSONL telemetry hooks

Versioned `/v1/*` routes coexist with compatibility aliases so the frontend and tests can adopt the richer API without breaking older scripts.

## Evaluation Architecture

Evaluation uses an explicit benchmark registry instead of depending only on the training eval split. Every run can compare two model variants or two run outputs and writes machine-readable plus human-readable artifacts:

- `summary.json`
- `summary.md`
- `leaderboard.json`
- `per_example.jsonl`

Metrics include final-answer accuracy, parse rate, step correctness, verifier agreement, formatting failure, arithmetic-slip detection, no-answer rate, candidate agreement, latency, and approximate token/cost reporting when tokenization metadata is available.

## Frontend Architecture

The frontend is a multi-route research product:

- `/`: solve workspace
- `/compare`: model-vs-model comparison lab
- `/datasets`: dataset and pack browser
- `/benchmarks`: benchmark registry explorer
- `/runs`: training/evaluation run viewer

The solve view mirrors the backend capabilities by surfacing model selection, prompt presets, solver modes, reasoning visibility, structured output, sample count, verification state, and candidate inspection.

## End-to-End Flow

1. Build or normalize datasets into canonical `v2` records.
2. Construct processed splits and derived packs with manifests and cards.
3. Launch raw subsystem commands or managed orchestration instances for training and packaging.
4. The async control plane dispatches tasks, records attempts, heartbeats, retries, and structured events, and projects status back into instance manifests.
5. Serve models through the inference API and frontend.
6. Evaluate against benchmark packs, mine failures, and let the control plane recommend or queue the next step.

For the concrete V1 route, degraded-mode, and artifact contract that the web and desktop shells consume, see [V1 Operational Contract](v1-operational-contract.md).
