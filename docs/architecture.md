# System Architecture

Atlas Math Lab is organized as a shared-core platform rather than a loose stack of scripts. The same canonical schema, answer utilities, artifact layout, and metadata conventions are reused by data preparation, training, inference, evaluation, notebooks, and the frontend.

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
3. Train an adapter or specialist profile and package artifacts.
4. Serve models through the inference API and frontend.
5. Evaluate against benchmark packs, mine failures, and feed them back into the next training run.
