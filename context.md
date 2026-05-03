# AI-Factory — Agent & Contributor Context Playbook

This document is the **canonical orientation layer** for anyone (human or automated agent) working in this monorepo. It synthesizes structure, contracts, operational flows, quality gates, and where to read more. For step-by-step commands, prefer **[quickstart.md](quickstart.md)**; for architecture narrative, **[docs/architecture.md](docs/architecture.md)**.

---

## 1. What this repository is

**AI-Factory** is a unified platform for the **full LLM lifecycle**: synthetic and curated data, training (including scratch and adapter paths), evaluation, inference API, web UI, optional desktop shell, and a **local-first orchestration control plane** (SQLite) for runs, tasks, and telemetry.

Design goals emphasized across docs:

- **Shared contracts**: canonical dataset schemas, manifests, artifact layouts, and reporting formats reused across data, training, evaluation, inference, and frontend.
- **Config-driven behavior**: prefer YAML profiles and registries over one-off hardcoding.
- **Subsystem independence**: training, evaluation, and inference remain separable deployables; shared logic lives in `ai_factory` (especially `ai_factory.core`), not in cross-imports between top-level trees.
- **Research + product**: API, benchmarks, and dashboard are part of the system, not an afterthought.

---

## 2. Top-level map

| Path | Role |
|------|------|
| **[ai_factory/](ai_factory/)** | Installable Python package: **core** primitives, orchestration, platform helpers, interfaces (CLI/TUI/web/desktop), domains. |
| **[data/](data/)** | Dataset generation, public adapters, processing, quality, tools, processed outputs. |
| **[training/](training/)** | Training entrypoints, configs, scripts, trainer stack. |
| **[evaluation/](evaluation/)** | Benchmarks, evaluation configs, result artifacts. |
| **[inference/](inference/)** | FastAPI app: routers, services, configs (model registry, prompts). |
| **[frontend/](frontend/)** | Next.js dashboard; typed API surface aligned with OpenAPI. |
| **[desktop/](desktop/)** | Electron wrapper for local dashboard usage. |
| **[ai_factory_titan/](ai_factory_titan/)** | Rust/C++ performance core (SIMD, quantization paths, telemetry bridge). |
| **[scripts/](scripts/)** | Bootstrap, doctor, OpenAPI export, and other automation. |
| **[tests/](tests/)** | Pytest suite including **architecture boundary** tests. |
| **[docs/](docs/)** | Deep guides: architecture, data, training, evaluation, deployment, API, troubleshooting. |
| **[configs/](configs/)** | Orchestration and high-level YAML (e.g. finetune, deploy). |
| **[prompts/](prompts/)** | Internal upgrade/spec prompts for tooling (not runtime prompts for models). |

**Entry CLI**: `ai-factory` → `ai_factory.cli:main` (see `[project.scripts]` in **[pyproject.toml](pyproject.toml)**).

---

## 3. Subsystem boundaries (critical for agents)

### 3.1 Dependency matrix

Authoritative diagram and rules: **[docs/architecture/dependency_matrix.md](docs/architecture/dependency_matrix.md)**.

**Rules of thumb:**

- **`ai_factory.core` must not import** `data`, `training`, `evaluation`, or `inference`. This is enforced by **`tests/test_architecture_boundaries.py`**.
- **Sibling subsystems must not import each other** (e.g. `training` → `inference` is forbidden). Shared behavior belongs in **`ai_factory.core`** (or another `ai_factory` package that respects the same rules), exposed via stable facades.
- **`evaluation` may depend on `training` only in the sense of the matrix** — the documented allowed flow is: data and training feed evaluation; **inference sits downstream**. The test file defines **forbidden edges** explicitly; keep it and `dependency_matrix.md` aligned when changing imports.

When you need cross-cutting math/generation or hardware utilities:

- **`ai_factory.core.math_stack`** — model catalog, loading, prompts, generation logic shared with inference (inference modules may re-export as thin facades).
- **`ai_factory.core.runtime`** — hardware detection and related PyTorch-oriented helpers factored out of training for boundary-safe reuse.

### 3.2 Why this matters

- Keeps GPU training stacks out of the API server and vice versa.
- Makes CI boundary tests meaningful: **import graph drift fails the build**.

---

## 4. The `ai_factory` package (where “platform” lives)

High-level layout (not exhaustive):

| Area | Purpose |
|------|---------|
| **`ai_factory.core`** | Schemas, hashing, artifacts, orchestration service, math_stack, runtime, execution helpers, security/cache/distributed primitives as documented in architecture docs. |
| **`ai_factory.platform`** | Deployment, monitoring, scaling helpers. |
| **`ai_factory.interfaces`** | CLI, TUI, web, desktop abstractions. |
| **`ai_factory.domains`** | Domain-specific logic (e.g. mathematics). |

**Orchestration** (agents, tasks, circuit breakers, retries) is described for humans in **[AGENTS.md](AGENTS.md)** and implemented under **`ai_factory.core.orchestration`**. Monitoring summaries and health-related behaviors are part of operating the control plane alongside FastAPI.

---

## 5. Versioning (single source of truth)

- **`[project].version`** in **[pyproject.toml](pyproject.toml)** and **`VERSION`** in **[ai_factory/version.py](ai_factory/version.py)** must stay aligned.
- **`ai_factory.__version__`** re-exports from **`ai_factory.version`**.
- Inference settings and health endpoints consume **`VERSION`** (and may honor **`AI_FACTORY_API_VERSION`** where configured — see inference app config).

Regression coverage: **`tests/test_version_consistency.py`**.

---

## 6. Inference API and runtime

- **App**: **`inference/app/main.py`** — FastAPI application; uses platform container wiring for orchestration warm-up (avoid ad-hoc `OrchestrationService()` construction).
- **Configs**: **`inference/configs/`** — `model_registry.yaml`, prompt presets, etc.
- **Operational contract** (routes, demo mode, degraded behavior): **[docs/v1-operational-contract.md](docs/v1-operational-contract.md)**.

**Demo / production**: gated by env vars such as `AI_FACTORY_DEMO_MODE`, `NEXT_PUBLIC_AI_FACTORY_DEMO_MODE` (see operational contract).

---

## 7. Frontend and API contracts

- **Stack**: Next.js (see **[frontend/package.json](frontend/package.json)**), TypeScript, Tailwind.
- **Contract-first workflow**: OpenAPI JSON is generated into **`frontend/lib/api/generated/openapi.json`** and TypeScript types into **`schema.d.ts`**, via:
  - `npm run api:export` → runs **`scripts/export_openapi.py`**
  - `npm run api:codegen` → `openapi-typescript`
  - `npm run api:sync` → both
  - `npm run api:check` → regenerate and **`git diff --exit-code`** on generated files (CI uses this pattern).

Agents changing FastAPI models or routes should **regenerate and commit** generated artifacts when APIs change, or CI will fail.

---

## 8. Data layer and artifacts

### 8.1 Data pipeline (typical)

1. **Generate** synthetic families (e.g. calculus generators under **`data/generator/`**, config-driven).
2. **Optional**: normalize public datasets per **`data/public/registry.yaml`**.
3. **Prepare** corpus: **`data/prepare_dataset.py`** with **`data/configs/processing.yaml`** — produces JSONL splits, manifests, packs, and commonly **`data/processed/corpus.sqlite`**.
4. **Validate / preview** with **`data/tools/`**.

**Important distinction**:

- **`data/processed/corpus.sqlite`** — durable **training corpus** store (distinct from orchestration DB).
- **`artifacts/control_plane/`** — orchestration **control plane** SQLite (runs, tasks, events).

### 8.2 Training artifacts

- Runs: **`artifacts/runs/<run_id>/`**
- Published models: **`artifacts/models/<name>/`**
- Tokenizers and foundation checkpoints: typically under **`artifacts/tokenizers/`**, **`artifacts/foundation/`** (paths vary by profile; see quickstart).

### 8.3 Evaluation outputs

- Under **`evaluation/results/<run_name>/`**: `summary.json`, `summary.md`, `leaderboard.json`, `per_example.jsonl`, etc.

---

## 9. Training and evaluation — operator mental model

**Authoritative step-by-step**: **[quickstart.md](quickstart.md)**.

Compressed checklist:

1. **Bootstrap**: **`scripts/start-linux.sh`** (cloud GPU) or **`scripts/start-mac.sh`** (Apple Silicon).
2. **Health**: `ai-factory ready`, **`python scripts/doctor.py`**, optional `docker compose config`.
3. **Data**: generate → prepare → validate.
4. **Training mode**: scratch vs continued pretrain vs adapter/full fine-tune — pick **`training/configs/profiles/*.yaml`** accordingly.
5. **Dry-run first**: `python -m training.train --config ... --dry-run` (and **`ai-factory train-preflight`** for hard checks).
6. **Evaluate**: smoke configs vs **`evaluation/configs/base_vs_finetuned.yaml`** after the **`failure_aware`** profile has published expected artifacts (see quickstart).
7. **Serve**: `ai-factory serve` + `ai-factory api-smoke`.

**Profiles** worth knowing by name: `pretraining.yaml`, `baseline_qlora.yaml`, **`failure_aware.yaml`** (feeds default finetuned registry expectations), `math_specialist.yaml`, `fast_dev.yaml`, ultimate CUDA/Metal profiles — details and hardware assumptions in **quickstart** and **specs.md**.

**Specs / hardware commands**: **[specs.md](specs.md)** (long operator guide: hardware inspection, command matrix).

---

## 10. Orchestration and control plane

- SQLite-backed **control plane** coexists with JSON/JSONL subsystem artifacts; configs under **`configs/*.yaml`** describe managed instances and policies (see **architecture.md**).
- **AGENTS.md** summarizes agent types, retry policies, circuit breakers, and monitoring hooks (`OrchestrationService.monitoring_summary()`).

---

## 11. Titan (Rust) engine

**`ai_factory_titan/`** provides native acceleration and telemetry; Python may bridge for metrics (see architecture notes on Titan telemetry and monitoring). **Makefile** targets: `rust-check`, `rust-test`, `rust-clippy`, `rust-bench`, `rust-build`.

---

## 12. Visualization and reporting

- **Unified script**: **`generate_visualizations.py`** at repo root — modes include aggregate analytics, single-run dashboards, evaluation comparisons, and `all` (default). Depends on training run loader under **`graph_generation/`** and writes under **`evaluation/results/visualizations/`** by default.
- Prior standalone visualization scripts were consolidated here; extend this entrypoint rather than scattering new generators.

---

## 13. Quality gates and CI

### 13.1 Local developer commands

From **README**, **Makefile**, and **contributor guide**:

| Concern | Typical command |
|---------|------------------|
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Types | `mypy .` (see overrides below) |
| Tests | `pytest`, or `make test` / `make test-parallel` |
| Frontend | `cd frontend && npm run typecheck` (and `npm run build` for full check) |

**Makefile** aggregates install, lint, test, serve, data, training, evaluation, frontend, docker, and rust targets — prefer it when unsure.

### 13.2 Mypy strategy

**[pyproject.toml](pyproject.toml)** enables strict typing for `ai_factory` but **ignores errors** for `data`, `training`, `evaluation`, `inference`, and some interfaces — incremental tightening is intentional. CI includes a **stricter spot-check**: `mypy ai_factory/core/platform`.

### 13.3 CI (GitHub Actions)

**[.github/workflows/ci.yml](.github/workflows/ci.yml)** (summary):

- Python **3.11 and 3.12**: ruff format check, ruff lint, mypy, pytest with coverage, Bandit (non-blocking upload), frontend install + **`npm run api:check`**, etc.

### 13.4 Pre-commit

**[.pre-commit-config.yaml](.pre-commit-config.yaml)** — includes hygiene hooks (e.g. large files, case conflict). Run `pre-commit install` after clone when working locally.

---

## 14. Environment variables (non-exhaustive)

| Variable | Role |
|----------|------|
| `AI_FACTORY_REPO_ROOT` | Repo root for tooling and paths. |
| `ARTIFACTS_DIR` | Writable artifacts base (especially cloud). |
| `CORS_ORIGINS` | API CORS (e.g. frontend origin). |
| `AI_FACTORY_API_VERSION` | Optional override for API-reported version. |
| `AI_FACTORY_DEMO_MODE` / `NEXT_PUBLIC_AI_FACTORY_DEMO_MODE` | Demo vs production behavior (see v1 operational contract). |

---

## 15. Documentation index (what to open when)

| Need | Document |
|------|----------|
| **First hands-on path** | **[quickstart.md](quickstart.md)** |
| **System design** | **[docs/architecture.md](docs/architecture.md)** |
| **Dependency rules** | **[docs/architecture/dependency_matrix.md](docs/architecture/dependency_matrix.md)** |
| **Data system** | **[docs/data-system.md](docs/data-system.md)** |
| **Training** | **[docs/training-system.md](docs/training-system.md)**, **[training/README.md](training/README.md)** |
| **Evaluation** | **[docs/evaluation-system.md](docs/evaluation-system.md)** |
| **Inference** | **[docs/inference-system.md](docs/inference-system.md)**, **[inference/README.md](inference/README.md)** |
| **Deployment** | **[docs/deployment-guide.md](docs/deployment-guide.md)** |
| **API** | **[docs/api/README.md](docs/api/README.md)**, **[docs/api-guide.md](docs/api-guide.md)**, **[docs/api/complete_reference.md](docs/api/complete_reference.md)** |
| **Operations / recovery** | **[docs/runbook.md](docs/runbook.md)**, **[docs/troubleshooting.md](docs/troubleshooting.md)** |
| **Contributing** | **[docs/contributor-guide.md](docs/contributor-guide.md)** |
| **Agents / orchestration** | **[AGENTS.md](AGENTS.md)** |
| **Short platform overview** | **[GEMINI.md](GEMINI.md)** |
| **Changelog** | **[CHANGELOG.md](CHANGELOG.md)** |
| **Command matrix / hardware** | **[specs.md](specs.md)** |
| **Notebook workflow** | **[docs/notebook-guide.md](docs/notebook-guide.md)**, **[notebooks/README.md](notebooks/README.md)** |

### 15.1 `MEMORY.md`

**[MEMORY.md](MEMORY.md)** is a **pointer index** to a few key files (and mentions `reports/audit-final.md`). Use it as a quick jump list, not as a second source of truth.

### 15.2 Known doc drift

**GEMINI.md** links to **`tune-guide.md`** and **`docs/cli-guide.md`**, which **are not present** in the repo as of this writing. Prefer **quickstart.md**, **specs.md**, and **docs/** for those topics.

---

## 16. Prompts and internal specs

**[prompts/](prompts/)** contains structured upgrade prompts and shared context for automation (e.g. platform, web, docs). These are **maintainer aids**, not user-facing product docs.

---

## 17. Agent workflow — practical rules

1. **Respect boundaries** before adding imports; run **`pytest -q tests/test_architecture_boundaries.py`** when touching cross-package code.
2. **Prefer `ai_factory.core`** for shared types and algorithms; avoid duplicating inference-only helpers inside training or vice versa.
3. **Keep APIs and frontend types in sync** via OpenAPI generation when changing backend schemas.
4. **Use dry-runs and preflight** before long GPU jobs.
5. **Artifacts are part of the contract** — preserve manifest/summary layouts when extending pipelines.
6. **Do not treat `evaluation/results/` or `data/processed/` as disposable** without understanding what your change deletes; follow repo hygiene rules in **`.gitignore`** and team practices.

---

## 18. One-line mission statement

**AI-Factory** is a **config-driven, contract-first ML platform** for building, training, evaluating, and serving math/reasoning-oriented models, with a **strict core boundary**, a **durable local control plane**, and **product-quality** API and web surfaces — validated by **automated architecture, version, and schema checks** in CI.
