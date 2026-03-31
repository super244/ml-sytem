# AI-Factory

## Project Overview

AI-Factory is a **Unified Autonomous AI Operating System** — a production-grade, enterprise-scale platform for the complete lifecycle governance of large language models. It goes beyond conventional MLOps tooling to deliver a self-operating research laboratory: one where data is synthesized, models are trained, experiments are orchestrated by intelligent agents, and inference is deployed — all within a single, coherent, lineage-tracked system.

The platform is architected around four foundational pillars:

1. **Data Sovereignty** — Every byte of training data is synthesized, validated, deduplicated, and tracked with immutable provenance records from raw source to packed dataset.
2. **Autonomous Experimentation** — Multi-agent swarms drive the scientific loop: spawning hyperparameter searches, pruning underperforming runs, escalating promising candidates to deeper alignment stages (DPO, RLHF), and logging every decision.
3. **Distributed Compute Orchestration** — A unified cluster control plane dispatches training workloads across heterogeneous hardware: Apple Silicon nodes, Linux rigs, and cloud-provisioned GPU instances (EC2, Lambda Labs, RunPod), with real-time health telemetry.
4. **Governed Inference & Deployment** — Zero-friction model promotion pipelines that publish checkpoints to local API servers, Ollama, LM Studio, or Hugging Face Hub, with built-in inference telemetry feeding failures back into the data pipeline as future training signals.

AI-Factory is the rare system that **operates itself**. It is built for researchers and engineers who want to compress the AI development loop from weeks to hours — and eventually to minutes.

---

## Core Technology Stack

### Backend & ML Infrastructure

| Layer | Technology | Role |
|---|---|---|
| Language Runtime | Python 3.10+ | Core execution environment |
| Deep Learning | PyTorch 2.x | Model computation substrate |
| Model Loading & Training | HuggingFace Transformers, PEFT, Accelerate | Fine-tuning, LoRA/QLoRA, distributed training |
| Quantization | bitsandbytes | 4-bit/8-bit inference and training |
| Alignment | TRL (trl library) | DPO, PPO, RLHF training loops |
| Hyperparameter Search | Optuna / Ray Tune | AutoML traversal strategies |
| Distributed Compute | Ray, NCCL, DeepSpeed ZeRO | Multi-node GPU orchestration |
| API Server | FastAPI + Uvicorn | Async REST API layer |
| Data Validation | Pydantic v2 | Schema enforcement, settings management |
| Task Queue | Celery + Redis | Async job dispatch, experiment queuing |
| Metrics & Observability | Prometheus + Grafana | Real-time training telemetry |
| Experiment Tracking | MLflow / W&B (pluggable) | Run logging, artifact storage |

### Frontend & Interface Stack

| Interface | Technology | Purpose |
|---|---|---|
| Web Control Center | Next.js 14, React, Tailwind CSS | Primary laboratory dashboard |
| Desktop Application | Electron | Native Mac/Linux wrapper with OS integration |
| Terminal UI | Textual (Python) | Low-latency keyboard-driven cluster ops |
| CLI | Typer + Rich | Scriptable pipeline control layer |

### Infrastructure & Tooling

| Tool | Purpose |
|---|---|
| Ruff | Linting and auto-formatting (line length: 120) |
| Mypy (strict) | Full static type checking |
| Pytest + pytest-asyncio | Test suite with async support |
| Docker + Docker Compose | Containerized service orchestration |
| Alembic + SQLAlchemy | Database schema migrations |
| Makefile | Unified developer command surface |

---

## System Architecture

AI-Factory is partitioned into seven semi-independent subsystems, each with clearly defined interfaces and dependency boundaries. All subsystems communicate through a shared `ai_factory.core` protocol layer — no subsystem imports another directly.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI-Factory Control Plane                         │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐  │
│  │   CLI   │  │   TUI    │  │  Web API  │  │Electron │  │ Webhooks │  │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  └────┬────┘  └────┬─────┘  │
│       └────────────┴──────────────┴──────────────┴────────────┘        │
│                              FastAPI Gateway                            │
│                         (Auth → Rate Limit → Router)                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        │                          │                              │
   ┌────▼─────┐            ┌───────▼──────┐              ┌────────▼──────┐
   │  Data    │            │  Training &  │              │  Inference &  │
   │ Subsystem│            │    AutoML    │              │  Deployment   │
   │          │            │  Subsystem   │              │  Subsystem    │
   └────┬─────┘            └───────┬──────┘              └────────┬──────┘
        │                          │                              │
        └──────────────────────────┼──────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
             ┌──────▼──────┐ ┌────▼────┐  ┌──────▼──────┐
             │  Agent &    │ │Cluster  │  │  Evaluation │
             │  AutoML     │ │ Control │  │  Framework  │
             │  Orchestr.  │ │  Plane  │  │             │
             └─────────────┘ └─────────┘  └─────────────┘
                    │
             ┌──────▼──────────────────────────────────────┐
             │          ai_factory.core (Shared Foundation) │
             │  Domains · Interfaces · Lineage · Config     │
             └─────────────────────────────────────────────┘
```

### Subsystem Descriptions

**`ai_factory/` — Core Shared Foundation**
The bedrock layer. Defines all domain enumerations, abstract base protocols (training, evaluation, data), the global configuration schema, the lineage registry, and all cross-cutting utilities (logging, metrics instrumentation, secrets management). No business logic. No imports from sibling subsystems.

**`data/` — Data Sovereignty Layer**
Manages the entire dataset lifecycle from raw source ingestion to packed, validated, lineage-tagged datasets. Includes synthesis pipelines for generating domain-specific training data (math proofs, code solutions, reasoning chains, creative completions), quality control filters (perplexity scoring, deduplication via MinHash LSH, toxicity filtering), format conversion, and the `pack_summary.json` provenance manifest.

**`training/` — Training & Experiment Engine**
Owns all model training logic: LoRA/QLoRA configuration, Accelerate-powered distributed training harnesses, DPO/RLHF alignment loops, checkpoint management, and the AutoML search tree executor. Receives job specifications from the Agent Orchestration layer and emits structured metric streams to the Evaluation framework.

**`evaluation/` — Evaluation Framework**
An extensible benchmark registry and automated evaluation harness. Supports standard benchmarks (MMLU, HumanEval, GSM8K, MT-Bench) and custom domain-specific evaluations. Produces structured `EvaluationReport` objects consumed by the Agent Orchestration layer for experiment pruning and promotion decisions.

**`inference/` — Inference & Deployment Server**
A production FastAPI inference server with multi-model routing, streaming token generation, request batching, and deployment adapters for Ollama, LM Studio, vLLM, and Hugging Face Hub. Includes built-in telemetry middleware that flags low-confidence or failed completions as candidate data points for re-entry into the data pipeline.

**`agents/` — Agent & AutoML Orchestration**
The autonomous loop engine. Houses the multi-agent swarm framework, the AutoML experiment tree (Bayesian / evolutionary / grid search strategies), the experiment scheduler, the pruning heuristics engine, and the promotion logic that escalates winning LoRA candidates to alignment training stages.

**`cluster/` — Distributed Compute Control Plane**
Manages the fleet of compute nodes. Abstracts over local Apple Silicon, Linux GPU rigs, and cloud instances through a unified `NodeDriver` protocol. Provides health monitoring (VRAM, utilization, temperature), job dispatch, and failure recovery with automatic re-queuing.

---

## The Autonomous Lifecycle Loop

The core value proposition of AI-Factory V2 is the **Autonomous Loop**: a closed feedback cycle that runs the ML scientific method without human intervention at every step.

### Phase 1 — Data Assembly & Curation

The data pipeline is not passive storage — it actively **builds** training datasets.

Raw sources (web crawls, curated corpora, synthetic generation prompts) enter the ingestion layer. From there:

- **Synthesis engines** generate domain-specific training examples using configured generation models. Math problems are generated with verifiable solutions. Code tasks include executable test suites. Reasoning traces are generated with chain-of-thought scaffolding.
- **Quality control filters** run in parallel: perplexity scoring removes incoherent samples, MinHash LSH deduplication removes near-duplicates at configurable Jaccard thresholds, toxicity classifiers filter harmful content, and format validators enforce schema compliance.
- **Lineage manifests** (`pack_summary.json`) are written atomically, recording the exact filter configuration, source distribution, sample count, quality score distribution, and the Git SHA of the pipeline code that produced the pack. Every packed dataset is immutably identified by a content-addressed hash.
- Packed datasets are registered in the central **Lineage Registry** and become available to the training subsystem.

### Phase 2 — Autonomous Experimentation (AutoML & Agent Swarm)

Once a dataset is packed and registered, agents can be dispatched to explore the training space.

An **AutoML Search Tree** is initialized with a search strategy (Bayesian optimization, evolutionary, or grid) and a hyperparameter space definition. The system then:

1. Materializes an initial population of training configurations (learning rate schedules, LoRA rank/alpha, warmup ratios, batch sizes, optimizer choices).
2. Dispatches up to N concurrent training runs across available cluster nodes.
3. **Agent evaluators** monitor live training metrics (loss curves, gradient norms, eval loss plateaus) and apply pruning heuristics to terminate underperforming runs early (Hyperband / ASHA scheduling).
4. Completed runs are evaluated against the registered benchmark suite. `EvaluationReport` objects are returned to the orchestration layer.
5. The **Promotion Engine** selects the top-K candidates by composite score (task accuracy, perplexity, latency profile) and optionally escalates them to an **alignment training stage**: DPO on a curated preference dataset, or PPO with a reward model.
6. The winning aligned model is registered in the Lineage Registry with full parentage: dataset hash → base model → LoRA config → DPO dataset → final checkpoint.

This loop runs continuously. The system trains while you sleep.

### Phase 3 — Distributed Cluster Orchestration

The Cluster Control Plane abstracts over heterogeneous hardware through the `NodeDriver` protocol:

- **Local Apple Silicon** nodes communicate via gRPC with PyTorch MPS backends.
- **Local Linux GPU rigs** are managed via SSH + Ray cluster bootstrap, with NCCL for multi-GPU collectives.
- **Cloud instances** (EC2, Lambda Labs, RunPod) are provisioned on-demand via provider APIs and auto-terminated on job completion to minimize cost.

All nodes emit a continuous health telemetry stream: VRAM usage, GPU utilization, memory bandwidth, temperature, and power draw. The web dashboard renders this as a live fleet map. Jobs that fail mid-run are automatically re-queued with exponential backoff; persistent failures trigger operator alerts via the configured notification channel (Slack, PagerDuty, or email).

### Phase 4 — Inference, Deployment & Feedback

Promoted model checkpoints flow through the deployment pipeline:

- **Local API Shim**: The FastAPI inference server loads the checkpoint with vLLM or transformers, serves a streaming `/v1/completions` endpoint, and is immediately accessible to the chat UI.
- **Ollama / LM Studio**: A deployment adapter converts the checkpoint to GGUF/GGML format and registers it with the local runtime.
- **Hugging Face Hub**: An automated push pipeline uploads the checkpoint, model card (auto-generated from lineage metadata), and eval results.

The inference server's **telemetry middleware** records every request: prompt, completion, latency, confidence score, and user feedback signal. Completions that fall below a configurable confidence threshold, or that receive explicit negative feedback, are flagged and routed to the data pipeline as candidate training examples — closing the loop.

---

## Interfaces

AI-Factory exposes a unified control surface across four perfectly synchronized interfaces. All interfaces communicate with the same FastAPI backend; there is no interface-specific business logic.

### Web Control Center (`frontend/`)

A Next.js 14 application organized into two primary spheres:

**The Lifecycle Sphere** (V1 capabilities, production-hardened):
- Model registry with lineage graph visualization (parent → child → grandchild checkpoint chains).
- Training job dashboard with live loss curve streaming via WebSocket.
- Evaluation results browser with cross-model comparison tables.
- Inference server management with endpoint health indicators.

**The Laboratory Sphere** (V2 autonomous capabilities):
- **Dataset Studio**: Visualize the curation pipeline as a directed acyclic graph. Monitor filter pass rates in real-time. Inspect individual samples. Trigger re-packing with modified filter configs.
- **AutoML Explorer**: Interactive search tree visualization showing the hyperparameter space, active runs, pruned branches, and the current Pareto frontier of cost vs. accuracy.
- **Agent Monitor**: Live feed of agent decisions — which runs were spawned, which were pruned, which were promoted, and the reasoning behind each decision.
- **Cluster Fleet Map**: Real-time hardware telemetry displayed as a node grid. Click any node to drill into per-GPU metrics.
- **Inference Chat UI**: Built-in chat interface connected to the local inference server, with response telemetry displayed in a side panel. Flag buttons route completions to the data pipeline.

### Desktop Application (`desktop/`)

An Electron wrapper around the Web Control Center, providing:

- **Native OS menus**: Full macOS menu bar integration with keyboard shortcuts for all major actions.
- **Deep linking**: `aifactory://` URL scheme for opening specific experiments, models, or dataset views directly from the CLI or other applications.
- **Background service management**: Auto-starts and monitors the backend server process; displays system tray indicators for active training jobs.
- **Native file dialogs**: Drag-and-drop dataset import, native save-as dialogs for checkpoint export.
- **Auto-update**: Squirrel-based update pipeline with staged rollouts.

### Terminal UI (`tui/`)

A Textual-based full-screen terminal application optimized for keyboard-driven cluster operations and real-time metric monitoring. Designed for SSH sessions on remote training rigs where a browser is unavailable.

Key panels:
- **Cluster Overview**: ASCII node grid with per-GPU utilization bars updated every second.
- **Job Queue**: Scrollable list of pending, running, and completed training jobs with live progress indicators.
- **Log Stream**: Filtered, colored log output from any running job, with regex search.
- **Quick Actions**: Keyboard-shortcut-driven commands for pausing jobs, promoting models, and triggering evaluations.

### CLI (`ai_factory/cli/`)

The scriptable control layer, built with Typer and Rich. Every operation available in the web UI is accessible as a CLI command, enabling pipeline integrations, cron scheduling, and CI/CD automation.

```bash
# Core commands
ai-factory serve                          # Start the backend API server
ai-factory tui                            # Launch the terminal UI
ai-factory cluster status                 # Show all node health
ai-factory cluster dispatch <job.yaml>    # Submit a training job

# Data pipeline
ai-factory data synthesize --domain math --count 50000
ai-factory data pack --filter-config configs/quality/strict.yaml
ai-factory data inspect <pack-hash>

# Training & AutoML
ai-factory train run configs/training/lora_7b.yaml
ai-factory automl search configs/automl/bayesian_search.yaml --parallelism 8
ai-factory automl status <search-id>

# Evaluation
ai-factory eval run <checkpoint-path> --benchmarks mmlu,humaneval,gsm8k
ai-factory eval compare <checkpoint-a> <checkpoint-b>

# Inference & Deployment
ai-factory infer serve <checkpoint-path> --port 8080
ai-factory deploy ollama <checkpoint-path>
ai-factory deploy hf <checkpoint-path> --repo-id org/model-name

# Lineage
ai-factory lineage show <checkpoint-hash>
ai-factory lineage graph --format svg > lineage.svg
```

---

## User Levels & Operational Modes

The system adapts its exposed complexity to the operator's depth of engagement.

### Beginner — Guided Workflow Mode

Beginners interact through the **Setup Wizard** in the Web Control Center. The wizard presents a sequence of opinionated defaults:

- Select a base model from the curated registry (Llama 3, Mistral, Gemma, Phi).
- Choose a training domain (math, code, reasoning, creative writing).
- Click **"Synthesize & Train"**: the system generates a synthetic dataset, selects appropriate LoRA defaults, trains for a preset number of steps, evaluates against the standard benchmark suite, and presents the result.
- The entire run is completed in under 15 minutes on a single consumer GPU.

No distributed complexity is exposed. No configuration files are required.

### Hobbyist — Experimenter Mode

The Hobbyist unlocks parameter controls:

- Upload custom datasets alongside synthesized data. Mix ratios are configurable.
- Adjust LoRA rank, alpha, target modules, learning rate, and batch size through the parameter editor UI.
- Enable multi-GPU training on a single local machine (DDP via Accelerate).
- Browse and re-run past experiments from the experiment history log.
- Inspect evaluation results at the individual-sample level to understand failure modes.

### Developer / Researcher — Architect Mode

The full system surface is exposed:

- **Agent Orchestration**: Author custom agent strategies in Python by implementing the `AgentStrategy` protocol. Define custom pruning heuristics, promotion criteria, and escalation logic.
- **AutoML Traversal**: Override search strategies with custom Optuna samplers. Define multi-objective search spaces mixing training performance, inference latency, and VRAM footprint.
- **Architecture Overrides**: Drop down to raw `peft` and `transformers` configuration. Inject custom model architectures, attention variants, or tokenizer modifications.
- **Cluster Topology**: Define heterogeneous cluster topologies in YAML. Configure NCCL communication backends, gradient compression, and pipeline parallelism stages.
- **Data Pipeline Extension**: Implement custom `SynthesisEngine`, `QualityFilter`, and `DataAdapter` classes. Register them with the plugin system for use in pipeline configs.
- **Benchmark Authoring**: Write and register custom evaluation benchmarks. Define custom scoring rubrics. Integrate with external evaluation APIs (GPT-4 judge, human eval platforms).
- **Lineage API**: Query the full lineage graph programmatically. Build custom reporting dashboards on top of the lineage registry.

---

## Development Standards & Conventions

### Code Quality

AI-Factory enforces a zero-tolerance policy on code quality. All contributions must pass the full quality gate before merging.

**Formatting & Linting**: `ruff` is the single source of truth for both formatting and linting. Line length is 120 characters. Import ordering, unused variable detection, and common anti-patterns are all enforced. Run `make format && make lint` before every commit. CI will reject any PR that fails this check.

**Static Typing**: All Python code uses strict `mypy` type checking (`--strict` flag). `Pydantic` v2 models are used for all configuration, API request/response schemas, and inter-subsystem data transfer objects. There are no `Any` types in public interfaces. Type ignore comments require an accompanying explanation comment.

**Testing**: Tests live in `tests/` and mirror the subsystem structure. The target is ≥90% line coverage on all non-trivial modules. `pytest-asyncio` handles async test cases. Integration tests spin up ephemeral in-process backends using `httpx.AsyncClient`. Data pipeline tests use deterministic synthetic generators seeded for reproducibility. CI runs the full test suite on every push.

**Architecture Discipline**: The `ai_factory.core` layer must never import from any subsystem. Subsystems must never import from each other directly — cross-subsystem communication flows through the API layer or shared core protocols. Violations are enforced by a custom `import-linter` rule in CI.

### Configuration Management

All system behavior is driven by YAML configuration files. There are no magic constants in business logic. Configuration schemas are defined as Pydantic models, validated at startup, and documented with field-level descriptions and example values.

Configuration hierarchy:
```
configs/
  defaults/          # Opinionated defaults for all subsystems
  training/          # Training job configurations
  automl/            # AutoML search space definitions
  quality/           # Data quality filter profiles
  cluster/           # Cluster topology definitions
  deployment/        # Deployment target configurations
  evaluation/        # Benchmark suite definitions
```

Environment-specific overrides are managed through `.env` files and the `Settings` class (Pydantic BaseSettings), which merges environment variables, `.env` files, and YAML configs with a well-defined precedence order.

### Extensibility & Plugin Architecture

AI-Factory is built on a plugin architecture centered on the `ai_factory.domains` registry. Every major extension point is defined as a Python Protocol (structural typing) in `ai_factory.core.interfaces`. Implementations are registered via a class decorator:

```python
@ai_factory.register(domain="math", kind="synthesis_engine")
class MathSynthesisEngine(SynthesisEngineProtocol):
    ...
```

The factory system discovers registered implementations at runtime, enabling zero-config extension: drop a new synthesis engine, quality filter, agent strategy, or benchmark into the appropriate module, decorate it, and it is immediately available to the configuration layer.

### Data Lineage Contract

Lineage is a first-class citizen, not an afterthought. The **Lineage Contract** mandates:

- Every packed dataset carries an immutable `pack_summary.json` with: content hash, source distribution, filter configuration hash, sample count, quality score statistics, pipeline code Git SHA, and creation timestamp.
- Every training run records: dataset hash, base model identifier, training configuration hash, hardware topology, elapsed wall time, peak VRAM usage, and final evaluation scores.
- Every deployed checkpoint records: training run identifier, promotion criteria, deployment target, and deployment timestamp.
- The Lineage Registry stores the full directed acyclic graph of all these relationships and exposes it via a queryable API and a visual graph in the Web Control Center.

This contract ensures that any checkpoint can be traced back to the exact data and configuration that produced it — and reproduced from scratch if the data and compute are available.

---

## Getting Started

### Prerequisites

- Python >= 3.10
- Node.js >= 18 and npm (for frontend and desktop)
- Docker and Docker Compose (optional, for containerized deployment)
- CUDA-capable GPU or Apple Silicon (for local training)
- Redis (for task queue; included in Docker Compose)

### Installation

```bash
# Clone the repository
git clone https://github.com/super244/ai-factory.git
cd ai-factory

# Install Python dependencies (editable mode with all dev extras)
pip install -e ".[dev,training,inference,agents]"

# Install frontend dependencies
cd frontend && npm install && cd ..

# Install desktop dependencies (optional)
cd desktop && npm install && cd ..

# Copy and configure environment
cp .env.example .env
# Edit .env to configure your cluster nodes, API keys, and storage paths

# Start all services via Docker Compose (recommended for first run)
make docker-up

# Or start services individually
make serve          # FastAPI backend on :8000
make frontend-dev   # Next.js frontend on :3000
make worker         # Celery task worker
```

### Key Makefile Targets

| Target | Description |
|---|---|
| `make serve` | Start FastAPI backend server (Uvicorn, port 8000) |
| `make frontend-dev` | Start Next.js development server (port 3000) |
| `make worker` | Start Celery task worker for async job processing |
| `make tui` | Launch the terminal UI |
| `make test` | Run full pytest suite with coverage report |
| `make lint` | Run ruff linter and mypy type checker |
| `make format` | Auto-format all Python code with ruff |
| `make typecheck` | Run mypy in strict mode |
| `make docker-up` | Start all services via Docker Compose |
| `make docker-down` | Stop and remove all Docker containers |
| `make clean` | Remove all build artifacts, caches, and temp files |
| `make lineage-graph` | Export the full lineage graph as SVG |

---

## Design Principles

**Lineage Above All** — Every model must know exactly what data made it, what configuration trained it, who its parent checkpoint was, and what its evaluation scores were at every stage. Lineage is never optional. There is no way to create a checkpoint outside the lineage system.

**Immaculate Codebase** — The codebase is held to the standard of a published library: strict PEP-8 compliance enforced by ruff, 100% strict mypy typing, comprehensive docstrings on all public interfaces, and ruthless enforcement of clean architecture boundaries. Technical debt is not deferred; it is addressed at the point of introduction.

**Modularity Without Compromise** — Every component is replaceable without modifying any other component. The frontend calls an API interface; the CLI calls the same interface. A new synthesis engine is a new class and a decorator. A new cluster backend is a new `NodeDriver` implementation. The core never changes when the periphery evolves.

**Autonomous by Default** — The system is designed to run unsupervised. Every long-running operation has a timeout, a failure recovery path, and an observable status. Agents make decisions with logged reasoning. The human operator is an auditor and approver, not a required participant in every step.

**Reproducibility as Infrastructure** — Any experiment must be fully reproducible from its lineage record. The system treats reproducibility not as a documentation concern but as an infrastructure guarantee: deterministic data packing, configuration hashing, environment pinning, and hardware topology recording are built into the core abstractions.

---

## Ultimate Vision

> AI-Factory becomes the autonomous AI laboratory for researchers and engineers who operate at hyperspeed: a system that rivals the capabilities of enterprise platforms like NVIDIA NeMo, but is built natively for individuals and small teams who refuse to be slowed down by bureaucratic tooling.

It is a system that compresses the distance between an idea and a deployed, evaluated model to the smallest interval physics and compute will allow — and then keeps compressing it.