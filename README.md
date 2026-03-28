# Atlas Math Lab

Atlas Math Lab is a research-grade, production-aware monorepo for building a specialized competitive mathematics and advanced calculus reasoning model. The repository combines synthetic data generation, public dataset normalization, curriculum-aware fine-tuning, verifier-assisted inference, rigorous benchmarking, and a polished research-product frontend around a shared schema and artifact system.

The default local path is intentionally practical: `Qwen2.5-Math-1.5B-Instruct` plus QLoRA/LoRA. The platform also includes larger 7B-class model templates, Accelerate/DeepSpeed-ready runtime configs, and packaging hooks for moving from fast local iteration to heavier training runs.

Start with `quickstart.md` if you want the fastest path to a working local workspace.

## What The System Supports

- Canonical `v2` dataset records with typed `step_checks`, lineage, contamination metadata, quality scores, reasoning style tags, and generator metadata.
- Six first-class custom synthetic families: derivatives, integrals, limits/series, multivariable calculus, ODEs/optimization, and olympiad reasoning.
- Public dataset registry and adapters for calculus-heavy competition and instruction corpora.
- Derived pack construction for `core_train_mix`, `calculus_hard_pack`, `olympiad_reasoning_pack`, `failure_replay_pack`, `verification_pack`, and `benchmark_holdout_pack`.
- Composed training profiles for `baseline_qlora`, `calculus_specialist`, `curriculum_specialist`, `failure_aware`, `verifier_augmented`, `long_context`, and `fast_iteration_small_model`.
- A SQLite-backed orchestration control plane for managed `prepare`, `train`, `finetune`, `evaluate`, `report`, `inference`, and `deploy` instances with async task attempts, heartbeats, retries/backoff, child-task lineage, publish hooks, and follow-up recommendations.
- Config-driven instance templates under `configs/` for `prepare`, `train`, `finetune`, `eval`, `inference`, `deploy`, and `report`, all routed through the same orchestration backend.
- FastAPI inference with prompt presets, lazy model loading, best-of-N candidate selection, self-consistency, answer extraction, safe calculator hooks, structured output, and compare-two-models mode.
- Benchmark-oriented evaluation with per-topic, per-difficulty, per-source, and per-pack reporting plus failure taxonomy and win-case extraction.
- A multi-route Next.js frontend for solving, comparing, browsing datasets, exploring benchmarks, and inspecting training/evaluation runs, including the shared runs/control-plane view.

## Repo Spine

```text
.
├── ai_factory/core/          shared schemas, artifact IO, hashing, answers, reports, token helpers
├── data/                     registries, adapters, synthesis, quality, pack building, audit/export tools
├── training/                 composed configs, trainer extensions, packaging, comparisons, run scripts
├── inference/                FastAPI app, model registry, prompts, generation services, metadata services
├── evaluation/               benchmark registry, metrics, reporting, failure analysis, eval configs
├── frontend/                 Next.js research product UI with solve/compare/datasets/benchmarks/runs views and control-plane surface
├── notebooks/                generated notebook lab for data, training, inference, and evaluation exploration
├── docs/                     architecture, subsystem guides, API/deployment/contributor documentation
├── tests/                    unit and contract coverage across shared-core, data, training, inference, evaluation
└── artifacts/                standardized run and model outputs written during training/eval/inference
```

## Quickstart

1. Install Python dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .[dev]
   ```

2. Generate local synthetic datasets and normalize any downloaded public datasets.

   ```bash
   python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
   python3 data/public/normalize_public_datasets.py --registry data/public/registry.yaml
   python3 data/prepare_dataset.py --config data/configs/processing.yaml
   ```

3. Validate or launch training.

   ```bash
   python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
   python3 -m training.train --config training/configs/profiles/calculus_specialist.yaml
   ```

4. Serve the inference API and the frontend.

   ```bash
   uvicorn inference.app.main:app --reload
   cd frontend
   npm install
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
   ```

5. Run evaluation and refresh the notebook lab.

   ```bash
   python3 -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
   python3 notebooks/build_notebooks.py
   ```

6. Optional: use the managed control plane instead of raw subsystem commands.

   ```bash
   ai-factory new --config configs/finetune.yaml
   ai-factory list
   ai-factory status <instance-id> --json
   ai-factory tasks <instance-id> --json
   ai-factory events <instance-id> --json
   ai-factory watch <instance-id> --timeout 30 --json
   ai-factory recommendations <instance-id> --json
   ai-factory children <instance-id> --json
   ai-factory --artifacts-dir /tmp/ai-factory-artifacts new --config configs/train.yaml --no-start
   ai-factory tui --refresh-seconds 2
   ```

## Artifact Layout

Run outputs are standardized around:

```text
artifacts/
├── runs/<run_id>/
│   ├── checkpoints/
│   ├── logs/
│   ├── manifests/
│   │   ├── config_snapshot.json
│   │   └── run_manifest.json
│   ├── metrics/
│   └── reports/
│       └── run_summary.md
└── models/<model_name>/
    ├── latest
    ├── adapter/
    ├── merged/
    └── serving/
```

The data system emits matching manifests and cards under `data/processed/`, `data/custom/`, `data/public/normalized/`, and `data/processed/packs/`.

The orchestration runtime stores durable control-plane state under:

```text
artifacts/control_plane/
├── control_plane.db
└── events.jsonl
```

Legacy `artifacts/instances/<instance-id>/...` directories are still materialized as compatibility projections for the CLI, API, and frontend.

## Training Profiles

- `baseline_qlora`: default local specialist baseline.
- `calculus_specialist`: heavier calculus weighting and hard-example emphasis.
- `curriculum_specialist`: curriculum-aware mixture ordering.
- `failure_aware`: replay-focused training against mined failure cases.
- `verifier_augmented`: prioritizes examples with explicit checks and verification anchors.
- `long_context`: long-context profile for extended derivations.
- `fast_iteration_small_model`: lower-cost rapid iteration profile.

## Documentation Map

- `docs/architecture.md`
- `docs/control-center.md`
- `docs/data-system.md`
- `docs/training-system.md`
- `docs/inference-system.md`
- `docs/evaluation-system.md`
- `docs/api-guide.md`
- `docs/benchmark-guide.md`
- `docs/experiment-playbook.md`
- `docs/runbook.md`
- `docs/deployment-guide.md`
- `docs/notebook-guide.md`
- `docs/contributor-guide.md`
- `docs/reproducibility.md`
- `docs/troubleshooting.md`
- `docs/model-card.md`
- `docs/roadmap.md`

## Useful Commands

```bash
make doctor
make refresh-lab
make latest-run
make api-smoke
make test
make train-dry
make evaluate
make notebooks
make frontend-typecheck
ai-factory new --config configs/finetune.yaml
ai-factory list --type finetune
ai-factory tasks <instance-id>
ai-factory events <instance-id>
ai-factory retry <instance-id>
ai-factory cancel <instance-id>
ai-factory new --config configs/finetune.yaml --environment cloud --port-forward 6006:6006
ai-factory tui --refresh-seconds 2
```

Start with `docs/runbook.md` for the end-to-end local workflow and `docs/experiment-playbook.md` for the recommended research loop.
See `docs/foundation-layer.md` for the shared control-plane backend, instance lifecycle, and CLI/TUI surface.
