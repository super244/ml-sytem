# AI-Factory Quickstart

This guide is the fastest way to get AI-Factory running for development, training, inference, and evaluation, whether you are working on a laptop, a single GPU workstation, or a cloud VM.

AI-Factory is designed to scale down to local iteration and scale up to cloud training without changing the repo layout or the main commands.

## What You Can Do From This Guide

- set up the Python and frontend dev environment
- prepare synthetic and normalized datasets
- run dry-run training validation
- launch real training locally or on a cloud GPU box
- serve the inference API
- run the research frontend
- evaluate runs and inspect artifacts

## 1. Choose A Runtime Mode

### Local dev

Use this when you want to:

- build the repo
- generate data
- run dry-runs
- run tests
- serve the API and frontend
- do small-model or adapter-based experiments

Recommended:

- Python 3.10+
- Node.js 20+
- optional NVIDIA GPU for training or real inference

### Cloud or remote GPU

Use this when you want to:

- run longer LoRA or QLoRA jobs
- use larger model variants
- keep artifacts on attached storage
- train on a rented GPU machine or managed VM

Recommended:

- Linux GPU instance
- CUDA-capable PyTorch environment
- persistent disk mounted for `artifacts/`

## 2. Clone And Enter The Repo

```bash
git clone <your-repo-url> ai-factory
cd ai-factory
```

If you are on a cloud VM, do this after attaching storage and activating your GPU environment.

## 3. Create The Python Environment

### Standard venv flow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Conda flow

```bash
conda create -n ai-factory python=3.11 -y
conda activate ai-factory
pip install -U pip
pip install -e .[dev]
```

If you are on a GPU machine, make sure your `torch` install matches the CUDA runtime you intend to use.

## 4. Configure Environment Variables

Copy the example env file:

```bash
cp .env.example .env
```

Important variables:

- `BASE_MODEL_NAME`
- `MODEL_REGISTRY_PATH`
- `PROMPT_LIBRARY_PATH`
- `BENCHMARK_REGISTRY_PATH`
- `ARTIFACTS_DIR`
- `INFERENCE_CACHE_DIR`
- `INFERENCE_TELEMETRY_PATH`
- `NEXT_PUBLIC_API_BASE_URL`
- `CORS_ORIGINS`

Typical local values already match the defaults in [.env.example](.env.example).

Typical cloud adjustments:

- point `ARTIFACTS_DIR` at mounted storage
- keep `NEXT_PUBLIC_API_BASE_URL` pointed at your API host
- widen `CORS_ORIGINS` if frontend and API are on different hosts

## 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

If you are only doing backend or training work, you can skip this until later.

## 6. Verify The Workspace

Start with the repo doctor:

```bash
python scripts/doctor.py
```

This checks:

- Python package availability
- dataset/catalog presence
- benchmark registry visibility
- discovered training runs
- frontend dependency state

For a fuller local validation pass:

```bash
python scripts/refresh_lab.py
```

Lighter version:

```bash
python scripts/refresh_lab.py --skip-tests --skip-notebooks
```

## 7. Prepare Data

### Generate synthetic custom families

```bash
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
```

This builds the custom families, including calculus-heavy packs and olympiad-style reasoning samples.

### Normalize public datasets

Run this only if you have network access and want the public adapters normalized locally:

```bash
python data/public/normalize_public_datasets.py --registry data/public/registry.yaml
```

### Build the processed corpus

```bash
python data/prepare_dataset.py --config data/configs/processing.yaml
```

### Validate the processed corpus

```bash
python data/tools/validate_dataset.py --input data/processed/train.jsonl --manifest data/processed/manifest.json
```

Main outputs:

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`
- `data/processed/test.jsonl`
- `data/processed/manifest.json`
- `data/processed/pack_summary.json`
- `data/processed/packs/<pack_id>/records.jsonl`

## 8. Train A Model

### First do a dry-run

Always validate before a real run:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
```

This validates:

- config composition
- dataset manifests and paths
- prompt rendering
- artifact layout
- tokenizer wiring

### Local training

Good starting profiles:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

### Cloud or remote GPU training

On a cloud GPU instance, the same commands work. The main differences are operational:

- use a larger attached disk for `artifacts/`
- choose a heavier profile if your GPU memory allows it
- prefer long-running jobs in `tmux`, `screen`, or a job runner
- keep logs and artifacts on persistent storage

Examples:

```bash
python -m training.train --config training/configs/profiles/curriculum_specialist.yaml
python -m training.train --config training/configs/profiles/failure_aware.yaml
python -m training.train --config training/configs/profiles/verifier_augmented.yaml
python -m training.train --config training/configs/profiles/long_context.yaml
```

Useful profiles:

- `baseline_qlora.yaml`
- `fast_iteration_small_model.yaml`
- `calculus_specialist.yaml`
- `curriculum_specialist.yaml`
- `failure_aware.yaml`
- `verifier_augmented.yaml`
- `long_context.yaml`

### Resume or inspect runs

Use these after a dry-run or full run:

```bash
python scripts/latest_run.py
```

Artifacts are written under:

- `artifacts/runs/<run_id>/`
- `artifacts/models/<model_name>/`

## 9. Serve The Inference API

Launch the FastAPI app:

```bash
uvicorn inference.app.main:app --reload
```

Important routes:

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/prompts`
- `GET /v1/datasets`
- `GET /v1/benchmarks`
- `GET /v1/runs`
- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`

When the API is running, you can use the smoke helper:

```bash
python scripts/api_smoke.py
```

If you do not want to test a live server, `python scripts/api_smoke.py --help` is still useful for checking the CLI surface.

## 10. Start The Frontend

From the repo root:

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Main routes:

- `/`
- `/compare`
- `/datasets`
- `/benchmarks`
- `/runs`

Useful frontend commands:

```bash
cd frontend
npm run typecheck
npm run build
```

## 11. Evaluate Models

Run the default comparison:

```bash
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

Other useful evaluation configs:

```bash
python -m evaluation.evaluate --config evaluation/configs/verifier_on_off.yaml
python -m evaluation.evaluate --config evaluation/configs/curriculum_ablation.yaml
python -m evaluation.evaluate --config evaluation/configs/source_ablation_calculus_only.yaml
```

Evaluation outputs typically land under `evaluation/results/` and include JSON, JSONL, and Markdown reports.

## 12. Refresh The Notebook Lab

```bash
python notebooks/build_notebooks.py
```

Use this after data rebuilds, new training runs, or evaluation changes.

## 13. Optional Container Workflow

For a lightweight packaged demo stack:

```bash
docker compose up --build
```

Repo files:

- [Dockerfile.api](/Users/raofu/ai-factory/Dockerfile.api)
- [Dockerfile.frontend](/Users/raofu/ai-factory/Dockerfile.frontend)
- [docker-compose.yml](/Users/raofu/ai-factory/docker-compose.yml)

This is useful when you want a cleaner separation between API and frontend services.

## 14. Recommended End-To-End Flows

### Fast local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
cp .env.example .env
python scripts/doctor.py
python scripts/refresh_lab.py --skip-tests --skip-notebooks
uvicorn inference.app.main:app --reload
```

### Local research loop

```bash
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python data/prepare_dataset.py --config data/configs/processing.yaml
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
python notebooks/build_notebooks.py
```

### Cloud training loop

```bash
cp .env.example .env
python scripts/doctor.py
python data/prepare_dataset.py --config data/configs/processing.yaml
python -m training.train --config training/configs/profiles/calculus_specialist.yaml --dry-run
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
python scripts/latest_run.py
```

## 15. Handy Make Targets

```bash
make doctor
make refresh-lab
make latest-run
make api-smoke
make train-dry
make evaluate
make notebooks
make frontend-typecheck
make frontend-build
```

## 16. Common Problems

- `ModuleNotFoundError` from repo scripts:
  Use the repo root as your working directory before running commands.
- Frontend build or typecheck fails:
  Install `frontend/node_modules` first with `npm install`.
- Public dataset normalization does nothing:
  Check network access and confirm `datasets` is installed.
- Full model loading fails:
  Confirm local model assets or Hugging Face access are available.
- Training is too heavy for local hardware:
  Start with `fast_iteration_small_model.yaml` or do dry-runs locally and full runs on a cloud GPU.

## 17. Where To Go Next

- [README.md](/Users/raofu/ai-factory/README.md)
- [docs/runbook.md](/Users/raofu/ai-factory/docs/runbook.md)
- [docs/experiment-playbook.md](/Users/raofu/ai-factory/docs/experiment-playbook.md)
- [docs/deployment-guide.md](/Users/raofu/ai-factory/docs/deployment-guide.md)
