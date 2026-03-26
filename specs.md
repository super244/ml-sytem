# Atlas Math Lab Specs And Command Guide

This file is the practical operator guide for this repo.

Use it to answer:

1. What hardware do I have?
2. Which commands should I run on that hardware?
3. Which training profile matches that machine?
4. What is the shortest path from setup to training, evaluation, and serving?

This repo scales from an Apple Silicon laptop to larger cloud GPUs without changing the main CLI shape. The command usually stays the same; the profile and expectations change.

## 1. Hardware Inspection Commands

### Apple Silicon macOS

Check chip, memory, and core counts:

```bash
system_profiler SPHardwareDataType
```

Quick total memory in bytes:

```bash
sysctl hw.memsize
```

Check CPU brand string:

```bash
sysctl -n machdep.cpu.brand_string
```

Important:

- Apple Silicon uses unified memory
- there is no separate NVIDIA-style VRAM number to look up
- GPU memory comes out of total system memory

Practical interpretation:

- 16 GB to 36 GB unified memory: development, data prep, dry-runs, small experiments
- 48 GB to 128 GB unified memory: stronger local iteration, some real training, still not the ideal box for the heaviest runs

### Linux CPU And RAM

CPU information:

```bash
lscpu
```

RAM information:

```bash
free -h
```

Disk space, which matters for `artifacts/` and processed data:

```bash
df -h
```

### NVIDIA GPU Linux

Basic GPU inventory:

```bash
nvidia-smi
```

Compact GPU and VRAM view:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Watch live utilization:

```bash
nvidia-smi -l 2
```

### PyTorch Runtime Checks

Check whether PyTorch sees CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Show detected GPU names:

```bash
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Useful interpretation:

- if `False`, assume CPU or non-CUDA execution
- if CUDA is visible, choose a real training profile based on VRAM tier

## 2. Core Repo Commands

These are the main commands you will actually use.

### Environment Setup

Create the Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

### Sanity Checks

Recommended first command:

```bash
make doctor
```

This is the best first stop when you are unsure whether the workspace is healthy.

Run a fuller local validation pass:

```bash
make refresh-lab
```

### Data Commands

Generate synthetic datasets:

```bash
make generate-datasets
```

Equivalent full command:

```bash
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
```

Normalize public datasets if you have network access:

```bash
python data/public/normalize_public_datasets.py --registry data/public/registry.yaml
```

Build the processed corpus:

```bash
make prepare-data
```

Equivalent full command:

```bash
python data/prepare_dataset.py --config data/configs/processing.yaml
```

Validate processed data:

```bash
make validate-data
```

If you want a stricter single-file style check:

```bash
python data/tools/validate_dataset.py --input data/processed/train.jsonl --manifest data/processed/manifest.json
```

### Training Commands

Always start with a dry-run:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
```

Validate model loading too:

```bash
make validate-model
```

Start a normal baseline run:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

Start a faster, cheaper run:

```bash
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
```

Start a more specialized calculus run:

```bash
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

Start a replay-focused run after mining failures:

```bash
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

Start the scale-up long-context run:

```bash
python -m training.train --config training/configs/profiles/long_context.yaml
```

### Post-Training Commands

See the latest run:

```bash
make latest-run
```

Compare two runs:

```bash
python training/scripts/compare_runs.py --run-a artifacts/runs/<run_a> --run-b artifacts/runs/<run_b>
```

Export a merged model:

```bash
python training/scripts/export_merged_model.py --run-dir artifacts/runs/<run_id>
```

### Inference And API

Serve the API locally:

```bash
make serve
```

Equivalent full command:

```bash
uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health-check the API:

```bash
curl http://127.0.0.1:8000/v1/health
```

Smoke-test the API flow:

```bash
make api-smoke
```

### Frontend

Start the frontend against local API:

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Typecheck the frontend:

```bash
make frontend-typecheck
```

Build the frontend:

```bash
make frontend-build
```

### Evaluation

Run the standard evaluation:

```bash
make evaluate
```

Equivalent full command:

```bash
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

Run verifier-on/off evaluation:

```bash
python -m evaluation.evaluate --config evaluation/configs/verifier_on_off.yaml
```

Analyze failures:

```bash
python evaluation/analysis/analyze_failures.py --input evaluation/results/latest/per_example.jsonl --output evaluation/results/latest/failure_analysis.json
```

Mine failures back into the next data cycle:

```bash
python data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl
```

### Tests And Notebook Refresh

Run tests:

```bash
make test
```

Refresh notebooks:

```bash
make notebooks
```

### Docker Demo Stack

Bring up the demo deployment:

```bash
docker compose up --build
```

## 3. Training Profiles And When To Use Them

### `fast_iteration_small_model`

Base model:

- `Qwen/Qwen2.5-Math-0.5B-Instruct`

Use when:

- you are on an Apple Silicon laptop
- you have limited VRAM
- you want the cheapest quick feedback loop
- you want to prove the pipeline works before spending time or money

Command:

```bash
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
```

### `baseline_qlora`

Base model:

- `Qwen/Qwen2.5-Math-1.5B-Instruct`

Use when:

- you want the default mainline experiment
- you have a decent workstation GPU or roomy Apple Silicon machine
- you are moving from dry-run to a real training pass

Command:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

### `calculus_specialist`

Base model:

- `Qwen/Qwen2.5-Math-1.5B-Instruct`

Use when:

- baseline already works
- you want a more specialized run
- you are focused on hard calculus performance

Command:

```bash
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

### `failure_aware`

Use when:

- you already ran evaluation
- you mined failures
- you want the next training cycle to focus on misses and regressions

Command:

```bash
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

### `long_context`

Base model:

- `Qwen/Qwen2.5-Math-7B-Instruct`

Use when:

- you are on a real scale-up GPU
- you want long-context experimentation
- your hardware budget is much stronger than laptop or small-workstation tier

Command:

```bash
python -m training.train --config training/configs/profiles/long_context.yaml
```

## 4. Hardware Scenarios And Recommended Commands

### Scenario A: Apple Silicon, 16 GB To 36 GB Unified Memory

Examples:

- MacBook Air
- smaller MacBook Pro

Best use:

- install and validate the repo
- generate data
- run tests
- serve API and frontend
- run dry-runs
- maybe do very small real experiments

Recommended command ladder:

```bash
make doctor
make generate-datasets
make prepare-data
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml --dry-run
make serve
```

Default recommendation:

- start with `fast_iteration_small_model`
- treat `baseline_qlora` as a dry-run target first, not your default real run

Avoid as a first move:

- `long_context`
- long all-day training jobs

### Scenario B: Apple Silicon, 48 GB To 128 GB Unified Memory

Examples:

- high-end MacBook Pro
- Mac Studio

Best use:

- strong local iteration
- data and evaluation workflows
- some real QLoRA work
- API serving and product development

Recommended command ladder:

```bash
make doctor
make prepare-data
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
```

Then, if stable:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

Caution:

- Apple unified memory is helpful, but it is still not a substitute for a large CUDA box when you want sustained heavier runs

### Scenario C: Single NVIDIA GPU, 12 GB To 24 GB VRAM

Examples:

- RTX 3080
- RTX 3090
- RTX 4090 in a tighter environment
- L4-class cloud GPU

Best use:

- real small-model runs
- short experimentation
- debugging actual training instead of only dry-runs

Recommended command:

```bash
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
```

Good supporting commands:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
make latest-run
make evaluate
```

Default advice:

- use `fast_iteration_small_model` as the real run
- use `baseline_qlora` primarily as a validated next step

### Scenario D: Single NVIDIA GPU, 24 GB To 48 GB VRAM

Examples:

- RTX 4090
- RTX 6000 class
- RTX Pro 6000 class
- A10
- A40
- L40S

Best use:

- normal local QLoRA experimentation
- primary single-GPU training box
- baseline and specialist runs

Recommended commands:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

Suggested operating pattern:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
make evaluate
python data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

This is the best general-purpose tier for most serious local work.

### Scenario E: 80 GB+ Cloud GPU

Examples:

- A100 80GB
- H100
- B200

Best use:

- scale-up training
- larger context budgets
- more ambitious experiments
- longer uninterrupted runs

Recommended commands:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
python -m training.train --config training/configs/profiles/long_context.yaml
```

Recommended flow:

```bash
make doctor
make prepare-data
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/long_context.yaml
make evaluate
python evaluation/analysis/analyze_failures.py --input evaluation/results/latest/per_example.jsonl --output evaluation/results/latest/failure_analysis.json
```

This is the right tier for the repo's most ambitious profile.

## 5. Command By Goal

### Goal: I Just Cloned The Repo

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
cd frontend
npm install
cd ..
make doctor
```

### Goal: I Want To Verify Everything Without Waiting Forever

Run:

```bash
make doctor
make generate-datasets
make prepare-data
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
make api-smoke
```

### Goal: I Want The Safest First Training Run

Run:

```bash
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
```

Use this especially on:

- laptops
- unknown hardware
- first-time setup
- smaller GPUs

### Goal: I Want The Default Serious Run

Run:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

### Goal: I Want More Calculus Specialization

Run:

```bash
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

### Goal: I Want To Close The Loop After Evaluation

Run:

```bash
make evaluate
python evaluation/analysis/analyze_failures.py --input evaluation/results/latest/per_example.jsonl --output evaluation/results/latest/failure_analysis.json
python data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

### Goal: I Want To Run The App Locally

Terminal A:

```bash
make serve
```

Terminal B:

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Check:

```bash
curl http://127.0.0.1:8000/v1/health
```

### Goal: I Want A Demo Stack Quickly

Run:

```bash
docker compose up --build
```

## 6. Practical Machine Recommendations

### Best Command On An Unknown Machine

If you know nothing yet about the machine, start here:

```bash
make doctor
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml --dry-run
```

This is the safest high-signal start.

### Best Command On A Laptop

Run:

```bash
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml --dry-run
```

### Best Command On A Single Good GPU

Run:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

### Best Command On A Stronger Workstation Or Cloud Box

Run:

```bash
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
```

### Best Command On H100 Or B200 Class Hardware

Run:

```bash
python -m training.train --config training/configs/profiles/long_context.yaml
```

## 7. Fast Decision Table

- Apple Silicon with modest unified memory: `fast_iteration_small_model`
- Apple Silicon with lots of unified memory: `fast_iteration_small_model`, then `baseline_qlora`
- 12 GB to 24 GB CUDA VRAM: `fast_iteration_small_model`
- 24 GB to 48 GB CUDA VRAM: `baseline_qlora`, then `calculus_specialist`
- 80 GB+ cloud GPU: `calculus_specialist` and `long_context`

## 8. Common Failure Patterns

### Dry-run Passes But Real Training Fails

Likely cause:

- actual model load exceeds your memory budget even though config validation succeeded

What to do:

- step down from `baseline_qlora` to `fast_iteration_small_model`
- verify CUDA visibility with the PyTorch checks above
- confirm actual GPU VRAM with `nvidia-smi`

### Public Dataset Normalization Fails

Likely cause:

- no network access
- missing dataset dependency setup

What to do:

- skip normalization
- continue with local synthetic data generation

Command to continue:

```bash
make generate-datasets
make prepare-data
```

### Frontend Cannot Reach The API

What to check:

- API is running
- `NEXT_PUBLIC_API_BASE_URL` points to the right host and port
- health endpoint responds

Check command:

```bash
curl http://127.0.0.1:8000/v1/health
```

## 9. Default End-To-End Ladder

If you want the repo's most sensible general workflow, use this exact order:

```bash
make doctor
make generate-datasets
make prepare-data
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
make evaluate
python data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

If your machine is weaker, replace `baseline_qlora` with `fast_iteration_small_model`.

If your machine is much stronger, add:

```bash
python -m training.train --config training/configs/profiles/long_context.yaml
```
