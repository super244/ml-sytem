# AI-Factory Quickstart

This file is the exact operator path for this repository in its current state.

Status as validated on April 5, 2026:

- `pytest` passes.
- `ruff check .` passes.
- `mypy .` passes.
- `python scripts/doctor.py` passes.
- `ai-factory serve` and `ai-factory api-smoke` pass.
- Dataset generation, corpus preparation, SQLite corpus export, tokenizer training smoke test, and scratch-training dry-run pass.
- `finetuned` evaluation, finetuned inference, and real deployment require a first real training run because the repo does not ship packaged adapters in `artifacts/models/`.

## 1. Choose Your Track

### Local track

Use this if you want to:

- generate and pack datasets
- pretrain a scratch model from your own corpus
- validate training with `--dry-run`
- run a real small-model or adapter-based fine-tune on your own GPU
- run the API and frontend locally

Recommended:

- Python 3.10+
- Node.js 20+
- optional CUDA GPU

### Cloud GPU track

Use this if you want to:

- run a real 2B scratch pretraining job on a rented GPU VM
- keep artifacts on persistent attached storage
- avoid local GPU limits

Recommended:

- Ubuntu or another Linux VM
- CUDA-compatible PyTorch environment
- persistent disk mounted for artifacts
- `tmux` or `screen`

Do not treat `--distributed` as the primary path yet. The validated cloud path for this repo is: run the same single-process training commands on a bigger machine.

## 2. Clone And Install

```bash
git clone https://github.com/super244/ai-factory.git
cd ai-factory

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
git lfs install
git lfs pull
cp .env.example .env
```

Frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Cloud-only environment variables:

```bash
export AI_FACTORY_REPO_ROOT="$PWD"
export ARTIFACTS_DIR="/mnt/ai-factory-artifacts"
```

Use a writable persistent path for `ARTIFACTS_DIR` on cloud VMs.

If you intentionally skip Git LFS, regenerate `data/catalog.json` locally before relying on `ai-factory ready` or `ai-factory doctor`.

## 3. Verify The Workspace

```bash
ai-factory ready
python scripts/doctor.py
ai-factory doctor
```

Optional container sanity check:

```bash
docker compose config
```

## 4. Generate And Prepare Data

### 4.1 Generate synthetic datasets

```bash
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
```

Default behavior now:

- the six custom math datasets target `2.0` GiB total
- the generator splits that total budget across the six custom families
- the generated files are still JSONL under `data/custom/`

### 4.2 Optional: normalize public datasets

Run this only if you want public data folded into the corpus.

```bash
python data/public/normalize_public_datasets.py --registry data/public/registry.yaml
```

### 4.3 Build the processed corpus

```bash
python data/prepare_dataset.py --config data/configs/processing.yaml
```

This now writes both JSONL splits and a SQLite corpus database.

### 4.4 Validate the processed training split

```bash
python data/tools/validate_dataset.py --input data/processed/train.jsonl --manifest data/processed/manifest.json
```

### 4.5 Preview examples

```bash
python data/tools/preview_dataset.py --input data/processed/train.jsonl --limit 3
```

Expected outputs:

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`
- `data/processed/test.jsonl`
- `data/processed/corpus.sqlite`
- `data/processed/manifest.json`
- `data/processed/pack_summary.json`
- `data/processed/packs/*`

### 4.6 Optional: inspect the SQLite corpus

```bash
python data/tools/preview_dataset.py --input data/processed/corpus.sqlite --split train --limit 3
python data/tools/validate_dataset.py --input data/processed/corpus.sqlite --split train --manifest data/processed/manifest.json
```

Important note on SQLite:

- `data/processed/corpus.sqlite` is now the durable structured corpus store for processed train/eval/test rows
- the training stack can read either JSONL or SQLite
- the existing JSONL outputs remain in place because they are simple to diff, inspect, and reuse with external tooling
- this SQLite corpus is separate from the orchestration control-plane SQLite database under `artifacts/control_plane/`

## 5. Choose The Scratch Model Size

The scratch-model path now supports target parameter counts directly.

To plan a model around a parameter budget:

```bash
python training/scripts/plan_model_scale.py --target-parameters 2b
python training/scripts/plan_model_scale.py --target-parameters 1.3b
python training/scripts/plan_model_scale.py --target-parameters 3b
```

The default scratch model component is:

- `training/configs/components/models/qwen2_scratch_2b.yaml`

It uses:

- `model_type: qwen2`
- `target_parameters: 2b`
- `vocab_size: 50257`

If you want a different model size, edit that file and change `target_parameters` to the budget you want, for example:

```yaml
target_parameters: 1.3b
```

Then keep the tokenizer vocab size aligned with the same component.

## 6. Train The Tokenizer For Scratch Pretraining

Before the first real scratch run, train a local tokenizer that matches the configured vocab budget.

```bash
python training/scripts/train_tokenizer.py \
  --config training/configs/profiles/pretraining.yaml \
  --output-dir artifacts/tokenizers/qwen2_math_2b
```

If you want a different vocab size, update `training/configs/components/models/qwen2_scratch_2b.yaml` first, then pass the same setting through the profile.

## 7. Validate The Scratch Training Path First

Always start with a dry-run.

```bash
python -m training.train --config training/configs/profiles/pretraining.yaml --dry-run
python -m training.train --config training/configs/profiles/pretraining.yaml --dry-run --validate-model-load
```

This validates:

- config composition
- dataset paths
- tokenizer loading
- scratch architecture resolution from `target_parameters`
- full model instantiation
- artifact layout
- plain-text pretraining tokenization

For SQLite-backed training data, point the data component at `data/processed/corpus.sqlite` for train/eval/test if you prefer DB-backed reads over JSONL.

You can also run the repo refresh flow:

```bash
ai-factory refresh-lab --skip-tests --skip-notebooks --skip-generate
```

## 8. Run The First Real 2B Scratch Pretraining Job

The default scratch-pretraining profile is:

- `training/configs/profiles/pretraining.yaml`

Run it like this:

```bash
python -m training.train --config training/configs/profiles/pretraining.yaml
```

What this profile does now:

- builds a Qwen2-style decoder from scratch instead of loading `from_pretrained`
- uses `target_parameters: 2b` to resolve the model scale
- reads plain-text math documents for next-token prediction
- reads the processed corpus from `data/processed/corpus.sqlite` by default
- trains all parameters
- publishes the result under `artifacts/models/atlas-math-2b-pretrained/`

What you need operationally:

- a serious GPU machine for a real 2B run
- persistent storage for `artifacts/`
- enough disk for the 2 GiB custom corpus, processed splits, tokenizer, checkpoints, and final model

Recommended operator order:

1. `python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml`
2. `python data/prepare_dataset.py --config data/configs/processing.yaml`
3. `python training/scripts/train_tokenizer.py --config training/configs/profiles/pretraining.yaml --output-dir artifacts/tokenizers/qwen2_math_2b`
4. `python -m training.train --config training/configs/profiles/pretraining.yaml --dry-run --validate-model-load`
5. `python -m training.train --config training/configs/profiles/pretraining.yaml`

## 9. Run The First Real Fine-Tune

### Recommended first real production run

Use `failure_aware.yaml` first if you want the default finetuned evaluation and inference flows to work afterward.

```bash
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

Why this profile first:

- it publishes to `artifacts/models/atlas-math-failure-aware/latest`
- the default `finetuned` model registry entry points there
- the default `base_vs_finetuned` evaluation config expects that artifact

### Faster local smoke run

If you want a cheaper local experiment first:

```bash
python -m training.train --config training/configs/profiles/fast_dev.yaml
```

This is useful for iteration, but it does not populate the default `finetuned` registry path.

### Good next profiles

Use these after the first real run depending on what evaluation shows:

- `training/configs/profiles/math_specialist.yaml`
  Use when calculus and symbolic manipulation errors dominate.
- `training/configs/profiles/verifier_augmented.yaml`
  Use when reasoning is plausible but step consistency is weak.
- `training/configs/profiles/long_context.yaml`
  Use only on larger GPU machines.
- `training/configs/profiles/full_finetune.yaml`
  Use only when you explicitly want a heavier, more expensive run.

### Artifacts written by training

Run outputs go under:

- `artifacts/runs/<run_id>/`
- `artifacts/models/<publish_model_name>/`

Useful command:

```bash
ai-factory latest-run
```

## 10. Evaluate

### 7.1 Bootstrap evaluation before any adapter exists

If you want to validate the evaluation pipeline before the first real fine-tune:

```bash
python -m evaluation.evaluate --config evaluation/configs/base_smoke.yaml
```

### 7.2 Real base-vs-finetuned evaluation

Run this only after `failure_aware.yaml` has completed and published `artifacts/models/atlas-math-failure-aware/latest`.

```bash
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

Expected outputs:

- `evaluation/results/base_vs_finetuned/per_example.jsonl`
- `evaluation/results/base_vs_finetuned/summary.json`
- `evaluation/results/base_vs_finetuned/summary.md`
- `evaluation/results/base_vs_finetuned/leaderboard.json`

## 11. Optimization Loop

### 8.1 Mine failures into new training examples

```bash
python data/mine_failure_cases.py \
  --input evaluation/results/base_vs_finetuned/per_example.jsonl \
  --output data/raw/failure_cases.jsonl
```

### 8.2 Compare two runs

```bash
python training/scripts/compare_runs.py \
  --left artifacts/runs/<run_a> \
  --right artifacts/runs/<run_b> \
  --markdown-output artifacts/runs/comparison.md
```

### 8.3 Pick the next fine-tune

Use this rule set:

- many calculus misses: rerun with `math_specialist.yaml`
- many step-check or verifier misses: rerun with `verifier_augmented.yaml`
- long prompts or truncation pressure: move to `long_context.yaml`
- low-cost local iteration: use `fast_dev.yaml`

A common loop is:

1. `failure_aware.yaml`
2. evaluate with `base_vs_finetuned.yaml`
3. mine failures
4. rerun with `math_specialist.yaml` or `verifier_augmented.yaml`
5. compare runs

## 12. Serve And Run Inference

### 9.1 Start the API

```bash
ai-factory serve --host 127.0.0.1 --port 8000
```

### 9.2 Smoke-test the live server

```bash
ai-factory api-smoke
```

### 9.3 Query the base model

This works on a fresh checkout because `base` does not require a local adapter.

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"Evaluate \\int_0^1 x dx.","model_variant":"base"}'
```

### 9.4 Query the finetuned model

Run this only after the `failure_aware` training run has published the adapter.

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"Evaluate \\int_0^1 x dx.","model_variant":"finetuned"}'
```

If that returns an adapter-path error, your first real fine-tune has not published yet.

## 13. Local Deployment Options

### 10.1 Demo stack with Docker Compose

```bash
docker compose up --build
```

This brings up:

- API on `http://localhost:8000`
- frontend on `http://localhost:3000`

### 10.2 Provider publish path

`configs/deploy.yaml` is intentionally a dry-run by default.

For a real publish you must:

1. train a model first so a publishable artifact exists
2. edit `configs/deploy.yaml`
3. set `dry_run: false`
4. set the provider-specific options
5. install the provider CLI (`ollama`, `hf`, or equivalent)

This repository is ready for local serving now. External model publishing is a post-training, provider-specific step.

## 11. Cloud GPU Workflow

The exact commands stay the same on a cloud box.

Recommended flow:

```bash
tmux new -s ai-factory
source .venv/bin/activate
export AI_FACTORY_REPO_ROOT="$PWD"
export ARTIFACTS_DIR="/mnt/ai-factory-artifacts"
python data/prepare_dataset.py --config data/configs/processing.yaml
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/failure_aware.yaml
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

Use cloud mainly for:

- `failure_aware.yaml`
- `math_specialist.yaml`
- `verifier_augmented.yaml`
- `long_context.yaml`
- `full_finetune.yaml`

## 12. Done Means Done

You are in a good state when all of these are true:

```bash
ai-factory ready
python scripts/doctor.py
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
ai-factory serve --host 127.0.0.1 --port 8000
ai-factory api-smoke
```

And for the full finetuned loop:

```bash
python -m training.train --config training/configs/profiles/failure_aware.yaml
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"Evaluate \\int_0^1 x dx.","model_variant":"finetuned"}'
```
