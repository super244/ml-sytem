# AI-Factory Quickstart

This file is the exact operator path for this repository in its current state.

Status as validated on April 5, 2026:

- `pytest` passes.
- `ruff check .` passes.
- `mypy .` passes.
- `python scripts/doctor.py` passes.
- `ai-factory ready` passes.
- `ai-factory train-preflight --config training/configs/profiles/failure_aware.yaml` passes with environment warnings on CPU-only hosts.
- **Hardware detection and ultimate optimization** are available via `python -m training.src.optimization`
- Ultimate profiles `m5_max_ultimate.yaml`, `cuda_ultimate_a100.yaml`, and `cuda_ultimate_h100.yaml` are ready for hardware-specific training
- Frontend `npm run typecheck` and `npm run build` pass.
- Dataset generation, corpus preparation, SQLite corpus export, tokenizer training smoke test, and scratch-training dry-run pass with a faster corpus/tokenizer path.
- The Linux cloud bootstrap script and macOS local bootstrap script are the default operator entry points.
- `finetuned` evaluation, finetuned inference, and real deployment require a first real training run because the repo does not ship packaged adapters in `artifacts/models/`.

## 1. Choose Your Track

### Local track

Use this if you want to:

- generate and pack datasets
- pretrain a scratch model from your own corpus
- validate training with `--dry-run`
- run a real small-model or adapter-based fine-tune on your own GPU
- run the API and frontend locally
- validate the full training configuration on a CPU-only workstation before moving to a GPU box
- use the Apple Silicon-safe local profile at `training/configs/profiles/local_metal.yaml`
- **use ultimate optimization on Apple Silicon** with `training/configs/profiles/m5_max_ultimate.yaml`

Recommended:

- Python 3.10+
- Node.js 20+
- optional CUDA GPU
- the macOS bootstrap script for Apple Silicon local runs

### Cloud GPU track

Use this if you want to:

- run a real 2B scratch pretraining job on a rented GPU VM
- keep artifacts on persistent attached storage
- avoid local GPU limits
- **use ultimate optimization on NVIDIA A100/H100** with `cuda_ultimate_a100.yaml` or `cuda_ultimate_h100.yaml`

Recommended:

- Ubuntu or another Linux VM
- CUDA-compatible PyTorch environment
- persistent disk mounted for artifacts
- `tmux` or `screen`
- the Linux cloud bootstrap script after SSHing into the instance

Do not treat `--distributed` as the primary path yet. The validated cloud path for this repo is: run the same single-process training commands on a bigger machine.

## 2. Clone And Install

```bash
git clone https://github.com/super244/ai-factory.git
cd ai-factory

# Linux cloud GPU instance
bash scripts/start-linux.sh

# Apple Silicon local machine
bash scripts/start-mac.sh
```

The bootstrap scripts install dependencies, download tokenizer and model prerequisites, handle the common CUDA and platform checks, and launch the appropriate training path without a manual virtualenv step.

Cloud-only environment variables:

```bash
export AI_FACTORY_REPO_ROOT="$PWD"
export ARTIFACTS_DIR="/mnt/ai-factory-artifacts"
```

Use a writable persistent path for `ARTIFACTS_DIR` on cloud VMs.

On Apple Silicon local machines, the bootstrap script prefers the Metal-aware local path and keeps the same corpus and tokenizer automation.

If you intentionally skip Git LFS, regenerate `data/catalog.json` locally before relying on `ai-factory ready` or `ai-factory doctor`.

## 2.5 Detect and Benchmark Your Hardware (Optional)

Before training, detect your hardware capabilities and run a quick benchmark:

```bash
# Detect hardware and print optimization recommendations
python -m training.src.optimization

# Quick performance benchmark
python -c "from training.src.ultimate_harness import quick_benchmark; quick_benchmark()"
```

Expected output includes:
- Device name and backend (Metal/CUDA/CPU)
- Memory capacity and bandwidth
- Supported precision (FP16/BF16/TF32)
- Recommended batch size and settings

Use this information to choose the right training profile.

```bash
ai-factory ready
python scripts/doctor.py
ai-factory doctor
```

If you are validating on a CPU-only machine, expect `ai-factory train-preflight` to warn about CUDA visibility and FlashAttention availability for GPU-oriented profiles. Those warnings are informational until you start a real training run.

The Titan runtime now reports CUDA, Metal, and CPU fallback capability more clearly, so hardware-specific launches are easier to validate before you commit to a long run.

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

This now writes both JSONL splits and a SQLite corpus database, and the tokenizer/tokenization path is optimized to reduce waiting on larger corpora.

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
- the faster preprocessing path is designed to reuse corpus artifacts instead of rebuilding work on every run

## 5. Choose Your Training Mode

Before you pick a profile, decide which training mode you want.

### 5.1 Scratch training from zero

Use this when you want to create a brand-new model with random initialization.

This path:

- uses `model.initialization: scratch`
- resolves the architecture from `target_parameters`
- does not load pretrained weights from disk or Hugging Face

Use this profile as the starting point:

- `training/configs/profiles/pretraining.yaml`
- run the Linux cloud bootstrap script first if you are training on a rented GPU VM
- run the macOS bootstrap script first if you are validating locally on Apple Silicon

### 5.2 Continued pretraining

Use this when you already have a model checkpoint and want to continue next-token training on your own corpus.

This path:

- keeps the plain-text causal language-modeling objective
- loads an existing checkpoint first
- is not a separate built-in profile yet

The usual workflow is:

1. start from `training/configs/profiles/pretraining.yaml`
2. change `model.initialization` to `pretrained`
3. set `model.base_model_name` to your real checkpoint or model ID
4. keep the same dry-run and tokenizer-validation steps before the real run

### 5.3 Training from a base model

Use this when you want QLoRA, LoRA, or full fine-tuning on an existing model.

This path:

- uses `model.initialization: pretrained`
- loads base weights first
- applies adapter training or full fine-tuning depending on the profile

Examples:

- `training/configs/profiles/baseline_qlora.yaml`
- `training/configs/profiles/failure_aware.yaml`
- `training/configs/profiles/math_specialist.yaml`
- `training/configs/profiles/full_finetune.yaml`
- `training/configs/profiles/local_metal.yaml` for Apple Silicon local dry-runs and lighter LoRA-style iteration

Important:

- the pretrained profiles in this repo now point at local foundation checkpoints such as `artifacts/foundation/qwen2-2b`
- if those folders do not exist yet, those profiles will fail at model load
- if you do not have local checkpoints yet, either create them first or edit the profile to use the real base model you want

## 5.4 Choose Ultimate Optimization Profile (Optional)

For maximum performance, use the hardware-specific ultimate profiles:

### Apple Silicon Ultimate (M5 Max)
```bash
python -m training.train --config training/configs/profiles/m5_max_ultimate.yaml --dry-run
```
Optimizations:
- Unified memory zero-copy path
- 614 GB/s bandwidth utilization
- 40-core GPU saturation
- Fused RMSNorm+SiLU kernels

### NVIDIA A100 Ultimate
```bash
python -m training.train --config training/configs/profiles/cuda_ultimate_a100.yaml --dry-run
```
Optimizations:
- BF16 Tensor Core acceleration
- TF32 automatic mixed precision
- 2039 GB/s memory bandwidth
- FlashAttention-2 fused kernels

### NVIDIA H100 Ultimate
```bash
python -m training.train --config training/configs/profiles/cuda_ultimate_h100.yaml --dry-run
```
Optimizations:
- FP8 Transformer Engine support
- 3350 GB/s memory bandwidth
- 4th gen Tensor Cores
- Distributed training ready

See the [Optimization Guide](docs/optimization-guide.md) for tuning details.

## 6. Choose The Scratch Model Size

The scratch-model path now supports target parameter counts directly, and the canonical ladder is:

- `1b`
- `2b`
- `4b`
- `9b`
- `12b`
- `20b`
- `27b`
- `30b`
- `70b`
- `120b`

To plan a model around a parameter budget:

```bash
python training/scripts/plan_model_scale.py --target-parameters 1b
python training/scripts/plan_model_scale.py --target-parameters 2b
python training/scripts/plan_model_scale.py --target-parameters 4b
```

The canonical scratch model components now live under:

- `training/configs/components/models/qwen2_scratch_1b.yaml`
- `training/configs/components/models/qwen2_scratch_2b.yaml`
- `training/configs/components/models/qwen2_scratch_4b.yaml`
- `training/configs/components/models/qwen2_scratch_9b.yaml`
- `training/configs/components/models/qwen2_scratch_12b.yaml`
- `training/configs/components/models/qwen2_scratch_20b.yaml`
- `training/configs/components/models/qwen2_scratch_27b.yaml`
- `training/configs/components/models/qwen2_scratch_30b.yaml`
- `training/configs/components/models/qwen2_scratch_70b.yaml`
- `training/configs/components/models/qwen2_scratch_120b.yaml`

The default scratch entry remains:

- `model_type: qwen2`
- `target_parameters: 2b`
- `vocab_size: 50257`

If you want a different model size, switch to the matching template and keep the tokenizer vocab size aligned with the same component.

## 7. Train The Tokenizer For Scratch Training Or Continued Pretraining

Before the first real scratch run, train a local tokenizer that matches the configured vocab budget.

You usually need this for:

- scratch training from zero
- continued pretraining on your own corpus

You usually do not need this for:

- QLoRA / LoRA / full fine-tuning against an existing base model that already has its tokenizer

```bash
python training/scripts/train_tokenizer.py \
  --config training/configs/profiles/pretraining.yaml \
  --output-dir artifacts/tokenizers/qwen2_math_2b
```

If you want a different vocab size, update the matching `training/configs/components/models/qwen2_scratch_<size>.yaml` template first, then pass the same setting through the profile.

## 8. Validate The Right Training Path First

Always start with a dry-run that matches the mode you chose.

### 8.1 Scratch training dry-run

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

### 8.2 Base-model fine-tuning dry-run

Use this for QLoRA, LoRA, or full fine-tuning:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/full_finetune.yaml --dry-run
```

This validates:

- config composition
- dataset paths
- tokenizer loading from the configured base model
- base checkpoint resolution
- adapter or full-tuning wiring

If this fails at model load, the usual cause is that the configured `artifacts/foundation/...` checkpoint does not exist yet.

You can also run the repo refresh flow:

```bash
ai-factory refresh-lab --skip-tests --skip-notebooks --skip-generate
```

Before a real long-running launch, run the hard preflight:

```bash
ai-factory train-preflight --config training/configs/profiles/pretraining.yaml
ai-factory train-preflight --config training/configs/profiles/failure_aware.yaml
```

Important:

- scratch training now expects the local tokenizer artifact under `artifacts/tokenizers/...` for a real run
- if that tokenizer is missing, `train-preflight` will fail and tell you exactly how to build it
- dry-runs can still fall back to the configured tokenizer name so you can validate the rest of the stack first

## 9. Run The First Real Training Job

### 9.1 Scratch training from zero

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
4. `ai-factory train-preflight --config training/configs/profiles/pretraining.yaml`
5. `python -m training.train --config training/configs/profiles/pretraining.yaml --dry-run --validate-model-load`
6. `python -m training.train --config training/configs/profiles/pretraining.yaml`

### 9.2 Continued pretraining

To do continued pretraining:

1. copy `training/configs/profiles/pretraining.yaml`
2. keep the pretraining data objective
3. change `model.initialization` to `pretrained`
4. set `model.base_model_name` to your real checkpoint or model ID
5. run the same dry-run checks before the real launch

This gives you more corpus exposure on an existing model instead of a fresh random initialization.

### 9.3 Train from a base model

This is the fastest path to a specialist model if you already have a usable base checkpoint.

Recommended first command:

```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
```

Use this only after confirming the configured base checkpoint exists.

## 10. Run The First Real Fine-Tune

### Recommended first real production run

Use `failure_aware.yaml` first if you want the default finetuned evaluation and inference flows to work afterward.

```bash
python -m training.train --config training/configs/profiles/failure_aware.yaml
```

Why this profile first:

- it publishes to `artifacts/models/atlas-math-failure-aware/latest`
- the default `finetuned` model registry entry points there
- the default `base_vs_finetuned` evaluation config expects that artifact

Operator recommendation:

- run `ai-factory train-preflight --config training/configs/profiles/failure_aware.yaml` before the first real launch

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

## 11. Evaluate

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

## 12. Optimization Loop

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

## 13. Serve And Run Inference

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

## 14. Local Deployment Options

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

## 15. Cloud GPU Workflow

The exact commands stay the same on a cloud box.

Recommended flow:

```bash
tmux new -s ai-factory
bash scripts/start-linux.sh
export AI_FACTORY_REPO_ROOT="$PWD"
export ARTIFACTS_DIR="/mnt/ai-factory-artifacts"
python data/prepare_dataset.py --config data/configs/processing.yaml
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python -m training.train --config training/configs/profiles/failure_aware.yaml
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

If you already launched through the bootstrap script, the remaining commands are the same validation and training steps you would run on any Linux GPU host.

Use cloud mainly for:

- `failure_aware.yaml`
- `math_specialist.yaml`
- `verifier_augmented.yaml`
- `long_context.yaml`
- `full_finetune.yaml`

## 16. Done Means Done

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
