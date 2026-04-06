# Runbook

This runbook describes the default local workflow for AI-Factory. It assumes local development with optional public-dataset access and a configured math model entry from `inference/configs/model_registry.yaml` or a local scratch template.

## 1. Install Dependencies

```bash
bash scripts/start_cloud_linux.sh
```

Use `bash scripts/start_local_macos.sh` on Apple Silicon local machines. Both bootstrap scripts install the runtime, fetch dependencies, and prepare the training entry points without a manual virtual environment step.

## 2. Generate Local Synthetic Packs

```bash
python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
```

This builds six custom families, emits per-dataset manifests and cards, and refreshes `data/catalog.json`.

## 3. Normalize Public Datasets

If you have network access and `datasets` installed:

```bash
python3 data/public/normalize_public_datasets.py --registry data/public/registry.yaml
```

Each normalized public dataset writes:

- `*.jsonl`
- `*.manifest.json`
- `*.md`

under `data/public/normalized/`.

## 4. Build The Processed Corpus

```bash
python3 data/prepare_dataset.py --config data/configs/processing.yaml
```

The processing pipeline now spends less time in the tokenizer and dataset assembly stages, so large corpora should move through this step faster than before.

Outputs in `data/processed/`:

- `normalized_all.jsonl`
- `train.jsonl`
- `eval.jsonl`
- `test.jsonl`
- `stats.json`
- `manifest.json`
- `card.md`
- `size_report.md`
- `pack_summary.json`
- `packs/<pack_id>/records.jsonl`
- `packs/<pack_id>/manifest.json`
- `packs/<pack_id>/card.md`

If you want a tokenization-aware quick look at the corpus, run:

```bash
python3 data/tools/preview_dataset.py --input data/processed/train.jsonl --tokenizer artifacts/tokenizers/qwen2_math_2b
```

## 5. Validate Training Configuration

Use dry-run mode before loading weights or starting a longer training job:

```bash
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
```

Dry-run validates:

- config composition
- dataset manifests and paths
- prompt rendering
- tokenizer wiring
- artifact directory creation
- the hardware-aware bootstrap assumptions used by the cloud Linux and macOS local start scripts

## 6. Train A Specialist Profile

Examples:

```bash
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml
python3 -m training.train --config training/configs/profiles/math_specialist.yaml
python3 -m training.train --config training/configs/profiles/failure_aware.yaml
```

Training runs write standardized artifacts under `artifacts/runs/<run_id>/` and publish packaged model assets under `artifacts/models/<name>/`.

For fresh runs on cloud GPU hosts, start from the Linux bootstrap script so the CUDA dependencies, tokenizer artifacts, and launch environment are all aligned before training begins. For Apple Silicon local runs, use the macOS bootstrap script so the same workflow lands on the Metal-aware local path.

Managed control-plane alternative:

```bash
ai-factory new --config configs/finetune.yaml
ai-factory list
ai-factory status <instance-id> --json
ai-factory tasks <instance-id> --json
ai-factory events <instance-id> --json
ai-factory watch <instance-id> --timeout 30 --json
ai-factory recommendations <instance-id> --json
ai-factory children <instance-id> --json
```

The managed path tracks progress, metrics summaries, recommendations, retries, heartbeats, and parent/child follow-up instances under `artifacts/instances/<instance-id>/`, with durable run/task/attempt state in `artifacts/control_plane/control_plane.db`.

## 7. Serve The API

```bash
uvicorn inference.app.main:app --reload
```

Primary metadata routes:

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `GET /v1/prompts`
- `GET /v1/datasets`
- `GET /v1/benchmarks`
- `GET /v1/runs`

Primary generation routes:

- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`
- `POST /v1/chat/completions`
- `GET /v1/usage`

Legacy aliases such as `/generate` and `/verify` remain available.

## 8. Run The Frontend

```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Available routes:

- `/`
- `/compare`
- `/datasets`
- `/benchmarks`
- `/runs`

## 9. Evaluate Benchmarks

```bash
python3 -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
python3 -m evaluation.evaluate --config evaluation/configs/verifier_on_off.yaml
```

Managed alternative:

```bash
ai-factory evaluate <instance-id> --config configs/eval.yaml
```

Evaluation outputs include:

- `summary.json`
- `summary.md`
- `leaderboard.json`
- `per_example.jsonl`

Optional failure analysis:

```bash
python3 evaluation/analysis/analyze_failures.py \
  --input evaluation/results/latest/per_example.jsonl \
  --output evaluation/results/latest/failure_analysis.json
```

The managed report template at `configs/report.yaml` is also used by the orchestration feedback loop when an evaluation run queues failure analysis automatically.

## 10. Replay Failures Into The Next Cycle

```bash
python3 data/mine_failure_cases.py \
  --input evaluation/results/latest/per_example.jsonl \
  --output data/raw/failure_cases.jsonl
```

Then add the resulting log to `failure_logs` in `data/configs/processing.yaml`, rebuild the processed corpus, and retrain with `failure_aware` or `verifier_augmented`.

## 11. Refresh The Notebook Lab

```bash
python3 notebooks/build_notebooks.py
```

## 12. Optional Container Workflow

```bash
docker compose up --build
```

Use `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml` for a lightweight demo stack. See `docs/deployment-guide.md` for environment and artifact-mount details.

## 13. Optional Cloud SSH Workflow

Create or reuse a cloud profile and launch a managed remote run:

```bash
ai-factory new --config configs/finetune.yaml --environment cloud --cloud-profile default
```

Cloud manifests store the resolved SSH profile, key path, and any configured port forwards so the same instance can be inspected through the CLI or API later.

## 14. Inspect Orchestration State

The additive orchestration API surface preserves the existing instance routes and adds:

- `GET /v1/orchestration/runs`
- `GET /v1/orchestration/runs/{run_id}`
- `GET /v1/orchestration/runs/{run_id}/tasks`
- `GET /v1/orchestration/runs/{run_id}/events`
- `POST /v1/orchestration/runs/{run_id}/cancel`
- `POST /v1/orchestration/tasks/{task_id}/retry`
- `GET /v1/orchestration/summary`

You can attach port forwards directly from the CLI when creating the instance:

```bash
ai-factory new \
  --config configs/finetune.yaml \
  --environment cloud \
  --cloud-profile default \
  --port-forward 6006:6006
```
