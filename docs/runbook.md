# Runbook

This runbook describes the default local workflow for Atlas Math Lab. It assumes local development with optional public-dataset access and a runnable math model such as `Qwen2.5-Math-1.5B-Instruct`.

## 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

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

## 6. Train A Specialist Profile

Examples:

```bash
python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml
python3 -m training.train --config training/configs/profiles/calculus_specialist.yaml
python3 -m training.train --config training/configs/profiles/failure_aware.yaml
```

Training runs write standardized artifacts under `artifacts/runs/<run_id>/` and publish packaged model assets under `artifacts/models/<name>/`.

## 7. Serve The API

```bash
uvicorn inference.app.main:app --reload
```

Primary metadata routes:

- `GET /v1/health`
- `GET /v1/status`
- `GET /v1/models`
- `GET /v1/prompts`
- `GET /v1/datasets`
- `GET /v1/benchmarks`
- `GET /v1/runs`

Primary generation routes:

- `POST /v1/generate`
- `POST /v1/generate/batch`
- `POST /v1/compare`
- `POST /v1/verify`

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
