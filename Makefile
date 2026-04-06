PYTHON ?= python
COVERAGE_ARGS = --cov=ai_factory --cov=data --cov=training --cov=evaluation --cov=inference \
                --cov-report=term-missing --cov-fail-under=55

# ── Phony target declarations ──────────────────────────────────────────────────
.PHONY: install install-full test test-fast test-parallel lint format check clean doctor smoke
.PHONY: serve serve-prod
.PHONY: data-generate data-prepare data-validate data-audit data-preview
.PHONY: train train-dry train-preflight validate-model
.PHONY: evaluate analyze-failures
.PHONY: frontend-install frontend-dev frontend-build frontend-check frontend-typecheck
.PHONY: desktop-start
.PHONY: docker-up docker-down docker-build docker-logs
.PHONY: rust-check rust-test rust-clippy rust-bench rust-build
.PHONY: optimize optimize-detect optimize-benchmark optimize-profile
.PHONY: generate-datasets prepare-data validate-data audit-data preview-data
.PHONY: refresh-lab latest-run api-smoke titan-status titan-doc notebooks
.PHONY: export-subset dedupe-near benchmark-pack mine-failures

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "AI-Factory v0.3.0 — Developer Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install Python + dev deps (Python 3.11+)"
	@echo "  make install-full      + flash-attn (Linux CUDA only)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint              ruff check + mypy"
	@echo "  make format            Auto-format with ruff"
	@echo "  make check             format-check + lint + test"
	@echo ""
	@echo "Testing:"
	@echo "  make test              pytest with coverage (sequential)"
	@echo "  make test-fast         pytest, skip slow / integration"
	@echo "  make test-parallel     pytest -n auto (parallel)"
	@echo ""
	@echo "Health:"
	@echo "  make doctor            10-point system health check"
	@echo "  make smoke             Compile all modules + notebooks"
	@echo ""
	@echo "Server:"
	@echo "  make serve             Start API server (dev mode, hot-reload)"
	@echo "  make serve-prod        Start API server (production, 4 workers)"
	@echo ""
	@echo "Frontend:"
	@echo "  make frontend-dev      Start Next.js dev server"
	@echo "  make frontend-check    Typecheck + build frontend"
	@echo "  make frontend-build    Production build"
	@echo ""
	@echo "Rust / Titan:"
	@echo "  make rust-check        cargo check (default + cpp features)"
	@echo "  make rust-test         cargo test"
	@echo "  make rust-clippy       cargo clippy (deny warnings)"
	@echo "  make rust-bench        cargo bench (Criterion)"
	@echo "  make rust-build        cargo build --release"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up         Start all services"
	@echo "  make docker-down       Stop all services"
	@echo "  make docker-build      Rebuild images"
	@echo "  make docker-logs       Tail service logs"
	@echo ""
	@echo "Data:"
	@echo "  make data-generate     Generate synthetic datasets"
	@echo "  make data-prepare      Normalize + pack datasets"
	@echo ""
	@echo "Training:"
	@echo "  make train-dry         Dry-run training validation"
	@echo "  make train-preflight   Hard fail preflight before a real run"
	@echo "  make train             Run full training"
	@echo ""
	@echo "Optimization:"
	@echo "  make optimize-detect   Detect hardware capabilities"
	@echo "  make optimize-bench    Run performance benchmark"
	@echo "  make optimize-profile  Show recommended optimization profile"
	@echo ""
	@echo "Eval:"
	@echo "  make evaluate          Run evaluation pipeline"

# ── Development Setup ─────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

install-full:
	pip install -e ".[dev,train-cuda]"

# ── Code Quality ──────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest $(COVERAGE_ARGS)

test-fast:
	$(PYTHON) -m pytest $(COVERAGE_ARGS) -m "not slow"

test-parallel:
	$(PYTHON) -m pytest $(COVERAGE_ARGS) -n auto

lint:
	ruff check .
	mypy .

format:
	ruff format .
	ruff check --fix .

check:
	ruff format --check .
	ruff check .
	mypy .
	$(PYTHON) -m pytest $(COVERAGE_ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/
	cd frontend && rm -rf node_modules/.cache .next/cache || true

# ── System Health ─────────────────────────────────────────────────────────────
doctor:
	$(PYTHON) scripts/doctor.py

smoke:
	$(PYTHON) -m compileall ai_factory data training inference evaluation
	$(PYTHON) notebooks/build_notebooks.py

# ── Server ────────────────────────────────────────────────────────────────────
serve:
	uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop

# ── Rust / Titan Engine ───────────────────────────────────────────────────────
rust-check:
	cargo check --manifest-path ai_factory_titan/Cargo.toml
	cargo check --manifest-path ai_factory_titan/Cargo.toml --features cpp

rust-test:
	cargo test --manifest-path ai_factory_titan/Cargo.toml --all-targets

rust-clippy:
	cargo clippy --manifest-path ai_factory_titan/Cargo.toml --all-targets -- -D warnings

rust-bench:
	cargo bench --manifest-path ai_factory_titan/Cargo.toml

rust-build:
	cargo build --manifest-path ai_factory_titan/Cargo.toml --release --features cpp

# ── Data Pipeline ─────────────────────────────────────────────────────────────
data-generate:
	$(PYTHON) data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml

data-prepare:
	$(PYTHON) data/prepare_dataset.py --config data/configs/processing.yaml

data-validate:
	$(PYTHON) data/tools/validate_dataset.py --input data/processed/*.jsonl --manifest data/processed/manifest.json

data-audit:
	$(PYTHON) data/tools/audit_dataset.py --input data/processed/normalized_all.jsonl --output data/processed/audit.json

data-preview:
	$(PYTHON) data/tools/preview_dataset.py --input data/processed/normalized_all.jsonl --limit 5

# ── Training Pipeline ─────────────────────────────────────────────────────────
train:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml

train-dry:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run

train-preflight:
	$(PYTHON) -m ai_factory.cli train-preflight --config training/configs/profiles/failure_aware.yaml

validate-model:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run --validate-model-load

# ── Evaluation Pipeline ───────────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml

analyze-failures:
	$(PYTHON) evaluation/analysis/analyze_failures.py \
		--input evaluation/results/latest/per_example.jsonl \
		--output evaluation/results/latest/failure_analysis.json

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-check:
	cd frontend && npm run typecheck && npm run build

frontend-typecheck:
	cd frontend && npm run typecheck

# ── Desktop ───────────────────────────────────────────────────────────────────
desktop-start:
	cd desktop && npm start

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

# ── Optimization ─────────────────────────────────────────────────────────────
optimize-detect:
	$(PYTHON) -m ai_factory.cli optimize detect

optimize-benchmark optimize-bench:
	$(PYTHON) -m ai_factory.cli optimize benchmark

optimize-profile:
	$(PYTHON) -m ai_factory.cli optimize profile

# ── Legacy aliases ────────────────────────────────────────────────────────────
refresh-lab:
	$(PYTHON) -m ai_factory.cli refresh-lab

latest-run:
	$(PYTHON) -m ai_factory.cli latest-run

api-smoke:
	$(PYTHON) -m ai_factory.cli api-smoke

titan-status:
	$(PYTHON) -m ai_factory.cli titan status

titan-doc:
	$(PYTHON) -m ai_factory.cli titan hardware-doc

generate-datasets: data-generate

prepare-data: data-prepare

validate-data: data-validate

audit-data: data-audit

preview-data: data-preview

export-subset:
	$(PYTHON) data/tools/export_subset.py \
		--input data/processed/normalized_all.jsonl \
		--output data/processed/calculus_hard_preview.jsonl \
		--topic calculus --difficulty hard --limit 64

dedupe-near:
	$(PYTHON) data/tools/deduplicate_simhash.py \
		--input data/processed/normalized_all.jsonl \
		--output data/processed/normalized_all.dedup.jsonl

benchmark-pack:
	$(PYTHON) data/tools/build_benchmark_pack.py \
		--input data/processed/normalized_all.jsonl \
		--output-dir data/processed/packs

mine-failures:
	$(PYTHON) data/mine_failure_cases.py \
		--input evaluation/results/latest/per_example.jsonl \
		--output data/raw/failure_cases.jsonl

notebooks:
	$(PYTHON) notebooks/build_notebooks.py
