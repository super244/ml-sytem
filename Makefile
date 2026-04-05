PYTHON ?= python
COVERAGE_ARGS = --cov=ai_factory --cov=data --cov=training --cov=evaluation --cov=inference --cov-report=term-missing --cov-fail-under=55

# Core Development
.PHONY: install test lint format check clean doctor serve help

# Data Operations
.PHONY: data-generate data-prepare data-validate data-audit data-preview

# Training Operations
.PHONY: train train-dry validate-model

# Evaluation Operations
.PHONY: evaluate analyze-failures

# Frontend Operations
.PHONY: frontend-install frontend-dev frontend-build frontend-check

# System Operations
.PHONY: docker-up docker-down smoke

help:
	@echo "AI-Factory Developer Commands"
	@echo "============================="
	@echo "Setup:      make install          Install Python + dev deps"
	@echo "Quality:    make lint             ruff check + mypy"
	@echo "            make format           Auto-format with ruff"
	@echo "            make check            format-check + lint + test"
	@echo "Testing:    make test             pytest with coverage"
	@echo "Health:     make doctor           10-point system health check"
	@echo "            make smoke            Compile all modules + notebooks"
	@echo "Server:     make serve            Start API server (dev mode)"
	@echo "Frontend:   make frontend-dev     Start Next.js dev server"
	@echo "            make frontend-check   Typecheck + build frontend"
	@echo "Docker:     make docker-up        Start all services"
	@echo "            make docker-down      Stop all services"
	@echo "Data:       make data-generate    Generate synthetic datasets"
	@echo "            make data-prepare     Normalize + pack datasets"
	@echo "Training:   make train-dry        Dry-run training validation"
	@echo "            make train            Run full training"
	@echo "Eval:       make evaluate         Run evaluation pipeline"

# Development Setup
install:
	pip install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

# Code Quality
test:
	$(PYTHON) -m pytest $(COVERAGE_ARGS)

lint:
	ruff check .
	mypy .

format:
	ruff format .

check:
	ruff format --check .
	ruff check .
	mypy .
	$(PYTHON) -m pytest $(COVERAGE_ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/
	cd frontend && rm -rf node_modules/.cache

# System Health
doctor:
	$(PYTHON) scripts/doctor.py

smoke:
	$(PYTHON) -m compileall ai_factory data training inference evaluation
	$(PYTHON) notebooks/build_notebooks.py

serve:
	uvicorn inference.app.main:app --host 0.0.0.0 --port 8000 --reload

# Data Pipeline
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

# Training Pipeline
train:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml

train-dry:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run

validate-model:
	$(PYTHON) -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run --validate-model-load

# Evaluation Pipeline
evaluate:
	$(PYTHON) -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml

analyze-failures:
	$(PYTHON) evaluation/analysis/analyze_failures.py --input evaluation/results/latest/per_example.jsonl --output evaluation/results/latest/failure_analysis.json

# Frontend Development
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-check:
	cd frontend && npm run typecheck
	cd frontend && npm run build

# Docker Operations
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Legacy Aliases (for backward compatibility)
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
	$(PYTHON) data/tools/export_subset.py --input data/processed/normalized_all.jsonl --output data/processed/calculus_hard_preview.jsonl --topic calculus --difficulty hard --limit 64

dedupe-near:
	$(PYTHON) data/tools/deduplicate_simhash.py --input data/processed/normalized_all.jsonl --output data/processed/normalized_all.dedup.jsonl

benchmark-pack:
	$(PYTHON) data/tools/build_benchmark_pack.py --input data/processed/normalized_all.jsonl --output-dir data/processed/packs

mine-failures:
	$(PYTHON) data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl --output data/raw/failure_cases.jsonl

notebooks:
	$(PYTHON) notebooks/build_notebooks.py

frontend-typecheck:
	cd frontend && npm run typecheck
