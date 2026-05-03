# AI-Factory Platform

## Overview
AI-Factory is a **Unified Autonomous AI Operating System** for complete LLM lifecycle management. It enables end-to-end management of large language models from data synthesis through training, evaluation, and deployment.

## Architecture Summary
- **Multi-Domain Support**: Mathematics, Code Generation, Reasoning, Creative Writing
- **Unified Interfaces**: CLI, TUI, Web Dashboard, Desktop App
- **Complete Lifecycle**: Data → Training → Evaluation → Deployment → Monitoring

## Key Components

### Core Modules (`ai_factory/`)
- `core/` - Shared foundation (schemas, artifacts, hashing, lineage)
- `domains/` - Domain-specific implementations (math, code, reasoning)
- `interfaces/` - Unified interface abstractions
- `platform/` - Platform capabilities (scaling, monitoring, deployment)
- `orchestration/` - Task orchestration and agent management

### Data Layer (`data/`)
- Adapters for multi-source integration (local, HuggingFace, S3)
- Data synthesis and quality control
- Dataset packing and lineage tracking

### Training System (`training/`)
- Fine-tuning profiles (QLoRA, full fine-tuning)
- Distributed training support
- Experiment tracking and manifests

### Evaluation (`evaluation/`)
- Benchmark registry (MMLU, HumanEval, GSM8K)
- Real-time metrics and health monitoring
- Automated failure analysis

### Inference (`inference/`)
- FastAPI backend server
- Multi-target deployment (HuggingFace, Ollama, LM Studio)
- Prompt management and model registry

### Frontend (`frontend/`)
- Next.js 14 dashboard
- Real-time WebSocket metrics streaming
- Model comparison and experiment browser

### Desktop App (`desktop/`)
- Electron wrapper with native OS integration
- Background service management
- Deep linking support

## Quick Start
```bash
# Clone and install
git clone https://github.com/super244/ai-factory.git
cd ai-factory
pip install -e ".[dev]"

# Basic commands
ai-factory domain list
ai-factory tui
ai-factory serve
```

## Documentation
- [Architecture Guide](docs/architecture.md)
- [Data System Guide](docs/data-system.md)
- [Training Guide](tune-guide.md)
- [Deployment Guide](docs/deployment-guide.md)
- [API Reference](docs/api/README.md)
- [CLI Reference](docs/cli-guide.md)

## Technology Stack
- **Backend**: Python 3.10+, PyTorch, FastAPI, Pydantic
- **Frontend**: Next.js 14, React, Tailwind CSS
- **Desktop**: Electron
- **ML**: Transformers, PEFT, bitsandbytes, TRL
- **Infrastructure**: Ray, NCCL, Docker, Celery + Redis
- **Monitoring**: Prometheus + Grafana

## Development
```bash
# Code quality
ruff check .
ruff format .
mypy ai_factory/

# Testing
pytest --cov=ai_factory

# Format
make format && make lint
```

## Community
- [Issues](https://github.com/super244/ai-factory/issues)
- [Discussions](https://github.com/super244/ai-factory/discussions)
- [Documentation](docs/)

---
*AI-Factory - Your complete AI development platform* 🚀

## Automated Maintenance Workflow
When instructed to perform codebase maintenance or at the conclusion of major feature development, Gemini CLI should automatically execute the following routine without asking for permission:
1. **Refactor & Cleanup**: Prune unnecessary configurations, stale logs, and cache directories (`__pycache__`, `.pytest_cache`, etc.).
2. **Quality Assurance**: Run `make format && make lint` and automatically resolve formatting and linting issues.
3. **Verification**: Run all tests (`pytest --no-header -v`) and fix any broken tests or ci.yml errors.
4. **Commit & Push**: Stage the verified changes, propose a clear commit message, commit, and push the updates.
