# AI-Factory Setup & Tuning Guide

## Quick Start

Below are exact setup steps for a clean machine.

### 1. Clone + Python Environment
```bash
git clone https://github.com/super244/ai-factory.git
cd ai-factory

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
```

### 2. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 3. Verify Installation
```bash
ai-factory --help
ai-factory doctor
```

## Training Configuration

### Basic Fine-tuning
```bash
ai-factory new --config configs/finetune.yaml
```

### Custom Training
```bash
ai-factory new --config training/configs/profiles/baseline_qlora.yaml
```

### Multi-Domain Training
```bash
ai-factory multi-train --domains mathematics code reasoning
```

## Data Preparation

### Generate Datasets
```bash
make generate-datasets
```

### Process and Validate
```bash
make prepare-data
make validate-data
```

## Evaluation

### Run Evaluation
```bash
ai-factory evaluate <instance-id>
```

### Custom Evaluation
```bash
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

## Development

### Code Quality
```bash
make lint
make format
make test
```

### Frontend Development
```bash
make frontend-dev
```

### API Server
```bash
make serve
```

## Troubleshooting

### Common Issues
1. **CUDA Memory**: Reduce batch size in training config
2. **Import Errors**: Run `pip install -e ".[dev]"`
3. **Permission Denied**: Check file permissions in artifacts directory

### Get Help
```bash
ai-factory doctor
ai-factory workspace --ready
```

## Advanced Configuration

### Custom Profiles
Create new profiles in `training/configs/profiles/` following the existing pattern.

### Domain Extensions
Add new domains in `ai_factory/domains/` and register them in the factory.

### Deployment Targets
Configure deployment in `configs/deploy.yaml` for HuggingFace, Ollama, or custom APIs.