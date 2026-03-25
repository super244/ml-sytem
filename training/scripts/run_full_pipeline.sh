#!/usr/bin/env bash
set -euo pipefail

python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python3 data/prepare_dataset.py --config data/configs/processing.yaml
python3 -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run
python3 -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
