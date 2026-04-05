#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-training/configs/profiles/failure_aware.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-evaluation/configs/base_vs_finetuned.yaml}"
DRY_RUN="${AI_FACTORY_DRY_RUN:-0}"

python3 data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python3 data/prepare_dataset.py --config data/configs/processing.yaml
if [[ "${DRY_RUN}" == "1" ]]; then
  python3 -m training.train --config "${PROFILE}" --dry-run
  echo "Skipped evaluation because AI_FACTORY_DRY_RUN=1 does not publish an adapter for finetuned evaluation."
  exit 0
fi

python3 -m training.train --config "${PROFILE}"
python3 -m evaluation.evaluate --config "${EVAL_CONFIG}"
