#!/usr/bin/env bash
set -euo pipefail

python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python3 -m training.train --config training/configs/profiles/math_specialist.yaml --dry-run
python3 -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run
python3 -m training.train --config training/configs/profiles/verifier_augmented.yaml --dry-run
python3 -m training.train --config training/configs/profiles/long_context.yaml --dry-run
python3 -m training.train --config training/configs/profiles/fast_dev.yaml --dry-run
python3 -m training.train --config training/configs/profiles/full_finetune.yaml --dry-run
