#!/usr/bin/env bash
set -euo pipefail

python3 -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
python3 -m training.train --config training/configs/profiles/calculus_specialist.yaml --dry-run
python3 -m training.train --config training/configs/profiles/curriculum_specialist.yaml --dry-run
python3 -m training.train --config training/configs/profiles/failure_aware.yaml --dry-run
python3 -m training.train --config training/configs/profiles/verifier_augmented.yaml --dry-run
python3 -m training.train --config training/configs/profiles/long_context.yaml --dry-run
python3 -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml --dry-run
