#!/usr/bin/env bash
set -euo pipefail

python -m training.train --config training/configs/profiles/failure_aware.yaml
