from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shortcut wrapper for specialized accuracy fine-tuning.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [sys.executable, "training/train.py", "--config", "training/configs/profiles/finetune_specialized_accuracy.yaml", *args.passthrough]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
