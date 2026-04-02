from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run LoRA, QLoRA, and full fine-tuning workflows together.")
    parser.add_argument("--run-prefix", default="finetune-suite")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for method in ("lora", "qlora", "full"):
        command = [
            sys.executable,
            "training/scripts/run_supervised_workflow.py",
            "--method",
            method,
            "--run-name",
            f"{args.run_prefix}-{method}",
            *args.passthrough,
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

