#!/usr/bin/env python3
"""
Removes caches and build artifacts that are not part of the active training run.

Only removes the directories that are explicitly listed below, nothing under
`artifacts/runs` or the dataset folders that hold the current training/eval data.
"""

from pathlib import Path
import shutil

TARGETS = [
    Path("cache"),
    Path("data/processed/.tokenized_cache"),
]


def safe_remove(target: Path) -> None:
    if not target.exists():
        print(f"Skipping {target} (does not exist).")
        return
    if not target.is_dir():
        print(f"Skipping {target} (not a directory).")
        return
    shutil.rmtree(target)
    print(f"Removed {target}.")


def main() -> None:
    print("Cleaning caches that are safe to drop before a new final run.")
    for target in TARGETS:
        safe_remove(target)
    print("Cleanup complete. Run `python generate_visualizations.py` to refresh the charts.")


if __name__ == "__main__":
    main()
