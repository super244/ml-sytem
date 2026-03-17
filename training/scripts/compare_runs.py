from __future__ import annotations

import argparse
import json

from training.src.comparison import compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two training run directories.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(compare_runs(args.left, args.right), indent=2))


if __name__ == "__main__":
    main()
