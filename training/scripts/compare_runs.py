from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.src.comparison import compare_runs, format_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two training run directories.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--primary-metric", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_runs(args.left, args.right, primary_metric=args.primary_metric)
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_comparison_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
