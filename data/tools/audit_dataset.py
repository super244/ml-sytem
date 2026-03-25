from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl, write_json
from data.quality.stats import compute_record_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quality and coverage audit for a dataset JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    stats = compute_record_stats(rows)
    write_json(args.output, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
