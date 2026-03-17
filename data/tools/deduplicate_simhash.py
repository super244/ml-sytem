from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl, write_jsonl
from data.quality.contamination import deduplicate_near_duplicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Near-duplicate dedupe for dataset JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.94)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = read_jsonl(in_path)
    kept, report = deduplicate_near_duplicates(rows, similarity_threshold=args.similarity_threshold)
    write_jsonl(out_path, kept)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
