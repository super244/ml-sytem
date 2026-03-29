from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_factory.core.io import read_jsonl, write_jsonl
from data.quality.mining import select_failure_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine training examples from model failure logs.")
    parser.add_argument("--input", required=True, help="Evaluation JSONL with per-example outputs.")
    parser.add_argument("--output", required=True, help="Output JSONL for curated failure cases.")
    parser.add_argument("--min-difficulty", default="hard")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input))
    curated = select_failure_cases(records, min_difficulty=args.min_difficulty, limit=args.limit)
    write_jsonl(Path(args.output), curated)
    print(json.dumps({"curated": len(curated), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
