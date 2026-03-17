from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a filtered subset from a dataset JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    filtered = [
        row
        for row in rows
        if (args.topic is None or row.get("topic") == args.topic)
        and (args.difficulty is None or row.get("difficulty") == args.difficulty)
    ]
    if args.limit:
        filtered = filtered[: args.limit]
    write_jsonl(Path(args.output), filtered)
    print(f"Exported {len(filtered)} rows to {args.output}")


if __name__ == "__main__":
    main()
