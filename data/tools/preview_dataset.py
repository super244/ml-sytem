from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a few records from a dataset JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))[: args.limit]
    preview = [
        {
            "id": row.get("id"),
            "topic": row.get("topic"),
            "difficulty": row.get("difficulty"),
            "question": row.get("question"),
            "final_answer": row.get("final_answer"),
        }
        for row in rows
    ]
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
