from __future__ import annotations

import argparse
import json

from ai_factory.core.io import read_jsonl, write_json
from evaluation.error_taxonomy import summarize_failure_taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze evaluation failures by error taxonomy.")
    parser.add_argument("--input", default="evaluation/results/latest/per_example.jsonl")
    parser.add_argument("--output", default="evaluation/results/latest/failure_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    taxonomy = summarize_failure_taxonomy(records)
    write_json(args.output, taxonomy)
    print(json.dumps(taxonomy, indent=2))


if __name__ == "__main__":
    main()
