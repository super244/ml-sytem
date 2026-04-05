from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from ai_factory.core.io import read_jsonl, write_json
from evaluation.error_taxonomy import summarize_failure_taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze evaluation failures by error taxonomy.")
    parser.add_argument("--input", default="evaluation/results/latest/per_example.jsonl")
    parser.add_argument("--output", default="evaluation/results/latest/failure_analysis.json")
    return parser.parse_args()


def _group_failure_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = str(record.get(field) or "unknown")
        counts[value] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_failure_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = summarize_failure_taxonomy(records)
    failure_examples = [
        {
            "id": record.get("id"),
            "question": record.get("question"),
            "topic": record.get("topic", "unknown"),
            "difficulty": record.get("difficulty", "unknown"),
            "source": record.get("source", "unknown"),
            "pack_id": record.get("pack_id", "unknown"),
            "primary_error_type": ((record.get("primary") or {}).get("error_type")),
            "secondary_error_type": ((record.get("secondary") or {}).get("error_type")),
        }
        for record in records
        if not ((record.get("primary") or {}).get("correct") and (record.get("secondary") or {}).get("correct"))
    ]
    return {
        "num_examples": len(records),
        "taxonomy": taxonomy,
        "by_topic": _group_failure_counts(records, "topic"),
        "by_difficulty": _group_failure_counts(records, "difficulty"),
        "by_source": _group_failure_counts(records, "source"),
        "by_pack": _group_failure_counts(records, "pack_id"),
        "failure_examples": failure_examples[:25],
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    analysis = build_failure_analysis(records)
    write_json(args.output, analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
