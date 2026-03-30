from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def compute_record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    difficulty_counts = Counter(record.get("difficulty", "unknown") for record in records)
    topic_counts = Counter(record.get("topic", "unknown") for record in records)
    source_counts = Counter(record.get("source", "unknown") for record in records)
    pack_counts = Counter(record.get("pack_id", "unknown") for record in records)
    reasoning_counts = Counter(record.get("reasoning_style", "mixed") for record in records)
    quality_scores = [float(record.get("quality_score", 0.0) or 0.0) for record in records]
    return {
        "num_records": len(records),
        "difficulty_counts": dict(difficulty_counts),
        "topic_counts": dict(topic_counts),
        "source_counts": dict(source_counts),
        "pack_counts": dict(pack_counts),
        "reasoning_style_counts": dict(reasoning_counts),
        "failure_case_count": sum(1 for record in records if record.get("failure_case")),
        "verification_ready_count": sum(1 for record in records if record.get("step_checks")),
        "avg_quality_score": round(mean(quality_scores), 4) if quality_scores else 0.0,
        "contaminated_count": sum(
            1
            for record in records
            if (record.get("contamination") or {}).get("exact_match")
            or (record.get("contamination") or {}).get("near_match")
        ),
    }
