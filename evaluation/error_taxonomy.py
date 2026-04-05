from __future__ import annotations

from collections import Counter
from typing import Any


def cluster_failures(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        entry = result.get(key)
        if not isinstance(entry, dict):
            continue
        if entry.get("correct"):
            continue
        counter[entry.get("error_type", "unknown")] += 1
    return dict(counter)


def summarize_failure_taxonomy(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary": cluster_failures(results, "primary"),
        "secondary": cluster_failures(results, "secondary"),
    }
