from __future__ import annotations

from typing import Any


def cluster_failures(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        block = row.get(key) or {}
        if block.get("correct"):
            continue
        error_type = str(block.get("error_type") or "unknown")
        counts[error_type] = counts.get(error_type, 0) + 1
    return counts


def summarize_failure_taxonomy(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in results:
        for column in ("primary", "secondary"):
            if column not in row:
                continue
            block = row[column] or {}
            if block.get("correct"):
                continue
            error_type = str(block.get("error_type") or "unknown")
            summary.setdefault(column, {})
            summary[column][error_type] = summary[column].get(error_type, 0) + 1
    return summary
