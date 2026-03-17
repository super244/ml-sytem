from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from ai_factory.core.hashing import normalize_text, stable_question_fingerprint


def _question_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=normalize_text(left).lower(), b=normalize_text(right).lower()).ratio()


def deduplicate_near_duplicates(
    records: list[dict[str, Any]],
    similarity_threshold: float = 0.94,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    report = {
        "input_rows": len(records),
        "deduped_exact": 0,
        "deduped_near": 0,
    }
    seen_exact: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (item.get("quality_score", 0.0), item.get("difficulty", "")), reverse=True):
        fingerprint = stable_question_fingerprint(record.get("question", ""))
        if fingerprint in seen_exact:
            report["deduped_exact"] += 1
            continue
        duplicate = False
        for existing in kept[-200:]:
            if _question_similarity(record.get("question", ""), existing.get("question", "")) >= similarity_threshold:
                report["deduped_near"] += 1
                duplicate = True
                break
        if duplicate:
            continue
        seen_exact[fingerprint] = record
        kept.append(record)
    return kept, report


def apply_contamination_status(
    records: list[dict[str, Any]],
    benchmark_records: list[dict[str, Any]],
    similarity_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    benchmark_pairs = [
        (stable_question_fingerprint(item.get("question", "")), normalize_text(item.get("question", "")))
        for item in benchmark_records
        if item.get("question")
    ]
    for record in records:
        question = record.get("question", "")
        fingerprint = stable_question_fingerprint(question)
        exact_match = any(candidate_fp == fingerprint for candidate_fp, _ in benchmark_pairs)
        max_similarity = 0.0
        if not exact_match:
            for _, other_question in benchmark_pairs[:500]:
                max_similarity = max(max_similarity, _question_similarity(question, other_question))
                if max_similarity >= similarity_threshold:
                    break
        record["contamination"] = {
            "checked_against": ["benchmark_holdout"],
            "exact_match": exact_match,
            "near_match": max_similarity >= similarity_threshold,
            "max_similarity": round(max_similarity, 4),
            "notes": ["auto_contamination_check"],
        }
    return records
