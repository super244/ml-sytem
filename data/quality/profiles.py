from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ai_factory.core.hashing import normalize_text, stable_question_fingerprint


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 4)
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return round(ordered[lower], 4)
    weight = position - lower
    interpolated = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(interpolated, 4)


def _top_items(counter: Counter[str], *, limit: int = 5) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def build_dataset_profile(
    records: list[dict[str, Any]],
    *,
    title: str | None = None,
    source_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    difficulty_counts = Counter(str(record.get("difficulty", "unknown")) for record in records)
    topic_counts = Counter(str(record.get("topic", "unknown")) for record in records)
    source_counts = Counter(str(record.get("source", "unknown")) for record in records)
    reasoning_counts = Counter(str(record.get("reasoning_style", "mixed")) for record in records)
    split_counts = Counter(str(record.get("dataset_split", "unspecified")) for record in records)
    quality_scores = [_safe_float(record.get("quality_score"), 0.0) for record in records]
    fingerprint_counts = Counter(stable_question_fingerprint(record.get("question", "")) for record in records)
    duplicate_fingerprints = sum(1 for count in fingerprint_counts.values() if count > 1)

    dominant_topic, dominant_topic_count = topic_counts.most_common(1)[0] if topic_counts else ("unknown", 0)
    dominant_source, dominant_source_count = source_counts.most_common(1)[0] if source_counts else ("unknown", 0)
    dominant_difficulty, dominant_difficulty_count = (
        difficulty_counts.most_common(1)[0] if difficulty_counts else ("unknown", 0)
    )

    quality_summary = {
        "mean": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
        "min": round(min(quality_scores), 4) if quality_scores else 0.0,
        "max": round(max(quality_scores), 4) if quality_scores else 0.0,
        "p10": _quantile(quality_scores, 0.1),
        "p50": _quantile(quality_scores, 0.5),
        "p90": _quantile(quality_scores, 0.9),
    }
    coverage_summary = {
        "failure_case_count": sum(1 for record in records if record.get("failure_case")),
        "verification_ready_count": sum(1 for record in records if record.get("step_checks")),
        "contaminated_count": sum(
            1
            for record in records
            if (record.get("contamination") or {}).get("exact_match")
            or (record.get("contamination") or {}).get("near_match")
        ),
    }
    profile = {
        "title": title,
        "num_records": len(records),
        "unique_sources": len(source_counts),
        "unique_topics": len(topic_counts),
        "unique_difficulties": len(difficulty_counts),
        "unique_reasoning_styles": len(reasoning_counts),
        "duplicate_question_groups": duplicate_fingerprints,
        "difficulty_counts": dict(difficulty_counts),
        "topic_counts": dict(topic_counts),
        "source_counts": dict(source_counts),
        "reasoning_style_counts": dict(reasoning_counts),
        "split_counts": dict(split_counts),
        "dominant_topic": {
            "value": dominant_topic,
            "count": dominant_topic_count,
        },
        "dominant_source": {
            "value": dominant_source,
            "count": dominant_source_count,
        },
        "dominant_difficulty": {
            "value": dominant_difficulty,
            "count": dominant_difficulty_count,
        },
        "quality_summary": quality_summary,
        "coverage_summary": coverage_summary,
        "top_topics": _top_items(topic_counts),
        "top_sources": _top_items(source_counts),
        "top_difficulties": _top_items(difficulty_counts),
    }
    if source_summaries is not None:
        profile["source_summaries"] = source_summaries
    return profile


def build_source_conflict_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source = str(record.get("source", "unknown"))
        fingerprint = stable_question_fingerprint(record.get("question", ""))
        grouped_by_fingerprint[fingerprint].append(
            {
                "id": record.get("id"),
                "source": source,
                "topic": record.get("topic"),
                "difficulty": record.get("difficulty"),
                "dataset_split": record.get("dataset_split"),
                "quality_score": _safe_float(record.get("quality_score"), 0.0),
            }
        )
        record_id = str(record.get("id") or "")
        if record_id:
            grouped_by_id[record_id].append(
                {
                    "source": source,
                    "question_fingerprint": fingerprint,
                    "dataset_split": record.get("dataset_split"),
                }
            )

    exact_question_conflicts = []
    for fingerprint, entries in grouped_by_fingerprint.items():
        sources = sorted({entry["source"] for entry in entries})
        if len(sources) < 2:
            continue
        exact_question_conflicts.append(
            {
                "question_fingerprint": fingerprint,
                "record_count": len(entries),
                "sources": sources,
                "records": entries,
                "best_record": max(entries, key=lambda item: item.get("quality_score", 0.0)),
            }
        )

    id_conflicts = []
    for record_id, entries in grouped_by_id.items():
        sources = sorted({entry["source"] for entry in entries})
        if len(sources) < 2:
            continue
        id_conflicts.append(
            {
                "record_id": record_id,
                "record_count": len(entries),
                "sources": sources,
                "records": entries,
            }
        )

    exact_question_conflicts.sort(key=lambda item: (item["record_count"], item["question_fingerprint"]), reverse=True)
    id_conflicts.sort(key=lambda item: (item["record_count"], item["record_id"]), reverse=True)
    return {
        "total_records": len(records),
        "unique_questions": len(grouped_by_fingerprint),
        "exact_question_conflicts": exact_question_conflicts,
        "id_conflicts": id_conflicts,
        "cross_source_conflict_count": len(exact_question_conflicts),
    }


def build_validation_summary(issues: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts = Counter(str(issue.get("stage", "unknown")) for issue in issues)
    source_counts = Counter(str(issue.get("source", "unknown")) for issue in issues)
    reason_counts = Counter(str(issue.get("reason", "unknown")) for issue in issues)
    return {
        "total_issues": len(issues),
        "stage_counts": dict(stage_counts),
        "source_counts": dict(source_counts),
        "reason_counts": dict(reason_counts),
        "sample_issues": issues[:50],
    }


def make_validation_issue(
    *,
    stage: str,
    source: str,
    record_id: str | None,
    reason: str,
    message: str,
    question: str | None = None,
    topic: str | None = None,
    difficulty: str | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "stage": stage,
        "source": source,
        "reason": reason,
        "message": message,
    }
    if record_id is not None:
        issue["record_id"] = record_id
    if question:
        issue["question_fingerprint"] = stable_question_fingerprint(question)
        issue["question_preview"] = normalize_text(question)[:160]
    if topic is not None:
        issue["topic"] = topic
    if difficulty is not None:
        issue["difficulty"] = difficulty
    return issue
