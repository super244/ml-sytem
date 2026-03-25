from __future__ import annotations

from ai_factory.core.hashing import normalize_text


def estimate_quality_score(record: dict) -> float:
    question = normalize_text(record.get("question"))
    solution = normalize_text(record.get("solution"))
    final_answer = normalize_text(record.get("final_answer"))
    step_checks = record.get("step_checks") or []
    tags = record.get("tags") or []
    score = 0.2
    if len(question) >= 20:
        score += 0.15
    if len(solution) >= 50:
        score += 0.2
    if len(solution) >= 180:
        score += 0.1
    if final_answer:
        score += 0.15
    if step_checks:
        score += min(0.2, 0.05 * len(step_checks))
    if record.get("failure_case"):
        score += 0.05
    if record.get("reasoning_style") in {"proof", "verification"}:
        score += 0.05
    if "verified" in tags or "benchmark" in tags:
        score += 0.05
    return max(0.0, min(1.0, round(score, 4)))
