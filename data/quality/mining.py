from __future__ import annotations

from typing import Any

from data.quality.difficulty import difficulty_score


def select_failure_cases(
    records: list[dict[str, Any]], min_difficulty: str = "hard", limit: int = 200
) -> list[dict[str, Any]]:
    threshold = difficulty_score(min_difficulty)
    curated: list[dict[str, Any]] = []
    for record in records:
        finetuned = record.get("finetuned") or {}
        if finetuned.get("correct") is True:
            continue
        difficulty = record.get("difficulty", "hard")
        if difficulty_score(difficulty) < threshold:
            continue
        reference_solution = record.get("reference_solution") or record.get("solution")
        if not reference_solution:
            continue
        curated.append(
            {
                "schema_version": "v2",
                "question": record["question"],
                "solution": reference_solution,
                "difficulty": difficulty,
                "topic": record.get("topic", "general"),
                "source": f"failure_mining::{record.get('source', 'evaluation')}",
                "final_answer": record.get("reference_answer"),
                "step_checks": record.get("step_checks", []),
                "failure_case": True,
                "reasoning_style": "verification",
                "tags": ["failure_replay", "mined"],
                "quality_score": 0.85,
                "metadata": {
                    "base_prediction": (record.get("base") or {}).get("final_answer"),
                    "finetuned_prediction": finetuned.get("final_answer"),
                    "error_type": finetuned.get("error_type", "incorrect"),
                },
            }
        )
        if len(curated) >= limit:
            break
    return curated


def select_hard_examples(records: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            difficulty_score(item.get("difficulty")),
            item.get("quality_score", 0.0),
            len(item.get("solution", "")),
        ),
        reverse=True,
    )
    return ranked[:limit]
