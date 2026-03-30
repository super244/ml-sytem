from __future__ import annotations

import re
from typing import Any

from ai_factory.core.hashing import normalize_text

DIFFICULTY_ALIASES = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "olympiad": 4,
    "amc": 2,
    "aime": 3,
    "imo": 4,
}


def normalize_difficulty(value: Any) -> str:
    if value is None:
        return "hard"
    if isinstance(value, (int, float)):
        if value <= 1:
            return "easy"
        if value <= 2:
            return "medium"
        if value <= 3:
            return "hard"
        return "olympiad"
    cleaned = normalize_text(str(value)).lower()
    if cleaned in DIFFICULTY_ALIASES:
        score = DIFFICULTY_ALIASES[cleaned]
        return ["easy", "medium", "hard", "olympiad"][min(max(score, 1), 4) - 1]
    match = re.search(r"(\d+)", cleaned)
    if match:
        return normalize_difficulty(int(match.group(1)))
    return "hard"


def difficulty_score(level: str | None) -> int:
    if not level:
        return 3
    return DIFFICULTY_ALIASES.get(normalize_text(level).lower(), 3)


def estimate_difficulty(question: str, solution: str | None = None) -> str:
    blob = normalize_text(f"{question} {solution or ''}").lower()
    score = 1
    if len(blob) > 180:
        score += 1
    if len(blob) > 340:
        score += 1
    if any(
        token in blob
        for token in ("prove", "show that", "olympiad", "infinity", "double integral", "functional equation")
    ):
        score += 1
    if any(token in blob for token in ("aime", "imo", "hard", "contest")):
        score += 1
    score = min(score, 4)
    return ["easy", "medium", "hard", "olympiad"][score - 1]
