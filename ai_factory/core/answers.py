from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from sympy import simplify, sympify

from ai_factory.core.hashing import normalize_text

SAFE_OPERATORS: dict[type[ast.AST], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}

SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

FINAL_ANSWER_PATTERNS = [
    re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE),
    re.compile(r"\\boxed\{(.+?)\}"),
    re.compile(r"Answer:\s*(.+)", re.IGNORECASE),
]


@dataclass(frozen=True)
class PredictionVerification:
    final_answer: str | None
    equivalent: bool
    step_correctness: float | None
    verifier_agreement: bool
    formatting_failure: bool
    arithmetic_slip: bool
    error_type: str


def safe_eval(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in SAFE_NAMES:
            return float(SAFE_NAMES[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
            fn = SAFE_OPERATORS[type(node.op)]
            return float(fn(_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
            fn = SAFE_OPERATORS[type(node.op)]
            return float(fn(_eval(node.operand)))
        raise ValueError(f"Unsafe expression: {expression}")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed)


def resolve_calculator_tags(text: str) -> tuple[str, list[dict[str, str]]]:
    traces: list[dict[str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        try:
            result = safe_eval(expression)
            formatted = str(int(result)) if float(result).is_integer() else f"{result:.8g}"
            traces.append({"expression": expression, "result": formatted})
            return f"{expression} = {formatted}"
        except Exception:
            traces.append({"expression": expression, "result": "error"})
            return expression

    updated = re.sub(r"\[\[calc:\s*(.+?)\s*\]\]", _replace, text, flags=re.IGNORECASE)
    return updated, traces


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().rstrip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidate = lines[-1].rstrip(".")
        if _looks_like_answer_candidate(candidate):
            return candidate
    return None


def _looks_like_answer_candidate(text: str) -> bool:
    cleaned = normalize_text(text).lower()
    if not cleaned:
        return False
    if cleaned in {"yes", "no", "true", "false", "undefined", "impossible", "does not exist"}:
        return True
    return any(
        token in cleaned
        for token in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\\", "/", "^", "=", "pi", "sqrt", "infty", "∞")
    )


def split_reasoning(text: str) -> tuple[str, str | None]:
    final_answer = extract_final_answer(text)
    if final_answer is None:
        return text.strip(), None
    marker_match = re.search(r"(Final Answer:|Answer:)\s*", text, flags=re.IGNORECASE)
    if marker_match:
        reasoning = text[: marker_match.start()].strip()
        return reasoning, final_answer
    return text.strip(), final_answer


def answer_key(answer: str | None) -> str:
    if not answer:
        return ""
    cleaned = normalize_text(answer)
    return cleaned.replace(" ", "")


def answers_equivalent(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if answer_key(left) == answer_key(right):
        return True
    try:
        equivalent = simplify(sympify(left) - sympify(right)) == 0
        return cast(bool, equivalent)
    except Exception:
        return False


def _iter_step_values(step_checks: Iterable[Any] | None) -> list[tuple[str, float, str]]:
    values: list[tuple[str, float, str]] = []
    if not step_checks:
        return values
    for item in step_checks:
        if isinstance(item, str):
            values.append((normalize_text(item), 1.0, "substring"))
            continue
        if isinstance(item, dict):
            values.append(
                (
                    normalize_text(str(item.get("value", ""))),
                    float(item.get("weight", 1.0) or 1.0),
                    str(item.get("kind", "substring")),
                )
            )
            continue
        value = normalize_text(getattr(item, "value", ""))
        weight = float(getattr(item, "weight", 1.0) or 1.0)
        kind = str(getattr(item, "kind", "substring"))
        values.append((value, weight, kind))
    return [(value, weight, kind) for value, weight, kind in values if value]


def compute_step_correctness(prediction: str, step_checks: Iterable[Any] | None) -> float | None:
    parsed_checks = _iter_step_values(step_checks)
    if not parsed_checks:
        return None
    normalized_prediction = normalize_text(prediction)
    total_weight = sum(weight for _, weight, _ in parsed_checks) or 1.0
    hits = 0.0
    for step_value, weight, kind in parsed_checks:
        if kind == "regex":
            matched = re.search(step_value, prediction, flags=re.IGNORECASE) is not None
        else:
            matched = step_value in normalized_prediction
        if matched:
            hits += weight
    return hits / total_weight


def detect_formatting_failure(prediction_text: str) -> bool:
    return "Final Answer:" not in prediction_text and "\\boxed" not in prediction_text


def detect_arithmetic_slip(prediction_text: str, reference_answer: str | None) -> bool:
    final_answer = extract_final_answer(prediction_text)
    if not final_answer or not reference_answer:
        return False
    try:
        if answers_equivalent(final_answer, reference_answer):
            return False
        candidate_val = float(sympify(final_answer))
        reference_val = float(sympify(reference_answer))
        if reference_val == 0:
            return abs(candidate_val) < 1e-6
        return abs(candidate_val - reference_val) / (abs(reference_val) + 1e-6) < 0.05
    except Exception:
        return False


def classify_prediction_failure(
    prediction_text: str,
    reference_answer: str | None,
    step_checks: Iterable[Any] | None = None,
) -> str:
    final_answer = extract_final_answer(prediction_text)
    if final_answer is None:
        return "no_answer"
    if detect_formatting_failure(prediction_text):
        return "formatting_failure"
    if detect_arithmetic_slip(prediction_text, reference_answer):
        return "arithmetic_slip"
    step_score = compute_step_correctness(prediction_text, step_checks)
    if step_score is not None and step_score <= 0.0:
        return "reasoning_off_track"
    if reference_answer and not answers_equivalent(final_answer, reference_answer):
        return "wrong_final_answer"
    return "correct"


def candidate_agreement(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    counts = Counter(answer_key(candidate.get("final_answer")) for candidate in candidates)
    if not counts:
        return 0.0
    return max(counts.values()) / len(candidates)


def choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    answer_counts = Counter(answer_key(candidate.get("final_answer")) for candidate in candidates)
    for candidate in candidates:
        candidate["vote_count"] = answer_counts[answer_key(candidate.get("final_answer"))]
        candidate["score"] = (
            2.5 * candidate["vote_count"]
            + (1.0 if candidate.get("final_answer") else 0.0)
            + (0.75 if not detect_formatting_failure(candidate.get("text", "")) else 0.0)
            + (0.5 if candidate.get("calculator_trace") else 0.0)
            + 0.5 * float(candidate.get("verification_score", 0.0) or 0.0)
        )
    return max(candidates, key=lambda item: (item["score"], len(item.get("text", ""))))


def verify_prediction(
    prediction_text: str,
    reference_answer: str | None,
    step_checks: Iterable[Any] | None = None,
) -> PredictionVerification:
    final_answer = extract_final_answer(prediction_text)
    equivalent = answers_equivalent(final_answer or prediction_text, reference_answer)
    step_correctness = compute_step_correctness(prediction_text, step_checks)
    formatting_failure = detect_formatting_failure(prediction_text)
    arithmetic_slip = detect_arithmetic_slip(prediction_text, reference_answer)
    verifier_agreement = equivalent or not formatting_failure
    error_type = classify_prediction_failure(prediction_text, reference_answer, step_checks)
    return PredictionVerification(
        final_answer=final_answer,
        equivalent=equivalent,
        step_correctness=step_correctness,
        verifier_agreement=verifier_agreement,
        formatting_failure=formatting_failure,
        arithmetic_slip=arithmetic_slip,
        error_type=error_type,
    )
