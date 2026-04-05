from __future__ import annotations

from typing import Any

from ai_factory.core.answers import candidate_agreement, verify_prediction
from ai_factory.core.tokens import approximate_token_count, estimate_generation_cost_usd


def _weighted_mean(components: list[tuple[str, float, float | None]]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for _, weight, value in components:
        if value is None:
            continue
        total_weight += weight
        weighted_sum += weight * float(value)
    return weighted_sum / total_weight if total_weight else 0.0


def _quality_components(
    *,
    verification: Any,
    candidate_score: float,
) -> dict[str, float | None]:
    return {
        "answer_present": 1.0 if verification.final_answer is not None else 0.0,
        "correctness": 1.0 if verification.equivalent else 0.0,
        "step_correctness": verification.step_correctness,
        "verifier_agreement": 1.0 if verification.verifier_agreement else 0.0,
        "format_adherence": 0.0 if verification.formatting_failure else 1.0,
        "arithmetic_stability": 0.0 if verification.arithmetic_slip else 1.0,
        "candidate_consensus": candidate_score,
    }


def score_prediction(
    prediction_text: str,
    reference_answer: str | None,
    step_checks: list[Any] | None,
    prompt_text: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> dict[str, Any]:
    verification = verify_prediction(
        prediction_text=prediction_text,
        reference_answer=reference_answer,
        step_checks=step_checks,
    )
    prompt_tokens = approximate_token_count(prompt_text, tokenizer)
    completion_tokens = approximate_token_count(prediction_text, tokenizer)
    candidate_score = candidate_agreement(candidates or [])
    metric_components = _quality_components(verification=verification, candidate_score=candidate_score)
    return {
        "final_answer": verification.final_answer,
        "solve": verification.final_answer is not None,
        "parse_rate": 1.0 if verification.final_answer is not None else 0.0,
        "correct": verification.equivalent,
        "step_correctness": verification.step_correctness,
        "verifier_agreement": verification.verifier_agreement,
        "formatting_failure": verification.formatting_failure,
        "arithmetic_slip": verification.arithmetic_slip,
        "no_answer": verification.final_answer is None,
        "error_type": verification.error_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_to_completion_ratio": prompt_tokens / max(completion_tokens, 1),
        "candidate_count": len(candidates or []),
        "estimated_cost_usd": estimate_generation_cost_usd(
            prompt_tokens,
            completion_tokens,
            input_cost_per_million,
            output_cost_per_million,
        ),
        "candidate_agreement": candidate_score,
        "metric_components": metric_components,
        "quality_score": _weighted_mean(
            [
                ("correctness", 0.45, metric_components["correctness"]),
                ("answer_present", 0.1, metric_components["answer_present"]),
                ("step_correctness", 0.15, metric_components["step_correctness"]),
                ("verifier_agreement", 0.1, metric_components["verifier_agreement"]),
                ("format_adherence", 0.1, metric_components["format_adherence"]),
                ("candidate_consensus", 0.1, metric_components["candidate_consensus"]),
            ]
        ),
    }
