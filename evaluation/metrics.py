from __future__ import annotations

from typing import Any

from ai_factory.core.answers import candidate_agreement, verify_prediction
from ai_factory.core.tokens import approximate_token_count, estimate_generation_cost_usd


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
        "estimated_cost_usd": estimate_generation_cost_usd(
            prompt_tokens,
            completion_tokens,
            input_cost_per_million,
            output_cost_per_million,
        ),
        "candidate_agreement": candidate_agreement(candidates or []),
    }
