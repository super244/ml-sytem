from __future__ import annotations

from typing import Any


def approximate_token_count(text: str | None, tokenizer: Any | None = None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except (KeyError, TypeError, ValueError):
            return max(1, int(len(text.split()) * 1.35))
    return max(1, int(len(text.split()) * 1.35))


def estimate_generation_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    return (prompt_tokens / 1_000_000) * input_cost_per_million + (
        completion_tokens / 1_000_000
    ) * output_cost_per_million
