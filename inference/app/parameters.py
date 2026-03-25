from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ai_factory.core.hashing import sha256_text


@dataclass
class GenerationParameters:
    question: str
    model_variant: str = "finetuned"
    prompt_preset: str = "atlas_rigorous"
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 768
    show_reasoning: bool = True
    difficulty_target: str | None = "hard"
    num_samples: int = 3
    use_calculator: bool = True
    solver_mode: str = "rigorous"
    output_format: str = "text"
    use_cache: bool = True
    reference_answer: str | None = None
    step_checks: list[Any] | None = None

    def cache_key(self) -> str:
        return sha256_text(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False))
