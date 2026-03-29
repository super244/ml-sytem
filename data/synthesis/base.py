from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import random
from typing import Any

from ai_factory.core.schemas import ContaminationStatus, GeneratorMetadata, SourceLineage, StepCheck


Difficulty = str


@dataclass
class DatasetSpec:
    id: str
    title: str
    family: str
    topic: str
    difficulty_mix: dict[str, float]
    pedagogical_focus: list[str]
    reasoning_style: str = "chain_of_thought"


def choose_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.items())
    levels, probs = zip(*items, strict=True)
    total = sum(probs)
    cursor = rng.random() * total
    running = 0.0
    for level, prob in items:
        running += prob
        if cursor <= running:
            return level
    return str(levels[-1])


def nz(rng: random.Random, low: int = -6, high: int = 6, exclude: set[int] | None = None) -> int:
    exclude = exclude or {0}
    value = 0
    while value in exclude:
        value = rng.randint(low, high)
    return value


def frac_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def signed_term(coeff: int, body: str, first: bool = False) -> str:
    sign = "-" if coeff < 0 else "+"
    magnitude = abs(coeff)
    if body == "":
        core = str(magnitude)
    elif magnitude == 1:
        core = body
    else:
        core = f"{magnitude}{body}"
    if first:
        return core if coeff > 0 else f"-{core}"
    return f" {sign} {core}"


def poly_str(terms: list[tuple[int, int]], variable: str = "x") -> str:
    rendered: list[str] = []
    for coeff, power in terms:
        if coeff == 0:
            continue
        if power == 0:
            body = ""
        elif power == 1:
            body = variable
        else:
            body = f"{variable}^{power}"
        rendered.append(signed_term(coeff, body, first=(len(rendered) == 0)))
    return "".join(rendered) if rendered else "0"


def poly_value(terms: list[tuple[int, int]], x_value: int) -> Fraction:
    total = Fraction(0, 1)
    for coeff, power in terms:
        total += Fraction(coeff * (x_value ** power), 1)
    return total


def derivative_terms(terms: list[tuple[int, int]]) -> list[tuple[int, int]]:
    derived = []
    for coeff, power in terms:
        if power == 0:
            continue
        derived.append((coeff * power, power - 1))
    return derived or [(0, 0)]


def integral_terms(terms: list[tuple[int, int]]) -> list[tuple[Fraction, int]]:
    antiderived: list[tuple[Fraction, int]] = []
    for coeff, power in terms:
        antiderived.append((Fraction(coeff, power + 1), power + 1))
    return antiderived


def fractional_poly_str(terms: list[tuple[Fraction, int]], variable: str = "x") -> str:
    rendered: list[str] = []
    for coeff, power in terms:
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        magnitude = abs(coeff)
        magnitude_text = frac_str(magnitude)
        if power == 0:
            body = ""
        elif power == 1:
            body = variable
        else:
            body = f"{variable}^{power}"
        if body and magnitude == 1:
            core = body
        elif body:
            core = f"{magnitude_text}{body}"
        else:
            core = magnitude_text
        if not rendered:
            rendered.append(core if coeff > 0 else f"-{core}")
        else:
            rendered.append(f" {sign} {core}")
    return "".join(rendered) if rendered else "0"


def evaluate_antiderivative(terms: list[tuple[Fraction, int]], x_value: int) -> Fraction:
    total = Fraction(0, 1)
    for coeff, power in terms:
        total += coeff * (x_value ** power)
    return total


def step_checks_from_strings(values: list[str]) -> list[dict[str, Any]]:
    return [StepCheck(kind="substring", value=value).model_dump() for value in values if value]


def make_record(
    spec: DatasetSpec,
    index: int,
    difficulty: Difficulty,
    question: str,
    solution: str,
    final_answer: str,
    step_checks: list[str],
    focus: str,
    topic: str | None = None,
    reasoning_style: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    lineage = SourceLineage(
        dataset_id=spec.id,
        dataset_family=spec.family,
        loader="synthetic",
        notes=[focus],
    )
    generator = GeneratorMetadata(
        generator_family=spec.family,
        curriculum_bucket=difficulty,
        pedagogical_focus=[focus, *spec.pedagogical_focus[:2]],
    )
    return {
        "schema_version": "v2",
        "id": f"{spec.id}-{index:06d}",
        "question": question,
        "solution": solution,
        "difficulty": difficulty,
        "topic": topic or spec.topic,
        "source": spec.id,
        "final_answer": final_answer,
        "step_checks": step_checks_from_strings(step_checks),
        "failure_case": False,
        "reasoning_style": reasoning_style or spec.reasoning_style,
        "quality_score": 0.0,
        "tags": list(dict.fromkeys((tags or []) + [spec.family, focus.replace(" ", "_")])),
        "pack_id": spec.id,
        "contamination": ContaminationStatus().model_dump(),
        "lineage": lineage.model_dump(),
        "generator": generator.model_dump(),
        "metadata": {
            "dataset_title": spec.title,
            "focus": focus,
            "family": spec.family,
            "curriculum_bucket": difficulty,
        },
    }
