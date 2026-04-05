from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from data.quality.profiles import build_dataset_profile
from data.reports.cards import dataset_card_text
from data.synthesis.base import DatasetSpec, choose_weighted
from data.synthesis.families import (
    generate_derivative_example,
    generate_integral_example,
    generate_limits_series_example,
    generate_multivariable_example,
    generate_odes_optimization_example,
    generate_olympiad_reasoning_example,
)

GeneratorFn = Callable[[random.Random, DatasetSpec, int, str], dict[str, Any]]


GENERATOR_MAP: dict[str, GeneratorFn] = {
    "derivatives": generate_derivative_example,
    "integrals": generate_integral_example,
    "limits_series": generate_limits_series_example,
    "multivariable": generate_multivariable_example,
    "odes_optimization": generate_odes_optimization_example,
    "olympiad_reasoning": generate_olympiad_reasoning_example,
}


def generate_records(spec: DatasetSpec, target_size_bytes: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    generator = GENERATOR_MAP[spec.family]
    records: list[dict[str, Any]] = []
    total_bytes = 0
    index = 0
    seen_questions: set[str] = set()
    duplicate_budget = 0
    while total_bytes < target_size_bytes:
        difficulty = choose_weighted(rng, spec.difficulty_mix)
        record = generator(rng, spec, index, difficulty)
        if record["question"] in seen_questions:
            duplicate_budget += 1
            if duplicate_budget < 2000:
                continue
        duplicate_budget = 0
        seen_questions.add(record["question"])
        payload = json.dumps(record, ensure_ascii=False)
        total_bytes += len(payload.encode("utf-8")) + 1
        records.append(record)
        index += 1
    return records


def build_catalog_entry(spec: DatasetSpec, output_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    profile_summary = build_dataset_profile(records, title=spec.title)
    preview_examples = [
        {
            "id": record["id"],
            "question": record["question"],
            "difficulty": record["difficulty"],
            "topic": record["topic"],
            "final_answer": record["final_answer"],
        }
        for record in records[:5]
    ]
    entry = {
        "id": spec.id,
        "title": spec.title,
        "kind": "custom",
        "family": spec.family,
        "topic": spec.topic,
        "path": str(output_path),
        "num_rows": len(records),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "description": " / ".join(spec.pedagogical_focus),
        "reasoning_style": spec.reasoning_style,
        "preview_examples": preview_examples,
        "profile_summary": profile_summary,
        "quality_summary": profile_summary.get("quality_summary", {}),
    }
    return entry


def write_dataset_card(entry: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(dataset_card_text(entry))
