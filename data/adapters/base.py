from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai_factory.core.hashing import normalize_text, stable_question_fingerprint
from ai_factory.core.schemas import ContaminationStatus, SourceLineage, StepCheck


@dataclass
class PublicDatasetSpec:
    id: str
    title: str
    kind: str
    loader: str
    path: str
    split: str = "train"
    dataset_family: str = "public_adapter"
    expected_topic: str = "calculus"
    reasoning_style: str = "chain_of_thought"
    usage: str = "train"
    default_weight: float = 1.0
    benchmark_tags: list[str] = field(default_factory=list)
    split_strategy: str = "native"
    question_keys: list[str] = field(default_factory=list)
    solution_keys: list[str] = field(default_factory=list)
    difficulty_keys: list[str] = field(default_factory=list)
    topic_keys: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    output_file: str = ""
    notes: str = ""
    source_url: str | None = None
    license: str | None = None


def load_public_registry(path: str | Path) -> list[PublicDatasetSpec]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return [PublicDatasetSpec(**entry) for entry in payload.get("datasets", [])]


def first_text(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            flattened = "\n".join(str(item) for item in value if item is not None).strip()
            if flattened:
                return flattened
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def matches_filters(record: dict[str, Any], spec: PublicDatasetSpec) -> bool:
    filters = spec.filters or {}
    question_blob = normalize_text(
        " ".join(str(record.get(key, "")) for key in set(spec.question_keys + spec.topic_keys)).lower()
    )
    topic_blob = normalize_text(" ".join(str(record.get(key, "")) for key in spec.topic_keys).lower())
    if "question_contains" in filters:
        needles = [item.lower() for item in filters["question_contains"]]
        if not any(needle in question_blob for needle in needles):
            return False
    if "topic_contains" in filters:
        needles = [item.lower() for item in filters["topic_contains"]]
        if not any(needle in topic_blob for needle in needles):
            return False
    if "exclude_question_contains" in filters:
        needles = [item.lower() for item in filters["exclude_question_contains"]]
        if any(needle in question_blob for needle in needles):
            return False
    return True


def iter_source_rows(spec: PublicDatasetSpec, cache_dir: str | None = None) -> Iterable[dict[str, Any]]:
    if spec.loader != "huggingface":
        return []
    from datasets import load_dataset

    dataset = load_dataset(spec.path, split=spec.split, cache_dir=cache_dir)
    return dataset


def normalize_public_record(row: dict[str, Any], spec: PublicDatasetSpec) -> dict[str, Any] | None:
    question = first_text(row, spec.question_keys)
    solution = first_text(row, spec.solution_keys)
    if not question or not solution:
        return None
    difficulty = first_text(row, spec.difficulty_keys) or "hard"
    topic = first_text(row, spec.topic_keys) or spec.expected_topic
    step_checks_raw = row.get("step_checks", [])
    if isinstance(step_checks_raw, list):
        step_checks = [
            item if isinstance(item, dict) else StepCheck(kind="substring", value=str(item)).model_dump()
            for item in step_checks_raw
            if str(item).strip()
        ]
    else:
        step_checks = []
    lineage = SourceLineage(
        dataset_id=spec.id,
        dataset_family=spec.dataset_family,
        origin_path=spec.path,
        loader=spec.loader,
        source_url=spec.source_url,
        license=spec.license,
        filters=spec.filters,
        source_record_id=str(row.get("id") or row.get("uuid") or stable_question_fingerprint(question)),
    )
    return {
        "schema_version": "v2",
        "id": str(row.get("id") or stable_question_fingerprint(question)),
        "question": question,
        "solution": solution,
        "difficulty": re.sub(r"\s+", " ", str(difficulty)).strip(),
        "topic": topic,
        "source": spec.id,
        "final_answer": row.get("final_answer") or row.get("answer"),
        "step_checks": step_checks,
        "reasoning_style": spec.reasoning_style,
        "quality_score": 0.0,
        "pack_id": spec.id,
        "failure_case": False,
        "tags": list(dict.fromkeys(spec.benchmark_tags + [spec.usage, spec.expected_topic.replace(" ", "_")])),
        "contamination": ContaminationStatus().model_dump(),
        "lineage": lineage.model_dump(),
        "metadata": {
            "public_dataset_id": spec.id,
            "loader_path": spec.path,
            "notes": spec.notes,
            "usage": spec.usage,
            "default_weight": spec.default_weight,
            "benchmark_tags": spec.benchmark_tags,
            "split_strategy": spec.split_strategy,
        },
    }
