from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from ai_factory.core.io import write_json, write_jsonl
from evaluation.metrics import score_prediction
from inference.app.generation import MathGenerator
from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml
from inference.app.parameters import GenerationParameters
from inference.app.prompts import load_prompt_presets
from data.synthesis import DatasetSpec, GENERATOR_MAP
from data.synthesis.base import choose_weighted

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATION_CONFIG = REPO_ROOT / "data" / "configs" / "generation.yaml"
DEFAULT_PROMPT_LIBRARY = REPO_ROOT / "inference" / "configs" / "prompt_presets.yaml"


def load_dataset_specs(config_path: str | Path = DEFAULT_GENERATION_CONFIG) -> list[DatasetSpec]:
    payload = yaml.safe_load(Path(config_path).read_text()) or {}
    return [DatasetSpec(**item) for item in payload.get("dataset_specs", [])]


def generate_benchmark_records(
    *,
    question_count: int,
    seed: int,
    config_path: str | Path = DEFAULT_GENERATION_CONFIG,
) -> list[dict[str, Any]]:
    specs = load_dataset_specs(config_path)
    if not specs:
        raise ValueError(f"No dataset specs were found in {config_path}")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    attempts = 0
    max_attempts = max(question_count * 20, 1_000)

    while len(records) < question_count:
        if attempts >= max_attempts:
            raise RuntimeError(f"Could not generate {question_count} unique benchmark questions.")
        spec = specs[attempts % len(specs)]
        difficulty = choose_weighted(rng, spec.difficulty_mix)
        generator = GENERATOR_MAP[spec.family]
        record = dict(generator(rng, spec, attempts, difficulty))
        attempts += 1
        if record["question"] in seen_questions:
            continue
        seen_questions.add(record["question"])
        record["dataset_split"] = "benchmark"
        record["benchmark_seed"] = seed
        records.append(record)
    return records


def create_temp_model_registry(
    registry_path: str | Path,
    *,
    model_name: str,
    base_model: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    dtype: str = "bfloat16",
    label: str | None = None,
    description: str | None = None,
) -> Path:
    payload = {
        "models": [
            {
                "name": model_name,
                "label": label or model_name,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
                "dtype": dtype,
                "description": description or "Generated evaluation model registry entry.",
                "tags": ["generated", "evaluation"],
            }
        ]
    }
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def build_generator(registry_path: str | Path, *, prompt_library_path: str | Path = DEFAULT_PROMPT_LIBRARY) -> MathGenerator:
    model_registry = MathModelRegistry(load_registry_from_yaml(registry_path))
    prompt_presets = load_prompt_presets(prompt_library_path)
    return MathGenerator(model_registry, prompt_presets=prompt_presets)


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    registry_path: str | Path,
    model_variant: str,
    prompt_preset: str = "atlas_rigorous",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    num_samples: int = 2,
    show_reasoning: bool = True,
    use_calculator: bool = True,
    use_cache: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generator = build_generator(registry_path)
    results: list[dict[str, Any]] = []
    total_marks = 0
    topic_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    correct_by_topic: Counter[str] = Counter()
    correct_by_difficulty: Counter[str] = Counter()

    for record in tqdm(records, desc="Evaluating generated benchmark"):
        params = GenerationParameters(
            question=record["question"],
            model_variant=model_variant,
            prompt_preset=prompt_preset,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            show_reasoning=show_reasoning,
            difficulty_target=record.get("difficulty", "hard"),
            num_samples=num_samples,
            use_calculator=use_calculator,
            solver_mode="rigorous",
            output_format="text",
            use_cache=use_cache,
            reference_answer=record.get("final_answer"),
            step_checks=record.get("step_checks"),
        )
        start = time.perf_counter()
        output = generator.generate(params)
        latency = time.perf_counter() - start
        scored = score_prediction(
            prediction_text=output["raw_text"],
            reference_answer=record.get("final_answer"),
            step_checks=record.get("step_checks"),
            prompt_text=output["prompt"],
            candidates=output["candidates"],
        )
        mark = 1 if scored["correct"] else 0
        total_marks += mark
        topic = str(record.get("topic", "general"))
        difficulty = str(record.get("difficulty", "hard"))
        topic_counter[topic] += 1
        difficulty_counter[difficulty] += 1
        if mark:
            correct_by_topic[topic] += 1
            correct_by_difficulty[difficulty] += 1
        results.append(
            {
                "id": record.get("id"),
                "question": record["question"],
                "reference_answer": record.get("final_answer"),
                "reference_solution": record.get("solution"),
                "topic": topic,
                "difficulty": difficulty,
                "prediction": output["answer"],
                "raw_text": output["raw_text"],
                "final_answer": scored["final_answer"],
                "correct": scored["correct"],
                "mark": mark,
                "latency_s": latency,
                "parse_rate": scored["parse_rate"],
                "step_correctness": scored["step_correctness"],
                "error_type": scored["error_type"],
                "verification": output.get("verification"),
            }
        )

    summary = {
        "question_count": len(records),
        "total_marks": total_marks,
        "accuracy": (total_marks / len(records)) if records else 0.0,
        "correct": total_marks,
        "incorrect_or_blank": len(records) - total_marks,
        "by_topic": {
            topic: {
                "questions": count,
                "marks": correct_by_topic[topic],
                "accuracy": (correct_by_topic[topic] / count) if count else 0.0,
            }
            for topic, count in sorted(topic_counter.items())
        },
        "by_difficulty": {
            difficulty: {
                "questions": count,
                "marks": correct_by_difficulty[difficulty],
                "accuracy": (correct_by_difficulty[difficulty] / count) if count else 0.0,
            }
            for difficulty, count in sorted(difficulty_counter.items())
        },
    }
    return results, summary


def write_evaluation_bundle(
    output_dir: str | Path,
    *,
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "questions.jsonl", questions)
    write_jsonl(directory / "results.jsonl", results)
    write_json(directory / "summary.json", summary)

