from __future__ import annotations

import argparse
import json
import sys
import time
import yaml
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_factory.core.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from evaluation.benchmark_registry import resolve_benchmark_file  # noqa: E402
from evaluation.metrics import score_prediction  # noqa: E402
from evaluation.reporting import build_summary, write_markdown_report  # noqa: E402
from inference.app.generation import MathGenerator  # noqa: E402
from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml  # noqa: E402
from inference.app.parameters import GenerationParameters  # noqa: E402
from inference.app.prompts import load_prompt_presets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two Atlas math model configurations side by side.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    return payload


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation config must define a '{key}' mapping.")
    return value


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def validate_model_availability(registry: MathModelRegistry, model_names: list[str]) -> None:
    missing: list[str] = []
    for name in model_names:
        runtime = registry.get_runtime(name)
        if runtime.spec.adapter_path and not runtime.is_available():
            missing.append(f"{name} -> {runtime.spec.adapter_path}")
    if missing:
        missing_lines = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Model artifacts required for evaluation are missing:\n"
            f"{missing_lines}\n"
            "Run training and packaging first (for example `make train`) or point the model registry "
            "to an existing adapter artifact."
        )


def build_generator(config: dict[str, Any]) -> tuple[MathGenerator, MathModelRegistry]:
    models = _require_mapping(config, "models")
    registry_path = models.get("registry_path")
    if not registry_path:
        raise ValueError("Evaluation config must define models.registry_path.")
    prompt_library_path = config.get("prompt_library_path")
    if not prompt_library_path:
        raise ValueError("Evaluation config must define prompt_library_path.")
    model_registry = MathModelRegistry(load_registry_from_yaml(_resolve_repo_path(str(registry_path))))
    prompt_presets = load_prompt_presets(_resolve_repo_path(str(prompt_library_path)))
    return MathGenerator(model_registry, prompt_presets=prompt_presets), model_registry


def merged_generation_config(config: dict[str, Any], side: str) -> dict[str, Any]:
    generation = config.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError("Evaluation config generation settings must be a mapping.")
    overrides = config.get(f"{side}_generation_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Evaluation config {side}_generation_overrides must be a mapping.")
    merged = dict(generation)
    merged.update(overrides)
    return merged


def run_generation(
    generator: MathGenerator,
    variant: str,
    example: dict[str, Any],
    generation_config: dict[str, Any],
    model_costs: dict[str, float | None],
) -> dict[str, Any]:
    params = GenerationParameters(
        question=example["question"],
        model_variant=variant,
        prompt_preset=generation_config.get("prompt_preset", "atlas_rigorous"),
        temperature=generation_config.get("temperature", 0.2),
        top_p=generation_config.get("top_p", 0.95),
        max_new_tokens=generation_config.get("max_new_tokens", 768),
        show_reasoning=generation_config.get("show_reasoning", True),
        difficulty_target=generation_config.get(
            "difficulty_target",
            example.get("difficulty", "hard"),
        ),
        num_samples=generation_config.get("num_samples", 3),
        use_calculator=generation_config.get("use_calculator", True),
        solver_mode=generation_config.get("solver_mode", "rigorous"),
        output_format=generation_config.get("output_format", "text"),
        use_cache=generation_config.get("use_cache", True),
        reference_answer=example.get("final_answer"),
        step_checks=example.get("step_checks"),
    )
    start = time.perf_counter()
    output = generator.generate(params)
    latency = time.perf_counter() - start
    scored = score_prediction(
        prediction_text=output["raw_text"],
        reference_answer=example.get("final_answer"),
        step_checks=example.get("step_checks"),
        prompt_text=output["prompt"],
        candidates=output["candidates"],
        input_cost_per_million=model_costs.get("input_cost_per_million"),
        output_cost_per_million=model_costs.get("output_cost_per_million"),
    )
    scored.update(
        {
            "text": output["answer"],
            "raw_text": output["raw_text"],
            "reasoning_steps": output["reasoning_steps"],
            "latency_s": latency,
            "selected_score": output["selected_score"],
            "candidates": output["candidates"],
            "verification": output.get("verification"),
            "cache_hit": output.get("cache_hit", False),
            "prompt_preset": output.get("prompt_preset"),
        }
    )
    return scored


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark_config = _require_mapping(config, "benchmark")
    models_config = _require_mapping(config, "models")
    output_dir_value = config.get("output_dir")
    if not output_dir_value:
        raise ValueError("Evaluation config must define output_dir.")
    registry_path = benchmark_config.get("registry_path")
    if not registry_path:
        raise ValueError("Evaluation config must define benchmark.registry_path.")
    if "primary_model" not in models_config or "secondary_model" not in models_config:
        raise ValueError("Evaluation config must define models.primary_model and models.secondary_model.")
    benchmark_file, benchmark_entry = resolve_benchmark_file(
        registry_path=_resolve_repo_path(str(registry_path)),
        benchmark_id=benchmark_config.get("benchmark_id"),
        benchmark_file=benchmark_config.get("benchmark_file"),
    )
    benchmark = read_jsonl(benchmark_file)
    
    # Apply max_eval_samples limit if specified
    max_samples = config.get("generation", {}).get("max_eval_samples")
    if max_samples and isinstance(max_samples, int) and max_samples > 0:
        benchmark = benchmark[:max_samples]
    
    generator, model_registry = build_generator(config)
    output_dir = Path(output_dir_value).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_model = models_config["primary_model"]
    secondary_model = models_config["secondary_model"]
    validate_model_availability(model_registry, [primary_model, secondary_model])
    labels = {
        "primary": models_config.get("primary_label", primary_model),
        "secondary": models_config.get("secondary_label", secondary_model),
    }
    model_costs = {
        "input_cost_per_million": models_config.get("input_cost_per_million"),
        "output_cost_per_million": models_config.get("output_cost_per_million"),
    }

    results: list[dict[str, Any]] = []
    for example in tqdm(benchmark, desc="Evaluating"):
        result = {
            "id": example.get("id", ""),
            "question": example["question"],
            "reference_answer": example.get("final_answer"),
            "reference_solution": example.get("solution"),
            "difficulty": example.get("difficulty", "hard"),
            "topic": example.get("topic", "general"),
            "source": example.get("source", "unknown"),
            "pack_id": example.get("pack_id", "unknown"),
            "generator_family": (
                ((example.get("generator") or {}).get("generator_family"))
                if isinstance(example.get("generator"), dict)
                else None
            )
            or "unknown",
            "step_checks": example.get("step_checks", []),
            "benchmark_id": benchmark_entry.get("id"),
        }
        result["primary"] = run_generation(
            generator,
            primary_model,
            example,
            merged_generation_config(config, "primary"),
            model_costs,
        )
        result["secondary"] = run_generation(
            generator,
            secondary_model,
            example,
            merged_generation_config(config, "secondary"),
            model_costs,
        )
        results.append(result)

    summary = build_summary(
        results,
        labels=labels,
        metadata={
            "benchmark_id": benchmark_entry.get("id"),
            "benchmark_path": benchmark_file,
            "benchmark_resolved_path": benchmark_entry.get("resolved_path"),
            "run_name": output_dir.name,
            "output_dir": str(output_dir),
        },
    )
    write_jsonl(output_dir / "per_example.jsonl", results)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "leaderboard.json",
        {
            "labels": labels,
            "primary": summary["primary"],
            "secondary": summary["secondary"],
        },
    )
    write_markdown_report(output_dir / "summary.md", summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
