from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
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
    return yaml.safe_load(Path(path).read_text())


def build_generator(config: dict[str, Any]) -> MathGenerator:
    model_registry = MathModelRegistry(load_registry_from_yaml(config["models"]["registry_path"]))
    prompt_presets = load_prompt_presets(config["prompt_library_path"])
    return MathGenerator(model_registry, prompt_presets=prompt_presets)


def merged_generation_config(config: dict[str, Any], side: str) -> dict[str, Any]:
    merged = dict(config.get("generation", {}))
    merged.update(config.get(f"{side}_generation_overrides", {}))
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
    benchmark_file, benchmark_entry = resolve_benchmark_file(
        registry_path=config["benchmark"]["registry_path"],
        benchmark_id=config["benchmark"].get("benchmark_id"),
        benchmark_file=config["benchmark"].get("benchmark_file"),
    )
    benchmark = read_jsonl(benchmark_file)
    generator = build_generator(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_model = config["models"]["primary_model"]
    secondary_model = config["models"]["secondary_model"]
    labels = {
        "primary": config["models"].get("primary_label", primary_model),
        "secondary": config["models"].get("secondary_label", secondary_model),
    }
    model_costs = {
        "input_cost_per_million": config["models"].get("input_cost_per_million"),
        "output_cost_per_million": config["models"].get("output_cost_per_million"),
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

    summary = build_summary(results, labels=labels)
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
    main()
