from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.generated_benchmark import (  # noqa: E402
    create_temp_model_registry,
    evaluate_records,
    generate_benchmark_records,
    write_evaluation_bundle,
)
from training.src.workflows import DEFAULT_BASE_MODEL, build_workflow_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on 10,000 newly generated questions.")
    parser.add_argument("--run-name", default="generated-eval")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--question-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = build_workflow_layout("generated_eval", args.run_name)
    questions = generate_benchmark_records(question_count=args.question_count, seed=args.seed)
    registry_path = create_temp_model_registry(
        layout.configs_dir / "generated_eval_registry.yaml",
        model_name="generated_eval_model",
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        label="Generated Eval Model",
        description="Temporary model registry entry for the generated evaluation workflow.",
    )
    results, summary = evaluate_records(
        questions,
        registry_path=registry_path,
        model_variant="generated_eval_model",
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )
    write_evaluation_bundle(layout.reports_dir, questions=questions, results=results, summary=summary)
    print(
        json.dumps(
            {
                "workflow": "generated_eval",
                "run_name": args.run_name,
                "registry_path": str(registry_path),
                "output_dir": str(layout.reports_dir),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
