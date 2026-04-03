from __future__ import annotations

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_factory.core.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from training.src.workflows import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    build_training_config_payload,
    build_workflow_corpus,
    build_workflow_layout,
    launch_training,
    parse_csv_values,
    write_training_config,
)


def _load_evaluation_helpers():
    module = import_module("evaluation.generated_benchmark")
    return module.create_temp_model_registry, module.evaluate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-guided reinforcement-style training over an existing model.")
    parser.add_argument("--run-name", default="reinforcement-workflow")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=None, help="Optional adapter path for the already trained model.")
    parser.add_argument("--public-datasets", default="")
    parser.add_argument("--private-categories", default="")
    parser.add_argument("--local-datasets", default="")
    parser.add_argument("--custom-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-prompts", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8.0e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_rollout_records(
    evaluation_results: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted = [result for result in evaluation_results if result.get("mark") == 1]
    if not accepted:
        raise RuntimeError("No successful rollout samples were produced. Lower the difficulty or change the model.")
    rollout_rows = [
        {
            "id": f"rollout-{index:06d}",
            "question": str(result["question"]),
            "solution": str(result["prediction"]),
            "final_answer": result.get("reference_answer"),
            "difficulty": result.get("difficulty", "hard"),
            "topic": result.get("topic", "general"),
            "source": "reinforcement_rollout",
            "pack_id": "reinforcement_rollout",
            "step_checks": [],
            "failure_case": False,
            "quality_score": 1.0,
            "reasoning_style": "chain_of_thought",
            "metadata": {
                "reward_mark": result.get("mark"),
                "latency_s": result.get("latency_s"),
                "error_type": result.get("error_type"),
            },
        }
        for index, result in enumerate(accepted)
    ]
    rng = random.Random(seed)
    rng.shuffle(rollout_rows)
    split_index = max(1, int(len(rollout_rows) * 0.9))
    return rollout_rows[:split_index], rollout_rows[split_index:] or rollout_rows[:1]


def main() -> None:
    args = parse_args()
    create_temp_model_registry, evaluate_records = _load_evaluation_helpers()
    public_datasets = parse_csv_values(args.public_datasets)
    private_categories = parse_csv_values(args.private_categories)
    local_datasets = parse_csv_values(args.local_datasets)
    prompt_corpus = build_workflow_corpus(
        workflow_name="reinforcement_prompts",
        run_name=args.run_name,
        public_datasets=public_datasets,
        private_categories=private_categories,
        local_datasets=local_datasets,
        custom_root=args.custom_root,
        seed=args.seed,
        eval_ratio=0.0,
        test_ratio=0.0,
        max_samples=args.rollout_prompts,
    )
    prompts = read_jsonl(prompt_corpus["train_file"])[: args.rollout_prompts]
    layout = build_workflow_layout("reinforcement", args.run_name)
    registry_path = create_temp_model_registry(
        layout.configs_dir / "reinforcement_registry.yaml",
        model_name="reinforcement_candidate",
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        label="Reinforcement Candidate",
        description="Temporary registry entry for reward-guided rollouts.",
    )
    evaluation_results, evaluation_summary = evaluate_records(
        prompts,
        registry_path=registry_path,
        model_variant="reinforcement_candidate",
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        use_cache=False,
    )
    rollout_train, rollout_eval = build_rollout_records(evaluation_results, seed=args.seed)
    rollout_dataset_dir = layout.datasets_dir
    train_path = rollout_dataset_dir / "train.jsonl"
    eval_path = rollout_dataset_dir / "eval.jsonl"
    test_path = rollout_dataset_dir / "test.jsonl"
    manifest_path = rollout_dataset_dir / "manifest.json"
    write_jsonl(train_path, rollout_train)
    write_jsonl(eval_path, rollout_eval)
    write_jsonl(test_path, rollout_eval)
    write_json(
        manifest_path,
        {
            "manifest_type": "reinforcement_rollout",
            "prompt_source_manifest": prompt_corpus["manifest_path"],
            "accepted_rollouts": len(rollout_train) + len(rollout_eval),
            "evaluation_summary": evaluation_summary,
        },
    )
    write_json(layout.reports_dir / "reinforcement_rollout_summary.json", evaluation_summary)
    config_payload = build_training_config_payload(
        workflow_name="reinforcement",
        run_name=args.run_name,
        base_model_name=args.base_model,
        train_file=str(train_path),
        eval_file=str(eval_path),
        test_file=str(test_path),
        pack_manifest=str(manifest_path),
        method="qlora",
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        run_description="Reward-guided reinforcement-style fine-tuning over accepted model rollouts.",
    )
    config_path = write_training_config(layout=layout, filename="reinforcement_training.yaml", payload=config_payload)
    print(
        json.dumps(
            {
                "workflow": "reinforcement",
                "run_name": args.run_name,
                "config_path": str(config_path),
                "registry_path": str(registry_path),
                "accepted_rollouts": len(rollout_train) + len(rollout_eval),
                "summary": evaluation_summary,
            },
            indent=2,
        )
    )
    if not args.prepare_only:
        launch_training(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
