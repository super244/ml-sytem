from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.src.workflows import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    build_training_config_payload,
    build_workflow_corpus,
    discover_private_categories,
    launch_training,
    parse_csv_values,
    write_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and launch supervised training over public and/or private datasets."
    )
    parser.add_argument("--run-name", default="supervised-workflow")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--method", choices=["qlora", "lora", "full"], default="qlora")
    parser.add_argument("--public-datasets", default="")
    parser.add_argument("--private-categories", default="")
    parser.add_argument("--local-datasets", default="")
    parser.add_argument("--custom-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prompt_for_dataset_selection(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    public = parse_csv_values(args.public_datasets)
    private = parse_csv_values(args.private_categories)
    local = parse_csv_values(args.local_datasets)
    if public or private or local:
        return public, private, local

    source_mode = input("Choose dataset source [public/private/mixed]: ").strip().lower() or "private"
    if source_mode in {"public", "mixed"}:
        public = parse_csv_values(input("Enter Hugging Face dataset URLs or ids separated by commas: ").strip())
    if source_mode in {"private", "mixed"}:
        categories = ", ".join(discover_private_categories(args.custom_root))
        print(f"Available private categories: {categories}")
        private = parse_csv_values(input("Select private categories separated by commas: ").strip())
    if source_mode == "local":
        local = parse_csv_values(input("Enter local dataset paths separated by commas: ").strip())
    return public, private, local


def main() -> None:
    args = parse_args()
    public_datasets, private_categories, local_datasets = prompt_for_dataset_selection(args)
    corpus = build_workflow_corpus(
        workflow_name="supervised",
        run_name=args.run_name,
        public_datasets=public_datasets,
        private_categories=private_categories,
        local_datasets=local_datasets,
        custom_root=args.custom_root,
        seed=args.seed,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
    )
    config_payload = build_training_config_payload(
        workflow_name="supervised",
        run_name=args.run_name,
        base_model_name=args.base_model,
        train_file=corpus["train_file"],
        eval_file=corpus["eval_file"],
        test_file=corpus["test_file"],
        pack_manifest=corpus["manifest_path"],
        method=args.method,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        run_description="Supervised training over selected public/private datasets.",
    )
    config_path = write_training_config(
        layout=corpus["layout"],
        filename="supervised_training.yaml",
        payload=config_payload,
    )
    summary = {
        "workflow": "supervised",
        "run_name": args.run_name,
        "config_path": str(config_path),
        "dataset_dir": corpus["dataset_dir"],
        "sources": corpus["source_specs"],
    }
    print(json.dumps(summary, indent=2))
    if not args.prepare_only:
        launch_training(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
