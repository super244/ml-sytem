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
    build_workflow_layout,
    launch_training,
    write_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classic post-training over the already packed processed corpus.")
    parser.add_argument("--run-name", default="post-training")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = build_workflow_layout("post_training", args.run_name)
    config_payload = build_training_config_payload(
        workflow_name="post_training",
        run_name=args.run_name,
        base_model_name=args.base_model,
        train_file=str(Path("data/processed/train.jsonl").resolve()),
        eval_file=str(Path("data/processed/eval.jsonl").resolve()),
        test_file=str(Path("data/processed/test.jsonl").resolve()),
        pack_manifest=str(Path("data/processed/manifest.json").resolve()),
        method="sft",
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        run_description="Classic post-training pass over the canonical processed corpus.",
    )
    config_path = write_training_config(layout=layout, filename="post_training.yaml", payload=config_payload)
    print(
        json.dumps(
            {
                "workflow": "post_training",
                "run_name": args.run_name,
                "config_path": str(config_path),
                "dataset_dir": "data/processed",
            },
            indent=2,
        )
    )
    if not args.prepare_only:
        launch_training(config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

