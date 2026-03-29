from __future__ import annotations

import argparse

from common import active_python, run_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the local AI-Factory workspace.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument("--skip-train-dry-run", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--profile",
        default="training/configs/profiles/baseline_qlora.yaml",
        help="Training profile to use for the dry-run validation step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = active_python()

    if not args.skip_generate:
        run_step(
            "Generate synthetic datasets",
            [python, "data/generator/generate_calculus_datasets.py", "--config", "data/configs/generation.yaml"],
        )
    run_step(
        "Prepare processed corpus",
        [python, "data/prepare_dataset.py", "--config", "data/configs/processing.yaml"],
    )
    run_step(
        "Validate processed corpus",
        [
            python,
            "data/tools/validate_dataset.py",
            "--input",
            "data/processed/train.jsonl",
            "--manifest",
            "data/processed/manifest.json",
        ],
    )
    if not args.skip_notebooks:
        run_step("Refresh notebook lab", [python, "notebooks/build_notebooks.py"])
    if not args.skip_train_dry_run:
        run_step(
            "Training dry-run",
            [python, "-m", "training.train", "--config", args.profile, "--dry-run"],
        )
    if not args.skip_tests:
        run_step("Pytest suite", [python, "-m", "pytest"])

    print("[ai-factory] Workspace refresh complete.")


if __name__ == "__main__":
    main()
