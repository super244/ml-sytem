#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from peft import PeftConfig

from training.src.config import load_experiment_config
from training.src.final_suite import (
    DEFAULT_FINAL_PROFILE,
    REPO_ROOT,
    build_candidate_registry,
    build_followup_training_config,
    resolve_latest_run_dir,
    resolve_published_adapter,
    run_command,
    run_inference_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the codex final training suite: cleanup, training, final evaluation, inference smoke, and visual refresh."
    )
    parser.add_argument(
        "--train-config",
        default=str(DEFAULT_FINAL_PROFILE.relative_to(REPO_ROOT)),
        help="Training profile to launch for the main run.",
    )
    parser.add_argument(
        "--eval-config",
        default="evaluation/configs/final_eval_suite.yaml",
        help="Evaluation suite definition for the post-training loop.",
    )
    parser.add_argument(
        "--run-dir",
        help="Existing run directory to evaluate instead of launching training.",
    )
    parser.add_argument(
        "--optimization-loops",
        type=int,
        default=1,
        help="How many train/eval passes to chain. Loop 2+ uses a short follow-up config generated from the previous adapter.",
    )
    parser.add_argument(
        "--secondary-model",
        default="boss_qwen3_5",
        help="Registry name to compare the new candidate against during the suite.",
    )
    parser.add_argument(
        "--candidate-name",
        default="final_accuracy_model",
        help="Temporary registry alias for the newly trained adapter.",
    )
    parser.add_argument(
        "--candidate-label",
        default="Final Accuracy Model",
        help="Temporary label for the newly trained adapter.",
    )
    parser.add_argument(
        "--inference-dataset",
        default="data/processed/test.jsonl",
        help="Dataset used for the final inference smoke test.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional sample cap applied across the evaluation suite.",
    )
    parser.add_argument("--skip-cleanup", action="store_true", help="Do not remove transient caches before training.")
    parser.add_argument(
        "--skip-training", action="store_true", help="Do not launch training; use --run-dir or the latest run."
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip the final evaluation suite.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference smoke test.")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip chart regeneration.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    initial_config_path = (REPO_ROOT / args.train_config).resolve()
    active_config_path = initial_config_path
    config = load_experiment_config(str(active_config_path))
    run_name = config.run_name
    current_run_dir = Path(args.run_dir).resolve() if args.run_dir else None

    if args.dry_run and current_run_dir is None and args.skip_training:
        raise ValueError(
            "--dry-run with --skip-training requires --run-dir so the suite knows which adapter to inspect."
        )

    if not args.skip_cleanup:
        run_command([sys.executable, "scripts/cleanup_training_cache.py"], dry_run=args.dry_run)

    for loop_index in range(1, max(args.optimization_loops, 1) + 1):
        if not args.skip_training:
            run_command(
                [sys.executable, "-m", "training.train", "--config", str(active_config_path)],
                dry_run=args.dry_run,
            )
            current_run_dir = None

        if args.dry_run and current_run_dir is None:
            print(
                "Dry run complete for training stage. Re-run with --run-dir or without --dry-run to continue the suite."
            )
            return

        if current_run_dir is None or not current_run_dir.exists():
            current_run_dir = resolve_latest_run_dir(run_name, config.training.artifacts_dir)

        adapter_path = resolve_published_adapter(current_run_dir)
        adapter_config = PeftConfig.from_pretrained(str(adapter_path))
        loop_suffix = f"loop{loop_index:02d}"

        if not args.skip_eval:
            eval_command = [
                sys.executable,
                "evaluation/final_eval_optimize.py",
                "--config",
                args.eval_config,
                "--primary-adapter",
                str(adapter_path),
                "--primary-name",
                args.candidate_name,
                "--primary-label",
                args.candidate_label,
                "--primary-base-model",
                adapter_config.base_model_name_or_path,
                "--secondary-model",
                args.secondary_model,
            ]
            if args.max_eval_samples is not None:
                eval_command.extend(["--max-eval-samples", str(args.max_eval_samples)])
            if args.dry_run:
                eval_command.append("--dry-run")
            run_command(eval_command, dry_run=False)

            if not args.dry_run:
                suite_root = REPO_ROOT / "evaluation/results/final_suite"
                _copy_if_exists(
                    suite_root / "suite_summary.json",
                    current_run_dir / "reports" / f"final_suite_summary_{loop_suffix}.json",
                )
                _copy_if_exists(
                    suite_root / "recommendation.json",
                    current_run_dir / "reports" / f"final_suite_recommendation_{loop_suffix}.json",
                )

        if not args.skip_inference and not args.dry_run:
            runtime_registry = build_candidate_registry(
                registry_path=REPO_ROOT / "inference/configs/model_registry.yaml",
                output_path=current_run_dir / "manifests" / "final_suite" / f"runtime_registry_{loop_suffix}.yaml",
                candidate_name=args.candidate_name,
                candidate_label=args.candidate_label,
                adapter_path=adapter_path,
            )
            run_inference_smoke(
                registry_path=runtime_registry,
                model_name=args.secondary_model,
                output_path=current_run_dir / "reports" / f"inference_smoke_{loop_suffix}_reference.json",
                dataset_path=args.inference_dataset,
            )
            run_inference_smoke(
                registry_path=runtime_registry,
                model_name=args.candidate_name,
                output_path=current_run_dir / "reports" / f"inference_smoke_{loop_suffix}.json",
                dataset_path=args.inference_dataset,
            )

        if not args.skip_visualizations:
            run_command([sys.executable, "generate_visualizations.py"], dry_run=args.dry_run)

        if loop_index >= args.optimization_loops:
            break

        next_run_name = f"{config.run_name}_loop{loop_index + 1:02d}"
        generated_config_path = REPO_ROOT / "training/configs/generated" / f"{next_run_name}.yaml"
        active_config_path = build_followup_training_config(
            source_config_path=active_config_path,
            output_path=generated_config_path,
            next_run_name=next_run_name,
            adapter_path=adapter_path,
            overrides={
                "training": {
                    "num_train_epochs": 2.0,
                    "learning_rate": 4.0e-5,
                }
            },
        )
        config = load_experiment_config(str(active_config_path))
        run_name = config.run_name
        current_run_dir = None

    print(f"Final training suite complete. Latest run: {current_run_dir}")


if __name__ == "__main__":
    main()
