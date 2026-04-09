#!/usr/bin/env python3
"""Run the final evaluation suite and emit the restart recommendation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Evaluate a checkpoint across the final math suite and suggest the next
        training rerun once the bottleneck is identified."""
    )
    parser.add_argument("--config", required=True, help="Suite definition YAML.")
    parser.add_argument(
        "--primary-adapter",
        help="Path to the recently produced adapter that should be registered for evaluation.",
    )
    parser.add_argument(
        "--primary-name",
        default="final_accuracy_model",
        help="Name for the temporary registry entry that points to --primary-adapter.",
    )
    parser.add_argument(
        "--primary-label",
        default="Final Accuracy Model",
        help="Label for the temporary registry entry.",
    )
    parser.add_argument(
        "--primary-base-model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model for the temporary registry entry.",
    )
    parser.add_argument(
        "--secondary-model",
        help="Optional override for the comparison model name (defaults to what the suite defines)." ,
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap that is merged into every generation block.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--registry",
        default="inference/configs/model_registry.yaml",
        help="Base model registry to extend with the new adapter.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Suite config must be a mapping: {path}")
    return payload


def deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def make_temp_registry(
    base_registry: Path,
    adapter_path: Optional[str],
    name: str,
    label: str,
    base_model: str,
) -> Optional[Path]:
    if not adapter_path:
        return None
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter missing: {adapter_path}")
    base_payload = yaml.safe_load(base_registry.read_text()) or {}
    models = [
        entry
        for entry in base_payload.get("models", [])
        if entry.get("name") != name
    ]
    models.append(
        {
            "name": name,
            "label": label,
            "base_model": base_model,
            "adapter_path": adapter_path,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "dtype": "float32",
            "description": "Temp final evaluation adapter.",
        }
    )
    payload = {**base_payload, "models": models}
    fd, temp_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    path = Path(temp_path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def build_run_config(
    suite_base: dict[str, Any],
    entry: dict[str, Any],
    run_label: str,
    output_dir: Path,
    temp_registry: Optional[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    merged = deep_merge(suite_base, {k: v for k, v in entry.items() if k not in {"label", "description"}})
    merged["name"] = f"{suite_base.get('name', 'final_suite')}-{run_label}"
    merged["output_dir"] = str(output_dir)
    models = merged.setdefault("models", {})
    if temp_registry:
        models["registry_path"] = str(temp_registry)
        models["primary_model"] = args.primary_name
    if args.secondary_model:
        models["secondary_model"] = args.secondary_model
    generation = merged.setdefault("generation", {})
    if args.max_eval_samples is not None:
        generation["max_eval_samples"] = args.max_eval_samples
    return merged


def run_evaluation(config_path: Path, dry_run: bool) -> None:
    cmd = [sys.executable, "-m", "evaluation.evaluate", "--config", str(config_path)]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def load_summary(summary_dir: Path) -> dict[str, Any]:
    summary_path = summary_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {summary_dir}")
    return json.loads(summary_path.read_text())


def build_suite_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    report_entries: list[dict[str, Any]] = []
    for entry in results:
        primary = entry["summary"].get("primary", {})
        report_entries.append(
            {
                "label": entry["label"],
                "benchmark_id": entry.get("benchmark_id"),
                "accuracy": primary.get("accuracy", 0.0),
                "formatting_failure": primary.get("formatting_failure_rate", 0.0),
                "arithmetic_slip": primary.get("arithmetic_slip_rate", 0.0),
                "solve_rate": primary.get("solve_rate", 0.0),
                "error_types": primary.get("error_types", {}),
                "summary_dir": entry["output_dir"],
                "description": entry.get("description"),
            }
        )
    accuracies = [item["accuracy"] for item in report_entries]
    weakest = min(report_entries, key=lambda item: item["accuracy"]) if report_entries else None
    return {
        "runs": report_entries,
        "mean_accuracy": mean(accuracies) if accuracies else 0.0,
        "weakest_run": weakest,
    }


def describe_error_types(entry: dict[str, Any]) -> Optional[str]:
    errors = entry.get("error_types") or {}
    if not errors:
        return None
    return max(errors.items(), key=lambda item: item[1])[0]


def build_recommendation(summary: dict[str, Any]) -> dict[str, Any]:
    weakest = summary.get("weakest_run")
    adjustments: list[str] = []
    note_sections: list[str] = []
    if weakest:
        accuracy = weakest.get("accuracy", 0.0)
        if accuracy < 0.6:
            adjustments.append(
                "Extend codex_accuracy_final to >=6 epochs, keep gradient accumulation high, and monitor hard-negative replay."
            )
        if weakest.get("formatting_failure", 0.2) > 0.2:
            adjustments.append(
                "Raise format_regularization_weight and push more \\boxed{} examples through the curriculum."
            )
        if weakest.get("arithmetic_slip", 0.15) > 0.15:
            adjustments.append(
                "Increase failure bank budget and emphasize hard negative mining for arithmetic slip cases."
            )
        top_error = describe_error_types(weakest)
        if top_error:
            adjustments.append(f"Target the {top_error} error type by replaying similar failure cases.")
        note_sections.append(
            f"Weakest suite leg: " f"{weakest['label']} ({weakest.get('benchmark_id')}) @ {accuracy*100:.1f}% accuracy."
        )
    rerun_command = "python -m training.train --config training/configs/profiles/codex_accuracy_final.yaml"
    return {
        "weakest_run": weakest["label"] if weakest else None,
        "mean_accuracy": summary.get("mean_accuracy"),
        "adjustments": adjustments or [
            "Re-run codex_accuracy_final with the default final weights if no single bottleneck is visible."
        ],
        "notes": note_sections,
        "rerun_command": rerun_command,
        "rerun_profile": "training/configs/profiles/codex_accuracy_final.yaml",
    }


def main() -> None:
    args = parse_args()
    suite_path = Path(args.config)
    suite_payload = load_yaml(suite_path)
    suite_entries = suite_payload.get("suite") or []
    if not suite_entries:
        raise ValueError("Suite config must include a non-empty 'suite' list.")
    suite_base = {k: v for k, v in suite_payload.items() if k != "suite"}
    base_output = Path(suite_base.get("output_dir", "evaluation/results/final_suite"))
    base_output.mkdir(parents=True, exist_ok=True)
    temp_registry = make_temp_registry(
        Path(args.registry),
        args.primary_adapter,
        args.primary_name,
        args.primary_label,
        args.primary_base_model,
    )
    results: list[dict[str, Any]] = []
    try:
        for entry in suite_entries:
            label = entry.get("label") or entry.get("benchmark", {}).get("benchmark_id") or "run"
            description = entry.get("description")
            run_output = base_output / label
            run_output.mkdir(parents=True, exist_ok=True)
            run_config = build_run_config(
                suite_base,
                entry,
                label,
                run_output,
                temp_registry,
                args,
            )
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as handle:
                handle.write(yaml.safe_dump(run_config, sort_keys=False).encode())
                temp_config = Path(handle.name)
            run_evaluation(temp_config, args.dry_run)
            if not args.dry_run:
                summary = load_summary(run_output)
                results.append(
                    {
                        "label": label,
                        "description": description,
                        "benchmark_id": run_config.get("benchmark", {}).get("benchmark_id"),
                        "summary": summary,
                        "output_dir": str(run_output),
                    }
                )
            temp_config.unlink(missing_ok=True)
    finally:
        if temp_registry:
            temp_registry.unlink(missing_ok=True)
    if args.dry_run:
        return
    suite_summary = build_suite_summary(results)
    summary_path = base_output / "suite_summary.json"
    summary_path.write_text(json.dumps(suite_summary, indent=2))
    recommendation = build_recommendation(suite_summary)
    recommendation_path = base_output / "recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation, indent=2))
    print(f"✅ Suite complete. Aggregated summary at: {summary_path}")
    print(f"✅ Recommendation / rerun command written to: {recommendation_path}")


if __name__ == "__main__":
    main()
