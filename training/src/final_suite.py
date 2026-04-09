from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from peft import PeftConfig

from ai_factory.core.io import read_jsonl, write_json
from inference.app.generation import MathGenerator
from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml
from inference.app.parameters import GenerationParameters
from inference.app.prompts import load_prompt_presets
from training.src.config import ExperimentConfig, load_experiment_config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_PROFILE = REPO_ROOT / "training/configs/profiles/codex_accuracy_final.yaml"
DEFAULT_REGISTRY = REPO_ROOT / "inference/configs/model_registry.yaml"
DEFAULT_PROMPT_LIBRARY = REPO_ROOT / "inference/configs/prompt_presets.yaml"
DEFAULT_BENCHMARKS = ["core_eval", "benchmark_holdout", "calculus_hard", "verification_suite"]
DEFAULT_REFERENCES = [
    ("boss_qwen3_5", "BOSS Qwen3.5 Conversational"),
    ("base", "Base Qwen2.5-Math-1.5B-Instruct"),
]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    rendered = " ".join(command)
    logger.info("Running: %s", rendered)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def resolve_latest_run_dir(run_name: str, artifacts_dir: str | Path = "artifacts") -> Path:
    runs_root = _resolve_repo_path(artifacts_dir) / "runs"
    candidates = sorted(path for path in runs_root.glob(f"{run_name}-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found for run_name={run_name!r} under {runs_root}")
    return candidates[-1]


def load_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "manifests" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def resolve_published_adapter(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    manifest = load_run_manifest(run_path)
    published = ((manifest.get("metadata") or {}).get("published") or {}).get("final_adapter")
    if published:
        candidate = _resolve_repo_path(published)
        if candidate.exists():
            return candidate
    fallback = run_path / "published" / "final_adapter"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Published adapter not found for run: {run_path}")


def build_candidate_registry(
    *,
    registry_path: str | Path,
    output_path: str | Path,
    candidate_name: str,
    candidate_label: str,
    adapter_path: str | Path,
) -> Path:
    registry_payload = yaml.safe_load(_resolve_repo_path(registry_path).read_text()) or {}
    adapter_path = _resolve_repo_path(adapter_path)
    adapter_config = PeftConfig.from_pretrained(str(adapter_path))
    models = list(registry_payload.get("models") or [])
    models = [model for model in models if model.get("name") != candidate_name]
    models.append(
        {
            "name": candidate_name,
            "label": candidate_label,
            "base_model": adapter_config.base_model_name_or_path,
            "adapter_path": str(adapter_path),
            "load_in_4bit": False,
            "load_in_8bit": False,
            "dtype": "float32",
            "target_parameters": "2b",
            "parameter_size_b": 2.0,
            "quantization": "none",
            "tier": "candidate",
            "description": f"Final suite candidate from {adapter_path.name}.",
            "tags": ["candidate", "final_suite", "qwen3_5", "local"],
            "scale_tags": ["2b", "candidate", "final_suite", "local"],
        }
    )
    registry_payload["models"] = models
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(registry_payload, sort_keys=False))
    return output_path


def build_eval_config(
    *,
    output_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    benchmark_id: str,
    primary_model: str,
    primary_label: str,
    secondary_model: str,
    secondary_label: str,
    max_eval_samples: int | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "name": f"{primary_model}_vs_{secondary_model}_{benchmark_id}",
        "prompt_library_path": str(DEFAULT_PROMPT_LIBRARY.relative_to(REPO_ROOT)),
        "output_dir": str(Path(output_dir).relative_to(REPO_ROOT)),
        "benchmark": {
            "registry_path": "evaluation/benchmarks/registry.yaml",
            "benchmark_id": benchmark_id,
        },
        "models": {
            "registry_path": str(Path(registry_path).relative_to(REPO_ROOT)),
            "primary_model": primary_model,
            "secondary_model": secondary_model,
            "primary_label": primary_label,
            "secondary_label": secondary_label,
        },
        "generation": {
            "prompt_preset": "atlas_rigorous",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_new_tokens": 512,
            "num_samples": 1,
            "show_reasoning": False,
            "difficulty_target": "mixed",
            "use_calculator": False,
            "solver_mode": "fast",
            "output_format": "text",
            "use_cache": True,
        },
    }
    if max_eval_samples is not None:
        payload["generation"]["max_eval_samples"] = max_eval_samples
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return output_path


def run_evaluation_matrix(
    *,
    run_dir: str | Path,
    candidate_name: str,
    candidate_label: str,
    registry_path: str | Path,
    benchmarks: list[str],
    references: list[tuple[str, str]],
    max_eval_samples: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    eval_root = REPO_ROOT / "evaluation/results/final_suite" / run_path.name
    config_root = run_path / "manifests" / "final_suite" / "evaluation_configs"
    summaries: list[dict[str, Any]] = []
    for benchmark_id in benchmarks:
        for secondary_model, secondary_label in references:
            eval_name = f"{candidate_name}_vs_{secondary_model}_{benchmark_id}"
            output_dir = eval_root / eval_name
            config_path = config_root / f"{eval_name}.yaml"
            build_eval_config(
                output_path=config_path,
                output_dir=output_dir,
                registry_path=registry_path,
                benchmark_id=benchmark_id,
                primary_model=candidate_name,
                primary_label=candidate_label,
                secondary_model=secondary_model,
                secondary_label=secondary_label,
                max_eval_samples=max_eval_samples,
            )
            run_command(
                [sys.executable, "evaluation/evaluate.py", "--config", str(config_path)],
                dry_run=dry_run,
            )
            summary_path = output_dir / "summary.json"
            if dry_run or not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            summary["suite_metadata"] = {
                "benchmark_id": benchmark_id,
                "reference_model": secondary_model,
                "reference_label": secondary_label,
                "output_dir": str(output_dir),
            }
            summaries.append(summary)
    return summaries


def _find_summary_for_reference(
    summaries: list[dict[str, Any]], *, benchmark_id: str, reference_model: str
) -> dict[str, Any] | None:
    for summary in summaries:
        meta = summary.get("suite_metadata") or {}
        if meta.get("benchmark_id") == benchmark_id and meta.get("reference_model") == reference_model:
            return summary
    return None


def build_optimization_recommendation(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {
            "overall": {},
            "weakest_benchmarks": [],
            "recommendations": ["Run the evaluation matrix first so the optimization loop has signal."],
            "training_overrides": {},
        }

    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        benchmark_id = str((summary.get("suite_metadata") or {}).get("benchmark_id") or "unknown")
        by_benchmark.setdefault(benchmark_id, []).append(summary)

    benchmark_rows: list[dict[str, Any]] = []
    for benchmark_id, benchmark_summaries in by_benchmark.items():
        primary_accuracies = [float(item["primary"]["accuracy"]) for item in benchmark_summaries]
        formatting_failure = max(float(item["primary"]["formatting_failure_rate"]) for item in benchmark_summaries)
        no_answer_rate = max(float(item["primary"]["no_answer_rate"]) for item in benchmark_summaries)
        error_counts: Counter[str] = Counter()
        for item in benchmark_summaries:
            error_counts.update(item["primary"].get("error_types") or {})
        benchmark_rows.append(
            {
                "benchmark_id": benchmark_id,
                "accuracy": sum(primary_accuracies) / len(primary_accuracies),
                "formatting_failure_rate": formatting_failure,
                "no_answer_rate": no_answer_rate,
                "top_error_types": dict(error_counts.most_common(3)),
            }
        )

    benchmark_rows.sort(key=lambda item: (item["accuracy"], item["formatting_failure_rate"]))
    weakest = benchmark_rows[: min(3, len(benchmark_rows))]

    recommendations: list[str] = []
    training_overrides: dict[str, Any] = {"training": {}, "data": {}}
    overall_errors: Counter[str] = Counter()
    formatting_failure = 0.0
    no_answer_rate = 0.0
    for summary in summaries:
        overall_errors.update(summary["primary"].get("error_types") or {})
        formatting_failure = max(formatting_failure, float(summary["primary"]["formatting_failure_rate"]))
        no_answer_rate = max(no_answer_rate, float(summary["primary"]["no_answer_rate"]))

    if formatting_failure >= 0.12:
        recommendations.append(
            "Formatting failures remain elevated; add stronger final-answer formatting pressure and mine those errors into the next replay pack."
        )
        training_overrides["data"]["verification_boost"] = 1.35
    if no_answer_rate >= 0.08:
        recommendations.append(
            "Some samples still fail to terminate with an answer; increase answer-focused replay and keep max_new_tokens generous on validation sweeps."
        )
        training_overrides["data"]["failure_replay_boost"] = 1.7
    verification_summary = _find_summary_for_reference(
        summaries, benchmark_id="verification_suite", reference_model="boss_qwen3_5"
    )
    if verification_summary and float(verification_summary["primary"]["accuracy"]) < 0.5:
        recommendations.append(
            "Verification-suite accuracy is lagging; preserve step anchors, oversample verifier-backed rows, and bias the next loop toward checked solutions."
        )
        training_overrides["data"]["verification_boost"] = max(
            float(training_overrides["data"].get("verification_boost", 1.25)),
            1.45,
        )
    holdout_summary = _find_summary_for_reference(
        summaries, benchmark_id="benchmark_holdout", reference_model="boss_qwen3_5"
    )
    core_summary = _find_summary_for_reference(summaries, benchmark_id="core_eval", reference_model="boss_qwen3_5")
    if holdout_summary and core_summary:
        if float(holdout_summary["primary"]["accuracy"]) + 0.05 < float(core_summary["primary"]["accuracy"]):
            recommendations.append(
                "Holdout accuracy trails core eval, which suggests overfitting. Keep the teacher, lower LR slightly, and bias the next loop toward failure-mined/generalization slices."
            )
            training_overrides["training"]["learning_rate"] = 4.0e-5
            training_overrides["data"]["failure_replay_boost"] = max(
                float(training_overrides["data"].get("failure_replay_boost", 1.5)),
                1.8,
            )
    if not recommendations:
        recommendations.append(
            "The candidate is stable across the sampled benchmarks. If you run another loop, keep the teacher and do a short 1-2 epoch failure-mining pass rather than a full retrain."
        )

    training_overrides["training"]["num_train_epochs"] = 2.0
    return {
        "overall": {
            "num_evaluations": len(summaries),
            "overall_top_errors": dict(overall_errors.most_common(5)),
            "max_formatting_failure_rate": formatting_failure,
            "max_no_answer_rate": no_answer_rate,
        },
        "weakest_benchmarks": weakest,
        "recommendations": recommendations,
        "training_overrides": training_overrides,
    }


def absolutize_refs(raw_payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    refs = raw_payload.get("refs")
    if not isinstance(refs, dict):
        return raw_payload
    updated = dict(raw_payload)
    updated["refs"] = {
        key: str((source_path.parent / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in refs.items()
    }
    return updated


def build_followup_training_config(
    *,
    source_config_path: str | Path,
    output_path: str | Path,
    next_run_name: str,
    adapter_path: str | Path,
    overrides: dict[str, Any],
) -> Path:
    source_config_path = _resolve_repo_path(source_config_path)
    raw_payload = yaml.safe_load(source_config_path.read_text()) or {}
    raw_payload = absolutize_refs(raw_payload, source_config_path)
    raw_payload["run_name"] = next_run_name
    raw_payload.setdefault("model", {})
    raw_payload["model"]["base_model_name"] = str(_resolve_repo_path(adapter_path))
    for section in ("training", "data", "model"):
        section_overrides = overrides.get(section)
        if not section_overrides:
            continue
        raw_payload.setdefault(section, {})
        raw_payload[section].update(section_overrides)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw_payload, sort_keys=False))
    return output_path


def write_optimization_report(run_dir: str | Path, recommendation: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = Path(run_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "final_suite_optimization.json"
    md_path = report_dir / "final_suite_optimization.md"
    write_json(json_path, recommendation)

    lines = ["# Final Suite Optimization Report", ""]
    overall = recommendation.get("overall") or {}
    if overall:
        lines.extend(
            [
                f"- Evaluations run: {overall.get('num_evaluations', 0)}",
                f"- Top errors: {overall.get('overall_top_errors', {})}",
                f"- Max formatting failure rate: {overall.get('max_formatting_failure_rate', 0.0):.2%}",
                f"- Max no-answer rate: {overall.get('max_no_answer_rate', 0.0):.2%}",
                "",
            ]
        )
    lines.append("## Weakest Benchmarks")
    for row in recommendation.get("weakest_benchmarks") or []:
        lines.append(
            f"- `{row['benchmark_id']}`: accuracy={row['accuracy']:.2%}, "
            f"formatting_failure={row['formatting_failure_rate']:.2%}, "
            f"no_answer={row['no_answer_rate']:.2%}, top_errors={row['top_error_types']}"
        )
    lines.extend(["", "## Recommendations"])
    for item in recommendation.get("recommendations") or []:
        lines.append(f"- {item}")
    overrides = recommendation.get("training_overrides") or {}
    lines.extend(["", "## Suggested Overrides", "```json", json.dumps(overrides, indent=2), "```", ""])
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def run_inference_smoke(
    *,
    registry_path: str | Path,
    model_name: str,
    output_path: str | Path,
    dataset_path: str | Path,
    max_examples: int = 2,
) -> Path:
    registry = MathModelRegistry(load_registry_from_yaml(_resolve_repo_path(registry_path)))
    prompts = load_prompt_presets(DEFAULT_PROMPT_LIBRARY)
    generator = MathGenerator(registry, prompt_presets=prompts)
    records = read_jsonl(_resolve_repo_path(dataset_path))[:max_examples]
    results: list[dict[str, Any]] = []
    for record in records:
        params = GenerationParameters(
            question=record["question"],
            model_variant=model_name,
            prompt_preset="atlas_rigorous",
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=384,
            show_reasoning=False,
            difficulty_target=record.get("difficulty", "hard"),
            num_samples=1,
            use_calculator=False,
            solver_mode="fast",
            output_format="text",
            use_cache=False,
            reference_answer=record.get("final_answer"),
            step_checks=record.get("step_checks"),
        )
        output = generator.generate(params)
        results.append(
            {
                "id": record.get("id"),
                "difficulty": record.get("difficulty"),
                "question": record["question"],
                "reference_answer": record.get("final_answer"),
                "prediction": output.get("answer"),
                "verification": output.get("verification"),
                "latency_s": output.get("latency_s"),
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, {"model_name": model_name, "examples": results})
    return output_path


def prune_irrelevant_artifacts(
    config: ExperimentConfig, *, execute: bool, current_run_dir: str | Path | None = None
) -> dict[str, list[str]]:
    keep_paths = {
        str(_resolve_repo_path(config.data.tokenized_cache_dir))
        for value in [config.data.tokenized_cache_dir]
        if value
    }
    keep_paths.update(
        {
            str(_resolve_repo_path(config.teacher_student.teacher_cache_dir))
            for value in [config.teacher_student.teacher_cache_dir if config.teacher_student else None]
            if value
        }
    )
    if current_run_dir:
        keep_paths.add(str(Path(current_run_dir).resolve()))

    candidates: list[Path] = []
    tokenized_cache = REPO_ROOT / "data/processed/.tokenized_cache"
    if tokenized_cache.exists():
        candidates.append(tokenized_cache)
    for directory in (REPO_ROOT / "cache").glob("tokenized/*"):
        candidates.append(directory)
    for pattern in ("__pycache__", ".pytest_cache", ".mypy_cache"):
        candidates.extend(path for path in REPO_ROOT.rglob(pattern) if path.is_dir())

    removed: list[str] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved in keep_paths:
            skipped.append(resolved)
            continue
        if any(keep_path.startswith(resolved) for keep_path in keep_paths):
            skipped.append(resolved)
            continue
        if execute:
            shutil.rmtree(candidate, ignore_errors=True)
            removed.append(resolved)
        else:
            skipped.append(f"dry-run:{resolved}")
    return {"removed": removed, "skipped": skipped}

