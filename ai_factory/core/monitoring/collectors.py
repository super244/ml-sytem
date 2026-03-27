from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ai_factory.core.discovery import latest_training_run, list_training_runs
from ai_factory.core.io import load_json, read_jsonl
from ai_factory.core.instances.models import InstanceManifest, MetricPoint
from ai_factory.core.monitoring.metrics import metric_points_from_summary


def _training_run_for_manifest(manifest: InstanceManifest, snapshot: dict[str, Any]) -> dict[str, Any] | None:
    raw = snapshot.get("resolved_subsystem_config") or {}
    run_name = raw.get("run_name")
    artifacts_dir = ((raw.get("training") or {}).get("artifacts_dir")) or "artifacts"
    runs = list_training_runs(artifacts_dir)
    if not run_name:
        return latest_training_run(runs)
    matching = [run for run in runs if run.get("run_name") == run_name]
    return latest_training_run(matching)


def _training_metrics(manifest: InstanceManifest, snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    run = _training_run_for_manifest(manifest, snapshot)
    if run is None:
        return {}, [], {}
    run_dir = Path(str(run["output_dir"]))
    metrics = load_json(run_dir / "metrics" / "metrics.json", default={}) or {}
    dataset = load_json(run_dir / "metrics" / "dataset_report.json", default={}) or {}
    model_report = load_json(run_dir / "metrics" / "model_report.json", default={}) or {}
    series_rows = read_jsonl(run_dir / "logs" / "training_metrics.jsonl")
    points: list[MetricPoint] = []
    for row in series_rows:
        step = row.get("step")
        for key, value in row.items():
            if key in {"step", "epoch"} or not isinstance(value, (int, float)):
                continue
            tags = {"stage": manifest.type}
            if step is not None:
                tags["step"] = str(step)
            points.append(MetricPoint(name=key, value=value, tags=tags))
    summary = {
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
        "train_rows": dataset.get("tokenized_train_rows") or dataset.get("train", {}).get("num_rows"),
        "eval_rows": dataset.get("tokenized_eval_rows") or dataset.get("eval", {}).get("num_rows"),
        "trainable_ratio": model_report.get("trainable_ratio"),
    }
    run_manifest = load_json(run_dir / "manifests" / "run_manifest.json", default={}) or {}
    refs = {
        "run_dir": str(run_dir),
        "run_manifest": str(run_dir / "manifests" / "run_manifest.json"),
        "training_metrics": str(run_dir / "logs" / "training_metrics.jsonl"),
        "base_model": run_manifest.get("base_model"),
    }
    published = (run_manifest.get("metadata") or {}).get("published", {})
    if published:
        refs["published"] = published
    return summary, points, refs


def _evaluation_metrics(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    raw = snapshot.get("resolved_subsystem_config") or {}
    output_dir = Path(str(raw.get("output_dir") or "evaluation/results"))
    summary_path = output_dir / "summary.json"
    summary = load_json(summary_path, default={}) or {}
    primary = summary.get("primary") or {}
    compact = {
        "accuracy": primary.get("accuracy"),
        "parse_rate": primary.get("parse_rate"),
        "verifier_agreement_rate": primary.get("verifier_agreement_rate"),
        "no_answer_rate": primary.get("no_answer_rate"),
        "avg_latency_s": primary.get("avg_latency_s"),
    }
    refs = {
        "evaluation_dir": str(output_dir),
        "summary_json": str(summary_path),
        "summary_markdown": str(output_dir / "summary.md"),
        "leaderboard": str(output_dir / "leaderboard.json"),
        "per_example": str(output_dir / "per_example.jsonl"),
    }
    return compact, metric_points_from_summary(compact, stage="evaluate"), refs


def _inference_metrics(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    telemetry_path = (
        ((snapshot.get("execution") or {}).get("env") or {}).get("INFERENCE_TELEMETRY_PATH")
        or "artifacts/inference/telemetry/requests.jsonl"
    )
    rows = read_jsonl(telemetry_path)
    latencies = [row.get("latency_s") for row in rows if isinstance(row.get("latency_s"), (int, float))]
    cache_hits = sum(1 for row in rows if row.get("cache_hit"))
    summary = {
        "requests": len(rows),
        "cache_hits": cache_hits,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
    }
    refs = {"telemetry": telemetry_path}
    return summary, metric_points_from_summary(summary, stage="inference"), refs


def _deploy_metrics(manifest: InstanceManifest) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    summary = {"status": manifest.status}
    return summary, metric_points_from_summary(summary, stage="deploy"), {}


def _gpu_snapshot() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    gpu_rows: list[dict[str, int]] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_rows.append(
            {
                "utilization_gpu": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
            }
        )
    return {"gpus": gpu_rows} if gpu_rows else None


def collect_metrics_for_instance(
    manifest: InstanceManifest,
    snapshot: dict[str, Any],
    *,
    collect_gpu: bool = True,
) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    if manifest.type in {"finetune", "train"}:
        summary, points, refs = _training_metrics(manifest, snapshot)
    elif manifest.type == "evaluate":
        summary, points, refs = _evaluation_metrics(snapshot)
    elif manifest.type == "inference":
        summary, points, refs = _inference_metrics(snapshot)
    else:
        summary, points, refs = _deploy_metrics(manifest)
    if collect_gpu:
        gpu = _gpu_snapshot()
        if gpu:
            summary["gpu"] = gpu
    return summary, points, refs
