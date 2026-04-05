from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from ai_factory.core.discovery import latest_training_run, list_training_runs
from ai_factory.core.instances.models import InstanceManifest, MetricPoint, ProgressSnapshot
from ai_factory.core.io import load_json, read_jsonl
from ai_factory.core.monitoring.metrics import build_observability_summary, metric_points_from_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU snapshot cache – avoid running nvidia-smi on every list_instances call.
# ---------------------------------------------------------------------------
_GPU_CACHE_TTL_S: float = 30.0
_gpu_cache: tuple[float, dict[str, Any] | None] | None = None  # (monotonic_ts, payload)
_NVIDIA_SMI_AVAILABLE: bool | None = None  # lazily determined
_NVIDIA_SMI_PATH: str | None = None
_TITAN_SMI_AVAILABLE: bool | None = None  # lazily determined


def _prepare_output_dir(snapshot: dict[str, Any]) -> Path:
    raw = snapshot.get("resolved_subsystem_config") or {}
    return Path(str(raw.get("output_dir") or "data/processed"))


def _prepare_metrics(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    output_dir = _prepare_output_dir(snapshot)
    manifest = load_json(output_dir / "manifest.json", default={}) or {}
    stats = load_json(output_dir / "stats.json", default={}) or {}
    pack_summary = load_json(output_dir / "pack_summary.json", default={}) or {}
    manifest_stats: dict[str, Any] = (
        cast(dict[str, Any], manifest.get("stats")) if isinstance(manifest.get("stats"), dict) else {}
    )
    summary: dict[str, Any] = {
        "records": (manifest_stats.get("num_records") or stats.get("num_records") or stats.get("records_total")),
        "train_rows": stats.get("train_rows"),
        "eval_rows": stats.get("eval_rows"),
        "test_rows": stats.get("test_rows"),
        "packs": len(pack_summary.get("packs") or []),
        "output_ready": output_dir.exists(),
    }
    refs = {
        "output_dir": str(output_dir),
        "manifest_json": str(output_dir / "manifest.json"),
        "stats_json": str(output_dir / "stats.json"),
        "pack_summary_json": str(output_dir / "pack_summary.json"),
        "card_markdown": str(output_dir / "card.md"),
    }
    points = metric_points_from_summary(summary, stage="prepare")
    summary["observability"] = build_observability_summary(points, summary, stage="prepare")
    summary["utilization_rollup"] = summary["observability"]["utilization_rollup"]
    return summary, points, refs


def _prepare_progress(snapshot: dict[str, Any]) -> ProgressSnapshot:
    output_dir = _prepare_output_dir(snapshot)
    manifest = load_json(output_dir / "manifest.json", default={}) or {}
    stats = load_json(output_dir / "stats.json", default={}) or {}
    completed = manifest.get("stats", {}).get("num_records") or stats.get("num_records")
    ready = output_dir.exists() and (output_dir / "manifest.json").exists()
    return ProgressSnapshot(
        stage="prepared" if ready else "preparing",
        status_message=(
            "Dataset preparation artifacts have been written." if ready else "Waiting for processed dataset artifacts."
        ),
        completed_steps=completed if isinstance(completed, int) else None,
        percent=1.0 if ready else 0.0,
        metrics={
            "records": completed if isinstance(completed, int) else None,
            "output_ready": ready,
        },
    )


def _training_run_for_manifest(manifest: InstanceManifest, snapshot: dict[str, Any]) -> dict[str, Any] | None:
    raw = snapshot.get("resolved_subsystem_config") or {}
    run_name = raw.get("run_name")
    artifacts_dir = ((raw.get("training") or {}).get("artifacts_dir")) or "artifacts"
    runs = list_training_runs(artifacts_dir)
    if not run_name:
        return latest_training_run(runs)
    matching = [run for run in runs if run.get("run_name") == run_name]
    return latest_training_run(matching)


def _training_metrics(
    manifest: InstanceManifest, snapshot: dict[str, Any]
) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
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
            tags: dict[str, str] = {"stage": str(manifest.type)}
            if step is not None:
                tags["step"] = str(step)
            points.append(MetricPoint(name=key, value=value, tags=tags))
    latest_row = series_rows[-1] if series_rows else {}
    summary = {
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
        "train_rows": dataset.get("tokenized_train_rows") or dataset.get("train", {}).get("num_rows"),
        "eval_rows": dataset.get("tokenized_eval_rows") or dataset.get("eval", {}).get("num_rows"),
        "trainable_ratio": model_report.get("trainable_ratio"),
        "latest_step": latest_row.get("step"),
        "latest_epoch": latest_row.get("epoch"),
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
    summary["observability"] = build_observability_summary(points, summary, stage=str(manifest.type))
    summary["utilization_rollup"] = summary["observability"]["utilization_rollup"]
    return summary, points, refs


def _training_progress(manifest: InstanceManifest, snapshot: dict[str, Any]) -> ProgressSnapshot | None:
    run = _training_run_for_manifest(manifest, snapshot)
    raw = snapshot.get("resolved_subsystem_config") or {}
    training = raw.get("training") or {}
    if run is None:
        if manifest.status == "pending":
            return ProgressSnapshot(stage="queued", status_message="Waiting for training resources.")
        return ProgressSnapshot(
            stage="training", status_message="Training run has started but no metrics are available yet."
        )
    run_dir = Path(str(run["output_dir"]))
    series_rows = read_jsonl(run_dir / "logs" / "training_metrics.jsonl")
    if not series_rows:
        return ProgressSnapshot(
            stage="training",
            status_message="Training run is active but has not emitted metric rows yet.",
        )
    latest = series_rows[-1]
    step = latest.get("step")
    total_steps = training.get("max_steps")
    try:
        total_steps = int(total_steps) if total_steps and str(total_steps).strip() not in ("", "-1") else None
    except (TypeError, ValueError):
        total_steps = None
    percent = None
    if isinstance(step, int) and isinstance(total_steps, int) and total_steps > 0:
        percent = min(step / total_steps, 1.0)
    metrics = {
        key: value for key, value in latest.items() if isinstance(value, (int, float)) and key not in {"step", "epoch"}
    }
    return ProgressSnapshot(
        stage="training",
        status_message="Training metrics are streaming.",
        completed_steps=step if isinstance(step, int) else None,
        total_steps=total_steps if isinstance(total_steps, int) and total_steps > 0 else None,
        percent=percent,
        metrics=metrics,
    )


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
    points = metric_points_from_summary(compact, stage="evaluate")
    observability = build_observability_summary(points, compact, stage="evaluate")
    compact["observability"] = observability
    compact["utilization_rollup"] = observability.get("utilization_rollup")
    return compact, points, refs


def _evaluation_progress(snapshot: dict[str, Any]) -> ProgressSnapshot | None:
    raw = snapshot.get("resolved_subsystem_config") or {}
    output_dir = Path(str(raw.get("output_dir") or "evaluation/results"))
    summary = load_json(output_dir / "summary.json", default={}) or {}
    primary = summary.get("primary") or {}
    per_example_rows = read_jsonl(output_dir / "per_example.jsonl")
    completed = len(per_example_rows)
    total = primary.get("num_examples")
    if not isinstance(total, int) or total <= 0:
        total = completed if summary else None
    percent = None
    if isinstance(total, int) and total > 0:
        percent = min(completed / total, 1.0)
    if summary:
        percent = 1.0
    return ProgressSnapshot(
        stage="evaluation",
        status_message="Benchmark evaluation has produced artifacts."
        if summary
        else "Evaluation is collecting example outputs.",
        completed_steps=completed or None,
        total_steps=total,
        percent=percent,
        metrics={key: value for key, value in primary.items() if isinstance(value, (int, float, bool))},
    )


def _inference_metrics(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    telemetry_path = ((snapshot.get("execution") or {}).get("env") or {}).get(
        "INFERENCE_TELEMETRY_PATH"
    ) or "artifacts/inference/telemetry/requests.jsonl"
    rows = read_jsonl(telemetry_path)
    latencies: list[float] = [float(row["latency_s"]) for row in rows if isinstance(row.get("latency_s"), (int, float))]
    prompt_tokens: list[float] = [
        float(row["prompt_tokens"]) for row in rows if isinstance(row.get("prompt_tokens"), (int, float))
    ]
    completion_tokens: list[float] = [
        float(row["completion_tokens"]) for row in rows if isinstance(row.get("completion_tokens"), (int, float))
    ]
    ttft_values: list[float] = [float(row["ttft_s"]) for row in rows if isinstance(row.get("ttft_s"), (int, float))]
    cache_hits = sum(1 for row in rows if row.get("cache_hit"))
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)
    total_latency_s = sum(latencies) if latencies else None
    summary: dict[str, Any] = {
        "requests": len(rows),
        "cache_hits": cache_hits,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "total_prompt_tokens": total_prompt_tokens or None,
        "total_completion_tokens": total_completion_tokens or None,
        "avg_tokens_per_second": (
            total_completion_tokens / total_latency_s if total_completion_tokens and total_latency_s else None
        ),
        "avg_time_to_first_token_s": (sum(t for t in ttft_values if t is not None) / len(ttft_values))
        if ttft_values
        else None,
    }
    refs = {"telemetry": telemetry_path}
    points = metric_points_from_summary(summary, stage="inference")
    observability = build_observability_summary(points, summary, stage="inference")
    summary["observability"] = observability
    summary["utilization_rollup"] = observability.get("utilization_rollup")
    return summary, points, refs


def _inference_progress(snapshot: dict[str, Any]) -> ProgressSnapshot | None:
    telemetry_path = ((snapshot.get("execution") or {}).get("env") or {}).get(
        "INFERENCE_TELEMETRY_PATH"
    ) or "artifacts/inference/telemetry/requests.jsonl"
    rows = read_jsonl(telemetry_path)
    latencies: list[float] = [float(row["latency_s"]) for row in rows if isinstance(row.get("latency_s"), (int, float))]
    completion_tokens: list[float] = [
        float(row["completion_tokens"]) for row in rows if isinstance(row.get("completion_tokens"), (int, float))
    ]
    total_latency_s = sum(latencies) if latencies else None
    total_completion_tokens = sum(completion_tokens)
    summary: dict[str, Any] = {
        "requests": len(rows),
        "cache_hits": sum(1 for row in rows if row.get("cache_hit")),
        "avg_tokens_per_second": (
            total_completion_tokens / total_latency_s if total_completion_tokens and total_latency_s else None
        ),
    }
    return ProgressSnapshot(
        stage="serving",
        status_message="Inference service is handling requests.",
        completed_steps=len(rows),
        metrics=summary,
    )


def _deploy_metrics(manifest: InstanceManifest) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    summary: dict[str, Any] = {"status": manifest.status}
    points = metric_points_from_summary(summary, stage="deploy")
    observability = build_observability_summary(points, summary, stage="deploy")
    summary["observability"] = observability
    summary["utilization_rollup"] = observability.get("utilization_rollup")
    return summary, points, {}


def _report_metrics(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    raw = snapshot.get("subsystem") or {}
    output_path = Path(str(raw.get("output_dir_override") or "evaluation/results/failure_analysis.json"))
    payload = load_json(output_path, default={}) or {}
    summary: dict[str, Any] = {
        "report_exists": output_path.exists(),
        "taxonomy_buckets": len(payload) if isinstance(payload, dict) else 0,
    }
    refs = {"report_json": str(output_path)}
    points = metric_points_from_summary(summary, stage="report")
    observability = build_observability_summary(points, summary, stage="report")
    summary["observability"] = observability
    summary["utilization_rollup"] = observability.get("utilization_rollup")
    return summary, points, refs


def _deploy_progress(manifest: InstanceManifest) -> ProgressSnapshot:
    status_map: dict[str, tuple[str, float | None, str | None]] = {
        "pending": ("queued", 0.0, "Deployment is queued."),
        "running": ("deploying", 0.5, "Deployment is in progress."),
        "completed": ("deployed", 1.0, "Deployment completed."),
        "failed": ("failed", 1.0, "Deployment failed."),
    }
    stage, percent, message = status_map.get(manifest.status, ("deploying", None, None))
    return ProgressSnapshot(stage=stage, percent=percent, status_message=message)


def _report_progress(snapshot: dict[str, Any]) -> ProgressSnapshot:
    raw = snapshot.get("subsystem") or {}
    output_path = Path(str(raw.get("output_dir_override") or "evaluation/results/failure_analysis.json"))
    return ProgressSnapshot(
        stage="reporting",
        status_message="Failure analysis report has been written."
        if output_path.exists()
        else "Waiting for failure analysis artifacts.",
        percent=1.0 if output_path.exists() else 0.0,
    )


def _gpu_snapshot() -> dict[str, Any] | None:
    """Return GPU utilisation snapshot from nvidia-smi, with caching and a timeout.

    • Checks ``shutil.which`` once so we skip the subprocess entirely on
      Apple-Silicon / CPU-only machines where ``nvidia-smi`` is absent.
    • Results are cached for ``_GPU_CACHE_TTL_S`` seconds to avoid running a
      subprocess on every ``list_instances`` call.
    • A 2-second ``timeout`` prevents the call from blocking indefinitely.
    """
    global _gpu_cache, _NVIDIA_SMI_AVAILABLE, _NVIDIA_SMI_PATH

    # Fast path: we already know nvidia-smi is not on this machine.
    if _NVIDIA_SMI_AVAILABLE is False:
        return None

    now = time.monotonic()

    # Return from cache if still fresh.
    if _gpu_cache is not None:
        cached_ts, cached_payload = _gpu_cache
        if now - cached_ts < _GPU_CACHE_TTL_S:
            return cached_payload

    # First call or cache expired – probe availability.
    if _NVIDIA_SMI_AVAILABLE is None:
        _NVIDIA_SMI_PATH = shutil.which("nvidia-smi")
        _NVIDIA_SMI_AVAILABLE = _NVIDIA_SMI_PATH is not None

    if not _NVIDIA_SMI_AVAILABLE:
        _gpu_cache = (now, None)
        return None

    nvidia_smi = _NVIDIA_SMI_PATH or "nvidia-smi"
    try:
        result = subprocess.run(  # nosec B603
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,  # Never block the main thread for more than 2 s
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi probe failed: %s", exc)
        _gpu_cache = (now, None)
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        _gpu_cache = (now, None)
        return None

    gpu_rows: list[dict[str, int]] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_rows.append(
                {
                    "utilization_gpu": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                }
            )
        except (ValueError, IndexError):
            continue

    payload = {"gpus": gpu_rows} if gpu_rows else None
    _gpu_cache = (now, payload)
    return payload


def _titan_snapshot() -> dict[str, Any] | None:
    """Return hardware snapshot from ai-factory-titan binary."""
    global _gpu_cache, _TITAN_SMI_AVAILABLE

    now = time.monotonic()

    # Reuse cache if still fresh.
    if _gpu_cache is not None:
        cached_ts, cached_payload = _gpu_cache
        if now - cached_ts < _GPU_CACHE_TTL_S:
            return cached_payload

    if _TITAN_SMI_AVAILABLE is None:
        # Check if the binary is built and available.
        # We look for it in the standard Cargo output directories.
        titan_bin = next(
            (
                candidate
                for candidate in (
                    Path("ai_factory_titan/target/debug/titan-status"),
                    Path("ai_factory_titan/target/release/titan-status"),
                    Path("ai_factory_titan/target/debug/titan-status.exe"),
                    Path("ai_factory_titan/target/release/titan-status.exe"),
                )
                if candidate.exists()
            ),
            None,
        )
        _TITAN_SMI_AVAILABLE = titan_bin is not None

    if not _TITAN_SMI_AVAILABLE:
        return None

    titan_command = next(
        (
            [str(candidate)]
            for candidate in (
                Path("ai_factory_titan/target/debug/titan-status"),
                Path("ai_factory_titan/target/release/titan-status"),
                Path("ai_factory_titan/target/debug/titan-status.exe"),
                Path("ai_factory_titan/target/release/titan-status.exe"),
            )
            if candidate.exists()
        ),
        None,
    )
    if titan_command is None:
        return None
    try:
        result = subprocess.run(  # nosec B603
            titan_command,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            import json

            payload = cast(dict[str, Any], json.loads(result.stdout))
            _gpu_cache = (now, payload)
            return payload
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("titan status probe failed: %s", exc)

    return None


def collect_metrics_for_instance(
    manifest: InstanceManifest,
    snapshot: dict[str, Any],
    *,
    collect_gpu: bool = True,
) -> tuple[dict[str, Any], list[MetricPoint], dict[str, Any]]:
    if manifest.type == "prepare":
        summary, points, refs = _prepare_metrics(snapshot)
    elif manifest.type in {"finetune", "train"}:
        summary, points, refs = _training_metrics(manifest, snapshot)
    elif manifest.type == "evaluate":
        summary, points, refs = _evaluation_metrics(snapshot)
    elif manifest.type == "inference":
        summary, points, refs = _inference_metrics(snapshot)
    elif manifest.type == "report":
        summary, points, refs = _report_metrics(snapshot)
    else:
        summary, points, refs = _deploy_metrics(manifest)
    if collect_gpu:
        gpu = _gpu_snapshot()
        if not gpu:
            gpu = _titan_snapshot()
        if gpu:
            summary["gpu"] = gpu
    return summary, points, refs


def collect_progress_for_instance(
    manifest: InstanceManifest,
    snapshot: dict[str, Any],
) -> ProgressSnapshot | None:
    if manifest.type == "prepare":
        return _prepare_progress(snapshot)
    if manifest.type in {"finetune", "train"}:
        return _training_progress(manifest, snapshot)
    if manifest.type == "evaluate":
        return _evaluation_progress(snapshot)
    if manifest.type == "inference":
        return _inference_progress(snapshot)
    if manifest.type == "report":
        return _report_progress(snapshot)
    if manifest.type == "deploy":
        return _deploy_progress(manifest)
    return None
