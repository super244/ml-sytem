from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ai_factory.core.artifacts import ArtifactLayout
from ai_factory.core.io import write_json
from training.src.config import ExperimentConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flatten(prefix: str, payload: Any, output: dict[str, Any]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_prefix, value, output)
        return
    if isinstance(payload, list):
        output[prefix] = json.dumps(payload, ensure_ascii=False)
        return
    output[prefix] = payload


def flatten_payload(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten("", payload, flattened)
    return flattened


class Tracker(Protocol):
    def log_params(self, payload: dict[str, Any]) -> None: ...
    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None: ...
    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None: ...


class NullTracker:
    def log_params(self, payload: dict[str, Any]) -> None:
        return None

    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        return None

    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        return None


class JsonArtifactTracker:
    def __init__(self, layout: ArtifactLayout, config: ExperimentConfig):
        self.layout = layout
        self.config = config
        self.context_path = layout.manifests_dir / "tracking_context.json"
        self.events_path = layout.logs_dir / "tracking_events.jsonl"
        self.summary_path = layout.metrics_dir / "tracking_summary.json"
        self.context = {
            "created_at": _utc_now(),
            "run_id": layout.run_id,
            "run_name": config.tracking.run_name or config.run_name,
            "profile_name": config.profile_name,
            "provider": config.tracking.provider,
            "project": config.tracking.project,
            "experiment_name": config.tracking.experiment_name,
            "tags": list(config.tracking.tags),
            "metadata": dict(config.tracking.metadata),
        }
        write_json(self.context_path, self.context)

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": _utc_now(),
                        "event_type": event_type,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def log_params(self, payload: dict[str, Any]) -> None:
        self._append_event("params", payload)

    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        metrics = {key: value for key, value in payload.items() if isinstance(value, (int, float, bool))}
        if not metrics:
            return
        if step is not None:
            metrics["step"] = step
        self._append_event("metrics", metrics)

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        self._append_event("artifact", {"path": str(Path(path)), "name": name})

    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        payload = {"status": status, "summary": summary or {}}
        write_json(self.summary_path, payload)
        self._append_event("finalize", payload)


class MlflowTracker(NullTracker):
    def __init__(self, layout: ArtifactLayout, config: ExperimentConfig):
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - exercised only when optional dep missing
            raise RuntimeError("Tracking provider 'mlflow' requires the `mlflow` package.") from exc
        self._mlflow = mlflow
        if config.tracking.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.tracking.mlflow_tracking_uri)
        mlflow.set_experiment(config.tracking.experiment_name or config.tracking.project)
        self.run = mlflow.start_run(run_name=config.tracking.run_name or config.run_name)
        tags = {
            "profile_name": config.profile_name,
            "run_id": layout.run_id,
            **dict(config.tracking.metadata),
        }
        if config.tracking.tags:
            tags["tags"] = ",".join(config.tracking.tags)
        mlflow.set_tags(tags)

    def log_params(self, payload: dict[str, Any]) -> None:
        params = flatten_payload(payload)
        sanitized = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in params.items()
        }
        self._mlflow.log_params(sanitized)

    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        metrics = {key: float(value) for key, value in payload.items() if isinstance(value, (int, float))}
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=name)

    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        if summary:
            self.log_metrics(summary)
        self._mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")


class WandbTracker(NullTracker):
    def __init__(self, layout: ArtifactLayout, config: ExperimentConfig):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - exercised only when optional dep missing
            raise RuntimeError("Tracking provider 'wandb' requires the `wandb` package.") from exc
        self._wandb = wandb
        self.run = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.wandb_entity,
            name=config.tracking.run_name or config.run_name,
            group=config.tracking.experiment_name,
            tags=config.tracking.tags,
            config=config.model_dump(),
            mode=config.tracking.wandb_mode,
        )
        if self.run is not None:
            self.run.summary["run_id"] = layout.run_id
            self.run.summary["profile_name"] = config.profile_name
            for key, value in config.tracking.metadata.items():
                self.run.summary[key] = value

    def log_params(self, payload: dict[str, Any]) -> None:
        return None

    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        metrics = {key: value for key, value in payload.items() if isinstance(value, (int, float, bool))}
        if metrics:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        artifact = self._wandb.Artifact(name or Path(path).stem, type="artifact")
        artifact.add_file(str(path))
        if self.run is not None:
            self.run.log_artifact(artifact)

    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        if self.run is not None and summary:
            for key, value in summary.items():
                self.run.summary[key] = value
        self._wandb.finish(exit_code=0 if status == "completed" else 1)


class CompositeTracker:
    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    def log_params(self, payload: dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_params(payload)

    def log_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(payload, step=step)

    def log_artifact(self, path: str | Path, *, name: str | None = None) -> None:
        for tracker in self.trackers:
            tracker.log_artifact(path, name=name)

    def finalize(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        for tracker in self.trackers:
            tracker.finalize(status=status, summary=summary)


def build_tracker(layout: ArtifactLayout, config: ExperimentConfig) -> CompositeTracker:
    trackers: list[Tracker] = [JsonArtifactTracker(layout, config)]
    provider = (config.tracking.provider or "none").lower()
    if provider == "mlflow":
        trackers.append(MlflowTracker(layout, config))
    elif provider == "wandb":
        trackers.append(WandbTracker(layout, config))
    elif provider not in {"none", "jsonl"}:
        raise ValueError(f"Unsupported tracking provider: {config.tracking.provider}")
    return CompositeTracker(trackers)
