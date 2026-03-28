from __future__ import annotations

import json

from ai_factory.core.artifacts import prepare_run_layout
from training.src.config import load_experiment_config
from training.src.environment import collect_environment_snapshot
from training.src.tracking import build_tracker


def test_profile_config_loads_tracking_component():
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    assert config.tracking.provider == "none"
    assert config.tracking.capture_environment is True
    assert config.tracking.log_summary_artifact is True


def test_environment_snapshot_writes_runtime_metadata(tmp_path):
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.training.artifacts_dir = str(tmp_path)
    config.tracking.capture_installed_packages = False
    layout = prepare_run_layout(tmp_path, "unit-tracking")

    output_path, snapshot = collect_environment_snapshot(config, layout)

    assert output_path.exists()
    body = json.loads(output_path.read_text())
    assert body["run_id"] == layout.run_id
    assert body["seed"] == config.seed
    assert "python" in body
    assert snapshot["files"]["config"]["exists"] is True


def test_json_tracker_writes_context_events_and_summary(tmp_path):
    config = load_experiment_config("training/configs/profiles/baseline_qlora.yaml")
    config.training.artifacts_dir = str(tmp_path)
    layout = prepare_run_layout(tmp_path, "unit-tracker")
    tracker = build_tracker(layout, config)

    tracker.log_params({"run": {"name": config.run_name}})
    tracker.log_metrics({"loss": 0.25}, step=10)
    tracker.finalize(status="completed", summary={"eval_loss": 0.2})

    context_path = layout.manifests_dir / "tracking_context.json"
    events_path = layout.logs_dir / "tracking_events.jsonl"
    summary_path = layout.metrics_dir / "tracking_summary.json"

    assert context_path.exists()
    assert events_path.exists()
    assert summary_path.exists()
    assert any('"event_type": "metrics"' in line for line in events_path.read_text().splitlines())
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    assert summary["summary"]["eval_loss"] == 0.2
