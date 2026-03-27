from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


MONITORING_MODULES = [
    "ai_factory.core.instances",
    "ai_factory.core.instances.models",
    "ai_factory.core.monitoring",
    "ai_factory.core.monitoring.events",
    "ai_factory.core.monitoring.metrics",
    "ai_factory.core.monitoring.collectors",
]


@pytest.fixture(autouse=True)
def _cleanup_monitoring_modules():
    for name in MONITORING_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in MONITORING_MODULES:
        sys.modules.pop(name, None)


def _prime_monitoring_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name, subdir in (
        ("ai_factory.core.instances", "instances"),
        ("ai_factory.core.monitoring", "monitoring"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(repo_root / "ai_factory" / "core" / subdir)]
        monkeypatch.setitem(sys.modules, name, module)


def test_collect_metrics_for_training_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_monitoring_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    collectors = importlib.import_module("ai_factory.core.monitoring.collectors")

    run_dir = tmp_path / "runs" / "finetune-20260101-000000"
    (run_dir / "manifests").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "manifests" / "run_manifest.json").write_text(
        "{\"run_id\": \"finetune-20260101-000000\", \"run_name\": \"finetune\", \"metadata\": {\"published\": {\"final_adapter\": \"artifacts/models/demo\"}}}"
    )
    (run_dir / "metrics" / "metrics.json").write_text("{\"eval_loss\": 0.42, \"train_loss\": 0.9}")
    (run_dir / "metrics" / "dataset_report.json").write_text("{\"tokenized_train_rows\": 12, \"tokenized_eval_rows\": 4}")
    (run_dir / "metrics" / "model_report.json").write_text("{\"trainable_ratio\": 0.123}")
    (run_dir / "logs" / "training_metrics.jsonl").write_text(
        "{\"step\": 1, \"loss\": 1.5, \"lr\": 0.001}\n{\"step\": 2, \"loss\": 1.0, \"lr\": 0.0005}\n"
    )

    manifest = models.InstanceManifest(
        id="finetune-001",
        type="finetune",
        name="finetune",
        environment=models.EnvironmentSpec(kind="local"),
    )
    snapshot = {
        "resolved_subsystem_config": {"run_name": "finetune", "training": {"artifacts_dir": str(tmp_path)}}
    }

    summary, points, refs = collectors.collect_metrics_for_instance(manifest, snapshot, collect_gpu=False)

    assert summary["eval_loss"] == 0.42
    assert summary["train_rows"] == 12
    assert summary["trainable_ratio"] == 0.123
    assert len(points) == 4
    assert refs["published"]["final_adapter"] == "artifacts/models/demo"


def test_collect_metrics_for_evaluation_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_monitoring_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    collectors = importlib.import_module("ai_factory.core.monitoring.collectors")

    output_dir = tmp_path / "evaluation" / "results" / "base_vs_finetuned"
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        "{\"primary\": {\"accuracy\": 0.88, \"parse_rate\": 0.75, \"verifier_agreement_rate\": 0.7, \"no_answer_rate\": 0.05, \"avg_latency_s\": 1.2}}"
    )

    manifest = models.InstanceManifest(
        id="evaluate-001",
        type="evaluate",
        name="evaluate",
        environment=models.EnvironmentSpec(kind="local"),
    )
    snapshot = {"resolved_subsystem_config": {"output_dir": str(output_dir)}}

    summary, points, refs = collectors.collect_metrics_for_instance(manifest, snapshot, collect_gpu=False)

    assert summary["accuracy"] == 0.88
    assert summary["parse_rate"] == 0.75
    assert any(point.name == "accuracy" for point in points)
    assert refs["summary_json"].endswith("summary.json")
