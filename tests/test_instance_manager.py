from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


MANAGER_MODULES = [
    "ai_factory.core.instances",
    "ai_factory.core.instances.models",
    "ai_factory.core.instances.store",
    "ai_factory.core.instances.queries",
    "ai_factory.core.instances.manager",
    "ai_factory.core.config",
    "ai_factory.core.config.schema",
    "ai_factory.core.config.loader",
    "ai_factory.core.execution",
    "ai_factory.core.execution.base",
    "ai_factory.core.execution.commands",
    "ai_factory.core.execution.local",
    "ai_factory.core.execution.ssh",
    "ai_factory.core.monitoring",
    "ai_factory.core.monitoring.events",
    "ai_factory.core.monitoring.metrics",
    "ai_factory.core.monitoring.collectors",
    "ai_factory.core.decisions",
    "ai_factory.core.decisions.rules",
]


@pytest.fixture(autouse=True)
def _cleanup_manager_modules():
    for name in MANAGER_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in MANAGER_MODULES:
        sys.modules.pop(name, None)


def _prime_manager_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name, subdir in (
        ("ai_factory.core.instances", "instances"),
        ("ai_factory.core.config", "config"),
        ("ai_factory.core.execution", "execution"),
        ("ai_factory.core.monitoring", "monitoring"),
        ("ai_factory.core.decisions", "decisions"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(repo_root / "ai_factory" / "core" / subdir)]
        monkeypatch.setitem(sys.modules, name, module)


def test_manager_creates_runs_and_finalizes_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "baseline_qlora.yaml").write_text("run_name: demo-run\ntraining:\n  artifacts_dir: artifacts\n")
    eval_config_path = tmp_path / "configs" / "eval.yaml"
    eval_config_path.parent.mkdir(parents=True, exist_ok=True)
    eval_config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: evaluate",
                "  environment: local",
                "subsystem:",
                "  config_ref: ../evaluation/configs/base_vs_finetuned.yaml",
            ]
        )
    )
    (tmp_path / "evaluation" / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "evaluation" / "configs" / "base_vs_finetuned.yaml").write_text("output_dir: evaluation/results/demo\n")
    config_path = tmp_path / "configs" / "finetune.yaml"
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: finetune",
                "  environment: local",
                "  name: demo-finetune",
                "sub_agents:",
                "  enabled: true",
                "  workloads:",
                "    - evaluation",
                "feedback_loop:",
                "  enabled: true",
                "  queue_follow_up_evaluation: true",
                "pipeline:",
                f"  default_eval_config: {eval_config_path}",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)

    fake_handle = models.ExecutionHandle(backend="local", pid=99, stdout_path="stdout", stderr_path="stderr")
    monkeypatch.setattr(
        manager_mod.LocalExecutor,
        "start",
        lambda self, manifest, command, *, artifacts_dir, stdout_path, stderr_path: fake_handle,
    )
    monkeypatch.setattr(
        manager_mod,
        "collect_metrics_for_instance",
        lambda manifest, snapshot, collect_gpu=True: (
            {"accuracy": 0.93, "parse_rate": 0.91},
            [models.MetricPoint(name="accuracy", value=0.93)],
            {"run_dir": "artifacts/runs/demo"},
        ),
    )

    created = manager.create_instance(str(config_path), start=True)
    finalized = manager.finalize_instance(created.id, 0, runtime_metadata={"note": "done"})

    loaded = store.load(created.id)

    assert created.status == "pending"
    assert loaded.status == "completed"
    assert finalized.execution is not None
    assert finalized.execution.exit_code == 0
    assert finalized.metrics_summary["accuracy"] == 0.93
    assert finalized.recommendations[0].action == "evaluate"
    assert store.read_current_metrics(created.id)["parse_rate"] == 0.91
    assert store.read_metric_points(created.id)[0]["name"] == "accuracy"
    assert manager.list_instances(parent_instance_id=created.id)[0].type == "evaluate"


def test_manager_can_create_evaluation_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    eval_dir = tmp_path / "evaluation" / "configs"
    eval_dir.mkdir(parents=True)
    model_registry = tmp_path / "inference" / "configs" / "model_registry.yaml"
    model_registry.parent.mkdir(parents=True, exist_ok=True)
    model_registry.write_text("models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n")
    (eval_dir / "base_vs_finetuned.yaml").write_text(
        "\n".join(
            [
                "name: demo-eval",
                "output_dir: evaluation/results/demo",
                "instance:",
                "  type: evaluate",
                "  environment: local",
                "benchmark:",
                "  registry_path: evaluation/benchmarks/registry.yaml",
                "subsystem:",
                f"  config_ref: {eval_dir / 'base_vs_finetuned.yaml'}",
                "models:",
                f"  registry_path: {model_registry}",
                "  primary_model: base",
                "  secondary_model: base",
            ]
        )
    )

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)
    source = models.InstanceManifest(
        id="finetune-001",
        type="finetune",
        name="source-finetune",
        environment=models.EnvironmentSpec(kind="local"),
        artifact_refs={"published": {"final_adapter": "artifacts/models/demo"}},
    )
    store.create(source, {"instance": {"type": "finetune"}})

    child = manager.create_evaluation_instance(source.id, config_path=str(eval_dir / "base_vs_finetuned.yaml"), start=False)

    assert child.parent_instance_id == source.id
    assert child.type == "evaluate"
    assert Path(child.config_snapshot_path).exists()


def test_manager_finalize_evaluation_schedules_reports_and_publish_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)

    eval_manifest = models.InstanceManifest(
        id="evaluate-001",
        type="evaluate",
        name="eval",
        environment=models.EnvironmentSpec(kind="local"),
        artifact_refs={"source_artifact": "artifacts/models/demo"},
    )
    store.create(
        eval_manifest,
        {
            "instance": {"type": "evaluate", "environment": {"kind": "local"}},
            "subsystem": {"config_ref": "evaluation/configs/base_vs_finetuned.yaml"},
            "feedback_loop": {"enabled": True, "suggest_failure_analysis": True},
            "publish_hooks": [{"target": "ollama", "enabled": True, "when": "after_evaluation"}],
            "pipeline": {"default_report_config": "configs/report.yaml", "default_deploy_config": "configs/deploy.yaml"},
        },
    )

    monkeypatch.setattr(
        manager_mod,
        "collect_metrics_for_instance",
        lambda manifest, snapshot, collect_gpu=True: (
            {
                "accuracy": 0.91,
                "parse_rate": 0.86,
                "verifier_agreement_rate": 0.73,
                "no_answer_rate": 0.04,
                "avg_latency_s": 1.2,
            },
            [models.MetricPoint(name="accuracy", value=0.91)],
            {
                "evaluation_dir": str(tmp_path / "evaluation" / "results" / "demo"),
                "per_example": str(tmp_path / "evaluation" / "results" / "demo" / "per_example.jsonl"),
            },
        ),
    )

    scheduled: dict[str, list[dict[str, str]]] = {"instances": [], "deployments": []}

    monkeypatch.setattr(
        manager,
        "create_instance",
        lambda config_path, **kwargs: scheduled["instances"].append({"config_path": config_path}) or eval_manifest,
    )
    monkeypatch.setattr(
        manager,
        "create_deployment_instance",
        lambda source_instance_id, **kwargs: scheduled["deployments"].append(
            {"source_instance_id": source_instance_id, "target": kwargs["target"], "config_path": kwargs["config_path"]}
        )
        or eval_manifest,
    )
    monkeypatch.setattr(manager, "start_instance", lambda instance_id: eval_manifest)

    finalized = manager.finalize_instance(eval_manifest.id, 0)

    assert finalized.decision is not None
    assert finalized.decision.action == "deploy"
    assert any(item["config_path"] == "configs/report.yaml" for item in scheduled["instances"])
    assert scheduled["deployments"][0]["target"] == "ollama"
