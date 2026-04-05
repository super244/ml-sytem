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
    "ai_factory.core.plugins",
    "ai_factory.core.plugins.base",
    "ai_factory.core.plugins.builtins",
    "ai_factory.core.plugins.registry",
    "ai_factory.core.monitoring",
    "ai_factory.core.monitoring.events",
    "ai_factory.core.monitoring.metrics",
    "ai_factory.core.monitoring.collectors",
    "ai_factory.core.decisions",
    "ai_factory.core.decisions.rules",
    "ai_factory.core.state",
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
        ("ai_factory.core.plugins", "plugins"),
        ("ai_factory.core.monitoring", "monitoring"),
        ("ai_factory.core.decisions", "decisions"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(repo_root / "ai_factory" / "core" / subdir)]
        monkeypatch.setitem(sys.modules, name, module)


def test_manager_creates_runs_and_finalizes_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_manager_defers_primary_start_until_preprocess_dependency_clears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "baseline_qlora.yaml").write_text("run_name: demo-run\ntraining:\n  artifacts_dir: artifacts\n")
    prepare_config = tmp_path / "configs" / "prepare.yaml"
    prepare_config.parent.mkdir(parents=True, exist_ok=True)
    prepare_config.write_text(
        "\n".join(
            [
                "instance:",
                "  type: prepare",
                "  environment: local",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )
    train_config = tmp_path / "configs" / "train.yaml"
    train_config.write_text(
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  environment: local",
                "sub_agents:",
                "  enabled: true",
                "  workloads:",
                "    - preprocess",
                "pipeline:",
                f"  default_prepare_config: {prepare_config}",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)
    started: list[tuple[str, str]] = []
    fake_handle = models.ExecutionHandle(backend="local", pid=101, stdout_path="stdout", stderr_path="stderr")
    monkeypatch.setattr(
        manager_mod,
        "build_command",
        lambda config, manifest, plugin_registry=None: types.SimpleNamespace(argv=["python", manifest.type]),
    )
    monkeypatch.setattr(
        manager_mod.LocalExecutor,
        "start",
        lambda self, manifest, command, *, artifacts_dir, stdout_path, stderr_path: (
            started.append((manifest.id, manifest.type)) or fake_handle
        ),
    )

    created = manager.create_instance(str(train_config), start=True)
    children = manager.list_instances(parent_instance_id=created.id)

    assert created.execution is None
    assert created.progress is not None
    assert created.progress.stage in {"blocked", "queued"}
    assert len(children) == 1
    assert children[0].type == "prepare"
    assert started == [(children[0].id, "prepare")]


def test_manager_can_create_evaluation_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    child = manager.create_evaluation_instance(
        source.id, config_path=str(eval_dir / "base_vs_finetuned.yaml"), start=False
    )

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
            "pipeline": {
                "default_report_config": "configs/report.yaml",
                "default_deploy_config": "configs/deploy.yaml",
            },
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

    def mock_create_instance(config_path, **kwargs):
        scheduled["instances"].append({"config_path": config_path})
        return eval_manifest

    def mock_create_deployment_instance(source_instance_id, **kwargs):
        scheduled["deployments"].append(
            {
                "source_instance_id": source_instance_id,
                "target": kwargs["target"],
                "config_path": kwargs["config_path"],
            }
        )
        return eval_manifest

    monkeypatch.setattr(manager, "create_instance", mock_create_instance)
    monkeypatch.setattr(manager, "create_deployment_instance", mock_create_deployment_instance)
    monkeypatch.setattr(manager, "start_instance", lambda instance_id: eval_manifest)

    finalized = manager.finalize_instance(eval_manifest.id, 0)

    assert finalized.decision is not None
    assert finalized.decision.action == "deploy"
    assert any(item["config_path"] == "configs/report.yaml" for item in scheduled["instances"])
    assert scheduled["deployments"][0]["target"] == "ollama"


def test_manager_applies_lifecycle_overrides_to_created_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "baseline_qlora.yaml").write_text("run_name: demo-run\ntraining:\n  artifacts_dir: artifacts\n")
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  environment: local",
                "experience:",
                "  level: hobbyist",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)

    manifest = manager.create_instance(
        str(config_path),
        start=False,
        name_override="scratch-branch",
        user_level_override="dev",
        lifecycle_override=models.LifecycleProfile(
            origin="from_scratch",
            learning_mode="supervised",
            architecture=models.ArchitectureSpec(
                family="transformer",
                hidden_size=1024,
                num_layers=24,
                num_attention_heads=16,
            ),
            deployment_targets=["ollama"],
        ),
        subsystem_updates={"labels": ["control_center"]},
        metadata_updates={"source": "test"},
    )

    loaded = store.load(manifest.id)
    snapshot = store.load_config_snapshot(manifest.id)

    assert loaded.name == "scratch-branch"
    assert loaded.user_level == "dev"
    assert loaded.lifecycle.origin == "from_scratch"
    assert loaded.lifecycle.architecture.family == "transformer"
    assert loaded.lifecycle.deployment_targets == ["ollama"]
    assert snapshot["lifecycle"]["origin"] == "from_scratch"
    assert snapshot["experience"]["level"] == "dev"
    assert snapshot["subsystem"]["labels"] == ["control_center"]


def test_manager_projects_live_training_state_from_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prime_manager_packages(monkeypatch)
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    artifacts_dir = tmp_path / "artifacts"
    (profile_dir / "baseline_qlora.yaml").write_text(
        f"run_name: live-run\ntraining:\n  artifacts_dir: {artifacts_dir}\n  max_steps: 10\n"
    )
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  environment: local",
                "sub_agents:",
                "  enabled: false",
                "feedback_loop:",
                "  queue_follow_up_evaluation: false",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    run_dir = artifacts_dir / "runs" / "live-run"
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests" / "run_manifest.json").write_text(
        '{"run_id":"train-20260328-000000","run_name":"live-run","base_model":"Qwen/Qwen2.5-Math-1.5B-Instruct"}'
    )
    (run_dir / "logs" / "training_metrics.jsonl").write_text(
        '{"step": 2, "loss": 1.1, "lr": 0.0005}\n{"step": 3, "loss": 0.9, "lr": 0.0004}\n'
    )
    (run_dir / "metrics" / "metrics.json").write_text('{"train_loss": 0.9, "eval_loss": 0.6}')
    (run_dir / "metrics" / "dataset_report.json").write_text('{"tokenized_train_rows": 24, "tokenized_eval_rows": 6}')
    (run_dir / "metrics" / "model_report.json").write_text('{"trainable_ratio": 0.12}')

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)
    manifest = manager.create_instance(str(config_path), start=False)
    manifest.status = "running"
    store.save(manifest)

    projected = manager.get_instance(manifest.id)
    metrics = manager.get_metrics(manifest.id)

    assert projected.progress is not None
    assert projected.progress.stage == "training"
    assert projected.progress.completed_steps == 3
    assert projected.metrics_summary["latest_step"] == 3
    assert metrics["summary"]["train_loss"] == 0.9
    assert metrics["points"][0]["name"] == "loss"


def test_manager_exposes_recommendation_backed_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)
    manifest = models.InstanceManifest(
        id="evaluate-001",
        type="evaluate",
        name="eval",
        status="completed",
        environment=models.EnvironmentSpec(kind="local"),
        artifact_refs={"source_artifact": "artifacts/models/demo"},
        recommendations=[
            models.FeedbackRecommendation(
                action="finetune",
                reason="Try another adapter pass.",
                target_instance_type="finetune",
                config_path="configs/finetune.yaml",
            ),
            models.FeedbackRecommendation(
                action="report",
                reason="Inspect failure slices before the next iteration.",
                target_instance_type="report",
                config_path="configs/report.yaml",
            ),
        ],
    )
    store.create(
        manifest,
        {
            "instance": {"type": "evaluate", "environment": {"kind": "local"}},
            "subsystem": {"config_ref": "evaluation/configs/base_vs_finetuned.yaml"},
            "pipeline": {
                "default_eval_config": "configs/eval.yaml",
                "default_inference_config": "configs/inference.yaml",
                "default_deploy_config": "configs/deploy.yaml",
            },
        },
    )

    actions = manager.get_available_actions(manifest.id)
    action_ids = {(item["action"], item.get("target_instance_type")) for item in actions}

    assert ("open_inference", "inference") in action_ids
    assert ("finetune", "finetune") in action_ids
    assert ("report", "report") in action_ids


def test_manager_can_create_inference_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    model_registry = tmp_path / "inference" / "configs" / "model_registry.yaml"
    model_registry.parent.mkdir(parents=True, exist_ok=True)
    model_registry.write_text("models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n")
    config_path = tmp_path / "configs" / "inference.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: inference",
                "  environment: local",
                "execution:",
                "  env:",
                f"    MODEL_REGISTRY_PATH: {model_registry}",
                "subsystem:",
                "  model_variant: finetuned",
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
        lifecycle=models.LifecycleProfile(origin="existing_model", learning_mode="qlora"),
        artifact_refs={"published": {"final_adapter": "artifacts/models/demo"}},
    )
    store.create(source, {"instance": {"type": "finetune"}})

    child = manager.create_inference_instance(source.id, config_path=str(config_path), start=False)
    snapshot = store.load_config_snapshot(child.id)

    assert child.parent_instance_id == source.id
    assert child.type == "inference"
    assert child.lifecycle.stage == "infer"
    assert snapshot["subsystem"]["model_variant"].startswith("instance_")
    assert "generated_model_registry.yaml" in snapshot["execution"]["env"]["MODEL_REGISTRY_PATH"]


def test_manager_finalize_evaluation_can_recommend_inference(
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
        id="evaluate-002",
        type="evaluate",
        name="eval-latency-missing",
        environment=models.EnvironmentSpec(kind="local"),
        lifecycle=models.LifecycleProfile(stage="evaluate", deployment_targets=["ollama"]),
        artifact_refs={"source_artifact": "artifacts/models/demo"},
    )
    store.create(
        eval_manifest,
        {
            "instance": {"type": "evaluate", "environment": {"kind": "local"}},
            "subsystem": {"config_ref": "evaluation/configs/base_vs_finetuned.yaml"},
            "lifecycle": {"stage": "evaluate", "deployment_targets": ["ollama"]},
            "feedback_loop": {"enabled": True, "suggest_failure_analysis": True},
        },
    )

    monkeypatch.setattr(
        manager_mod,
        "collect_metrics_for_instance",
        lambda manifest, snapshot, collect_gpu=True: (
            {
                "accuracy": 0.9,
                "parse_rate": 0.85,
                "verifier_agreement_rate": 0.72,
                "no_answer_rate": 0.03,
            },
            [models.MetricPoint(name="accuracy", value=0.9)],
            {"evaluation_dir": str(tmp_path / "evaluation" / "results" / "demo")},
        ),
    )

    finalized = manager.finalize_instance(eval_manifest.id, 0)
    actions = manager.get_available_actions(eval_manifest.id)

    assert finalized.decision is not None
    assert finalized.decision.action == "open_inference"
    assert any(item.action == "open_inference" for item in finalized.recommendations)
    assert any(item["action"] == "open_inference" for item in actions)


def test_manager_cancel_is_noop_for_completed_instances_and_retry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prime_manager_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    manager_mod = importlib.import_module("ai_factory.core.instances.manager")

    profile_dir = tmp_path / "training" / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "baseline_qlora.yaml").write_text("run_name: demo-run\ntraining:\n  artifacts_dir: artifacts\n")
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  environment: local",
                "sub_agents:",
                "  enabled: false",
                "feedback_loop:",
                "  queue_follow_up_evaluation: false",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
            ]
        )
    )

    store = store_mod.FileInstanceStore(tmp_path)
    manager = manager_mod.InstanceManager(store)
    fake_handle = models.ExecutionHandle(backend="local", pid=102, stdout_path="stdout", stderr_path="stderr")
    monkeypatch.setattr(
        manager_mod,
        "build_command",
        lambda config, manifest, plugin_registry=None: types.SimpleNamespace(argv=["python", manifest.type]),
    )
    monkeypatch.setattr(
        manager_mod.LocalExecutor,
        "start",
        lambda self, manifest, command, *, artifacts_dir, stdout_path, stderr_path: fake_handle,
    )
    monkeypatch.setattr(
        manager_mod,
        "collect_metrics_for_instance",
        lambda manifest, snapshot, collect_gpu=True: (
            {"accuracy": 0.93},
            [models.MetricPoint(name="accuracy", value=0.93)],
            {"run_dir": "artifacts/runs/demo"},
        ),
    )

    created = manager.create_instance(str(config_path), start=True)
    completed = manager.finalize_instance(created.id, 0)
    cancelled = manager.cancel_instance(created.id)

    assert completed.status == "completed"
    assert cancelled.status == "completed"
    assert cancelled.progress is not None
    assert cancelled.progress.stage == "completed"

    with pytest.raises(ValueError, match="not retryable"):
        manager.retry_instance(created.id)
