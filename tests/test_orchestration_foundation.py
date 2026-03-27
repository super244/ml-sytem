from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


ORCHESTRATION_MODULES = [
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
def _cleanup_orchestration_modules():
    for name in ORCHESTRATION_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in ORCHESTRATION_MODULES:
        sys.modules.pop(name, None)


def _stub_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, module)


def _prime_orchestration_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _stub_package(monkeypatch, "ai_factory.core.instances", repo_root / "ai_factory" / "core" / "instances")
    _stub_package(monkeypatch, "ai_factory.core.config", repo_root / "ai_factory" / "core" / "config")
    _stub_package(monkeypatch, "ai_factory.core.execution", repo_root / "ai_factory" / "core" / "execution")
    _stub_package(monkeypatch, "ai_factory.core.monitoring", repo_root / "ai_factory" / "core" / "monitoring")
    _stub_package(monkeypatch, "ai_factory.core.decisions", repo_root / "ai_factory" / "core" / "decisions")


def test_orchestration_config_loads_and_resolves_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_orchestration_packages(monkeypatch)
    loader = importlib.import_module("ai_factory.core.config.loader")

    (tmp_path / "training" / "configs" / "profiles").mkdir(parents=True)
    (tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml").write_text(
        "run_name: baseline-run\ntraining:\n  artifacts_dir: artifacts\n"
    )
    config_path = tmp_path / "configs" / "finetune.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: finetune",
                "  environment: local",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
                "pipeline:",
                "  auto_continue: false",
            ]
        )
    )

    config = loader.load_orchestration_config(str(config_path))

    assert config.instance.type == "finetune"
    assert config.instance.environment.kind == "local"
    assert config.experience.level == "hobbyist"
    assert config.resolved_subsystem_config_path is not None
    assert config.resolved_subsystem_config["run_name"] == "baseline-run"


def test_file_instance_store_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_orchestration_packages(monkeypatch)
    models = importlib.import_module("ai_factory.core.instances.models")
    store_mod = importlib.import_module("ai_factory.core.instances.store")
    events_mod = importlib.import_module("ai_factory.core.monitoring.events")

    store = store_mod.FileInstanceStore(tmp_path)
    manifest = models.InstanceManifest(
        id="finetune-001",
        type="finetune",
        name="baseline",
        environment=models.EnvironmentSpec(kind="local"),
    )

    created = store.create(manifest, {"instance": {"type": "finetune"}})
    created.status = "running"
    store.save(created)
    store.append_event("finetune-001", events_mod.InstanceEvent(type="instance.running", message="running"))
    store.write_current_metrics("finetune-001", {"accuracy": 0.9})
    store.append_metric_points(
        "finetune-001",
        [
            models.MetricPoint(name="loss", value=1.5, tags={"stage": "finetune"}),
            models.MetricPoint(name="eval_loss", value=1.1, tags={"stage": "finetune"}),
        ],
    )
    store.stdout_path("finetune-001").write_text("hello\n")
    store.stderr_path("finetune-001").write_text("warn\n")
    store.write_decision_report("finetune-001", {"action": "deploy"})
    store.write_recommendations_report("finetune-001", [{"action": "evaluate"}])

    loaded = store.load("finetune-001")

    assert loaded.id == "finetune-001"
    assert loaded.config_snapshot_path is not None
    assert store.list_instances()[0].id == "finetune-001"
    assert store.read_events("finetune-001")[0]["type"] == "instance.created"
    assert store.read_current_metrics("finetune-001")["accuracy"] == 0.9
    assert len(store.read_metric_points("finetune-001")) == 2
    assert store.read_logs("finetune-001") == {"stdout": "hello\n", "stderr": "warn\n"}
    assert store.decision_path("finetune-001").exists()
    assert store.read_recommendations_report("finetune-001")[0]["action"] == "evaluate"


def test_decision_rules_cover_deploy_finetune_and_retrain(monkeypatch: pytest.MonkeyPatch):
    _prime_orchestration_packages(monkeypatch)
    schema = importlib.import_module("ai_factory.core.config.schema")
    rules = importlib.import_module("ai_factory.core.decisions.rules")

    policy = schema.DecisionPolicy(min_accuracy=0.8, min_parse_rate=0.7, min_verifier_agreement=0.6)

    deploy = rules.decide_next_step(
        {"accuracy": 0.9, "parse_rate": 0.8, "verifier_agreement_rate": 0.7, "no_answer_rate": 0.05},
        policy,
    )
    finetune = rules.decide_next_step(
        {"accuracy": 0.5, "parse_rate": 0.75, "verifier_agreement_rate": 0.2, "no_answer_rate": 0.2},
        policy,
    )
    retrain = rules.decide_next_step(
        {"accuracy": 0.1, "parse_rate": 0.1, "verifier_agreement_rate": 0.1, "no_answer_rate": 0.9},
        policy,
    )

    assert deploy.action == "deploy"
    assert finetune.action == "finetune"
    assert retrain.action == "retrain"


def test_command_builder_covers_instance_types(monkeypatch: pytest.MonkeyPatch):
    _prime_orchestration_packages(monkeypatch)
    schema = importlib.import_module("ai_factory.core.config.schema")
    models = importlib.import_module("ai_factory.core.instances.models")
    commands = importlib.import_module("ai_factory.core.execution.commands")

    config = schema.OrchestrationConfig.model_validate(
        {
            "instance": {"type": "finetune", "environment": {"kind": "local"}},
            "subsystem": {"config_ref": "training/configs/profiles/baseline_qlora.yaml"},
        }
    )
    manifest = models.InstanceManifest(
        id="finetune-001",
        type="finetune",
        name="baseline",
        environment=models.EnvironmentSpec(kind="local"),
    )

    finetune_cmd = commands.build_command(config, manifest)
    evaluate_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "evaluate", "environment": {"kind": "local"}},
                "subsystem": {"config_ref": "evaluation/configs/base_vs_finetuned.yaml"},
            }
        ),
        models.InstanceManifest(
            id="evaluate-001",
            type="evaluate",
            name="eval",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )
    deploy_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "deploy", "environment": {"kind": "local"}},
                "subsystem": {
                    "provider": "huggingface",
                    "provider_options": {"dry_run": True},
                    "source_artifact_ref": "artifacts/models/demo",
                },
            }
        ),
        models.InstanceManifest(
            id="deploy-001",
            type="deploy",
            name="deploy",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )
    train_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "train", "environment": {"kind": "local"}},
                "orchestration_mode": "hybrid",
                "experience": {"level": "dev"},
                "subsystem": {"config_ref": "training/configs/profiles/baseline_qlora.yaml"},
            }
        ),
        models.InstanceManifest(
            id="train-001",
            type="train",
            name="train",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )
    prepare_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "prepare", "environment": {"kind": "local"}},
                "subsystem": {"config_ref": "data/configs/processing.yaml"},
            }
        ),
        models.InstanceManifest(
            id="prepare-001",
            type="prepare",
            name="prepare",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )
    report_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "report", "environment": {"kind": "local"}},
                "subsystem": {
                    "command_override": [
                        "python3",
                        "-m",
                        "evaluation.analysis.analyze_failures",
                        "--input",
                        "evaluation/results/latest/per_example.jsonl",
                    ]
                },
            }
        ),
        models.InstanceManifest(
            id="report-001",
            type="report",
            name="report",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )
    custom_api_cmd = commands.build_command(
        schema.OrchestrationConfig.model_validate(
            {
                "instance": {"type": "deploy", "environment": {"kind": "local"}},
                "subsystem": {
                    "provider": "custom_api",
                    "provider_options": {"endpoint": "https://example.invalid/publish"},
                    "source_artifact_ref": "artifacts/models/demo",
                },
            }
        ),
        models.InstanceManifest(
            id="deploy-002",
            type="deploy",
            name="deploy-api",
            environment=models.EnvironmentSpec(kind="local"),
        ),
    )

    assert finetune_cmd.argv[-2:] == ["--config", "training/configs/profiles/baseline_qlora.yaml"]
    assert evaluate_cmd.argv[-2:] == ["--config", "evaluation/configs/base_vs_finetuned.yaml"]
    assert deploy_cmd.argv[1] == "-c"
    assert train_cmd.argv[-2:] == ["--config", "training/configs/profiles/baseline_qlora.yaml"]
    assert prepare_cmd.argv[-2:] == ["--config", "data/configs/processing.yaml"]
    assert report_cmd.argv[:3] == ["python3", "-m", "evaluation.analysis.analyze_failures"]
    assert custom_api_cmd.argv[1] == "-c"
    assert train_cmd.env["AI_FACTORY_INSTANCE_TYPE"] == "train"
    assert train_cmd.env["AI_FACTORY_ORCHESTRATION_MODE"] == "hybrid"


def test_experience_guardrails_strip_unsafe_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prime_orchestration_packages(monkeypatch)
    loader = importlib.import_module("ai_factory.core.config.loader")

    (tmp_path / "training" / "configs" / "profiles").mkdir(parents=True)
    (tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml").write_text("run_name: baseline\n")
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "instance:",
                "  type: train",
                "  environment: local",
                "experience:",
                "  level: beginner",
                "sub_agents:",
                "  enabled: true",
                "  max_parallelism: 4",
                "subsystem:",
                "  config_ref: ../training/configs/profiles/baseline_qlora.yaml",
                "  extra_args:",
                "    - --unsafe",
                "  command_override:",
                "    - python",
                "    - train.py",
            ]
        )
    )

    config = loader.apply_experience_guardrails(loader.load_orchestration_config(str(config_path)))

    assert config.subsystem.extra_args == []
    assert config.subsystem.command_override is None
    assert config.sub_agents.max_parallelism == 1
