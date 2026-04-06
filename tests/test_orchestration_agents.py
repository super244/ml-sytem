from __future__ import annotations

from ai_factory.core.orchestration.agents import AgentRegistry, TrainingOrchestrationAgent


def test_agent_descriptor_exposes_primary_resource_and_support_checks() -> None:
    descriptor = TrainingOrchestrationAgent.descriptor

    assert descriptor.primary_resource_class == "gpu"
    assert descriptor.supports_task_type("train") is True
    assert descriptor.supports_task_type("deploy") is False
    assert descriptor.supports_resource_class("gpu") is True
    assert descriptor.supports_resource_class("network") is False


def test_agent_descriptor_capability_summary_is_operator_friendly() -> None:
    descriptor = TrainingOrchestrationAgent.descriptor
    summary = descriptor.capability_summary()

    assert summary["agent_type"] == "training_orchestration"
    assert summary["label"] == "Training orchestration"
    assert summary["primary_resource_class"] == "gpu"
    assert summary["max_concurrency"] == 1
    assert summary["retry_policy"]["max_attempts"] == 2


def test_registry_indexes_task_types_and_describes_candidates() -> None:
    registry = AgentRegistry()

    report_candidates = registry.agents_for_task_type("report")
    matrix = registry.task_type_matrix()
    deploy_description = registry.describe_task_type("deploy")

    assert [descriptor.agent_type for descriptor in report_candidates] == [
        "evaluation_benchmarking",
        "monitoring_telemetry",
        "optimization_feedback",
    ]
    assert matrix["deploy"] == ["deployment"]
    assert deploy_description["supported"] is True
    assert deploy_description["selected_agent_type"] == "deployment"
    assert deploy_description["primary_resource_class"] == "network"


def test_registry_capability_summaries_preserve_registry_order() -> None:
    registry = AgentRegistry()
    summaries = registry.capability_summaries()

    assert [item["agent_type"] for item in summaries] == [
        "data_processing",
        "training_orchestration",
        "evaluation_benchmarking",
        "monitoring_telemetry",
        "optimization_feedback",
        "deployment",
    ]
