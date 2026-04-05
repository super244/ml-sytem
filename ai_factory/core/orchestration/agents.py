from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_factory.core.orchestration.models import (
    AgentCapability,
    AgentType,
    ResourceClass,
    RetryPolicy,
    TaskType,
)

SUPPORTED_TASK_TYPES: tuple[TaskType, ...] = (
    "prepare",
    "train",
    "finetune",
    "evaluate",
    "report",
    "inference",
    "deploy",
    "monitor",
    "optimize",
)


@dataclass(frozen=True)
class AgentDescriptor:
    agent_type: AgentType
    label: str
    supported_task_types: tuple[TaskType, ...]
    max_concurrency: int = 1
    resource_classes: tuple[ResourceClass, ...] = ("control",)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    @property
    def primary_resource_class(self) -> ResourceClass:
        return self.resource_classes[0]

    def supports_task_type(self, task_type: TaskType) -> bool:
        return task_type in self.supported_task_types

    def supports_resource_class(self, resource_class: ResourceClass) -> bool:
        return resource_class in self.resource_classes

    def capability(self) -> AgentCapability:
        return AgentCapability(
            agent_type=self.agent_type,
            task_types=list(self.supported_task_types),
            resource_classes=list(self.resource_classes),
            max_concurrency=self.max_concurrency,
            retry_policy=self.retry_policy,
        )

    def capability_summary(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "label": self.label,
            "task_types": list(self.supported_task_types),
            "resource_classes": list(self.resource_classes),
            "primary_resource_class": self.primary_resource_class,
            "max_concurrency": self.max_concurrency,
            "retry_policy": self.retry_policy.model_dump(mode="json"),
        }


class BaseAsyncAgent:
    descriptor: AgentDescriptor

    async def execute(self, task_input: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, "input": task_input}


class DataProcessingAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="data_processing",
        label="Data processing",
        supported_task_types=("prepare",),
        max_concurrency=2,
        resource_classes=("cpu", "io"),
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=3, max_delay_s=30),
    )


class TrainingOrchestrationAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="training_orchestration",
        label="Training orchestration",
        supported_task_types=("train", "finetune", "inference"),
        max_concurrency=1,
        resource_classes=("gpu", "cpu"),
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=10, max_delay_s=120),
    )


class EvaluationBenchmarkingAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="evaluation_benchmarking",
        label="Evaluation and benchmarking",
        supported_task_types=("evaluate", "report"),
        max_concurrency=2,
        resource_classes=("cpu", "gpu"),
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=5, max_delay_s=60),
    )


class MonitoringTelemetryAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="monitoring_telemetry",
        label="Monitoring and telemetry",
        supported_task_types=("monitor", "report", "inference"),
        max_concurrency=4,
        resource_classes=("control", "io"),
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=2, max_delay_s=30),
    )


class OptimizationFeedbackAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="optimization_feedback",
        label="Optimization and feedback loops",
        supported_task_types=("optimize", "evaluate", "report"),
        max_concurrency=1,
        resource_classes=("control", "cpu"),
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=5, max_delay_s=45),
    )


class DeploymentAgent(BaseAsyncAgent):
    descriptor = AgentDescriptor(
        agent_type="deployment",
        label="Deployment",
        supported_task_types=("deploy",),
        max_concurrency=2,
        resource_classes=("network", "io"),
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=10, max_delay_s=60),
    )


class AgentRegistry:
    def __init__(self) -> None:
        descriptors = (
            DataProcessingAgent.descriptor,
            TrainingOrchestrationAgent.descriptor,
            EvaluationBenchmarkingAgent.descriptor,
            MonitoringTelemetryAgent.descriptor,
            OptimizationFeedbackAgent.descriptor,
            DeploymentAgent.descriptor,
        )
        self._agents: dict[AgentType, AgentDescriptor] = {
            descriptor.agent_type: descriptor
            for descriptor in descriptors
        }
        self._ordered_agents: tuple[AgentDescriptor, ...] = descriptors
        self._task_type_index: dict[TaskType, tuple[AgentDescriptor, ...]] = {}
        for task_type in SUPPORTED_TASK_TYPES:
            matching = tuple(descriptor for descriptor in self._ordered_agents if descriptor.supports_task_type(task_type))
            if matching:
                self._task_type_index[task_type] = matching

    def list_capabilities(self) -> list[AgentCapability]:
        return [descriptor.capability() for descriptor in self._agents.values()]

    def list_descriptors(self) -> tuple[AgentDescriptor, ...]:
        return self._ordered_agents

    def get(self, agent_type: AgentType) -> AgentDescriptor:
        return self._agents[agent_type]

    def agents_for_task_type(self, task_type: TaskType) -> tuple[AgentDescriptor, ...]:
        return self._task_type_index.get(task_type, ())

    def agent_for_task_type(self, task_type: TaskType) -> AgentDescriptor:
        candidates = self.agents_for_task_type(task_type)
        if candidates:
            return candidates[0]
        return self._agents["monitoring_telemetry"]

    def task_type_matrix(self) -> dict[TaskType, list[AgentType]]:
        return {
            task_type: [descriptor.agent_type for descriptor in descriptors]
            for task_type, descriptors in self._task_type_index.items()
        }

    def capability_summaries(self) -> list[dict[str, Any]]:
        return [descriptor.capability_summary() for descriptor in self._ordered_agents]

    def describe_task_type(self, task_type: TaskType) -> dict[str, Any]:
        candidates = self.agents_for_task_type(task_type)
        fallback = self._agents["monitoring_telemetry"]
        return {
            "task_type": task_type,
            "supported": bool(candidates),
            "agent_types": [descriptor.agent_type for descriptor in candidates],
            "labels": [descriptor.label for descriptor in candidates],
            "selected_agent_type": (candidates[0] if candidates else fallback).agent_type,
            "primary_resource_class": (candidates[0] if candidates else fallback).primary_resource_class,
            "max_concurrency": (candidates[0] if candidates else fallback).max_concurrency,
        }
