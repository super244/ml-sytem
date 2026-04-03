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


@dataclass(frozen=True)
class AgentDescriptor:
    agent_type: AgentType
    label: str
    supported_task_types: tuple[TaskType, ...]
    max_concurrency: int = 1
    resource_classes: tuple[ResourceClass, ...] = ("control",)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def capability(self) -> AgentCapability:
        return AgentCapability(
            agent_type=self.agent_type,
            task_types=list(self.supported_task_types),
            resource_classes=list(self.resource_classes),
            max_concurrency=self.max_concurrency,
            retry_policy=self.retry_policy,
        )


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
        self._agents: dict[AgentType, AgentDescriptor] = {
            descriptor.agent_type: descriptor
            for descriptor in (
                DataProcessingAgent.descriptor,
                TrainingOrchestrationAgent.descriptor,
                EvaluationBenchmarkingAgent.descriptor,
                MonitoringTelemetryAgent.descriptor,
                OptimizationFeedbackAgent.descriptor,
                DeploymentAgent.descriptor,
            )
        }

    def list_capabilities(self) -> list[AgentCapability]:
        return [descriptor.capability() for descriptor in self._agents.values()]

    def get(self, agent_type: AgentType) -> AgentDescriptor:
        return self._agents[agent_type]

    def agent_for_task_type(self, task_type: TaskType) -> AgentDescriptor:
        for descriptor in self._agents.values():
            if task_type in descriptor.supported_task_types:
                return descriptor
        return self._agents["monitoring_telemetry"]
