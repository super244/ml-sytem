# Agent Orchestration Guide

## Overview
AI-Factory uses an agent-based orchestration system to manage the complete ML lifecycle. Agents handle tasks from data processing through model deployment, with built-in retry policies, circuit breakers, and lineage tracking.

## Agent Types

### Core Agents
| Agent Type | Responsibility | Resource Class |
|------------|---------------|----------------|
| `data_processing` | Dataset synthesis, quality filtering, packing | io |
| `training_orchestration` | Model fine-tuning, distributed training jobs | gpu |
| `evaluation_benchmarking` | Benchmark execution, metric aggregation | cpu |
| `monitoring_telemetry` | Health monitoring, log collection, alerts | cpu |
| `deployment` | Model packaging, artifact push, registry updates | network |
| `optimization_feedback` | Hyperparameter tuning, failure recovery | gpu |

### Agent Lifecycle
1. **Task Registration** - Agent registered via decorator
2. **Capability Declaration** - Resource classes and retry policies defined
3. **Execution** - Task dispatched to available agent
4. **Circuit Tracking** - Failure count and recovery logic
5. **Lineage Recording** - All events logged with full provenance

## Agent Registration
```python
from ai_factory.core.orchestration.agents import agent

@agent(
    task_type="training",
    label="LoRA Fine-Tuning Agent",
    resource_classes=["gpu"],
    retry_policy=RetryPolicy(max_attempts=3, base_delay_s=5),
)
class TrainingAgent:
    def execute(self, task: Task) -> TaskResult:
        # Agent implementation
        pass
```

## Circuit Breakers
The orchestration system implements circuit breakers per agent type:
- **Closed** (default) - Agent is operational
- **Open** - Agent disabled after 3 consecutive failures
- **Half-Open** - Reopening after cooldown period

## Retry Logic
```python
RetryPolicy(
    max_attempts=3,
    base_delay_s=5,
    max_delay_s=300,
    multiplier=2.0,  # Exponential backoff
    jitter_s=10,     # Random jitter to prevent thundering herd
)
```

## Event System
All agent actions emit events with full lineage:
```json
{
  "event_type": "task.running",
  "agent_type": "training_orchestration",
  "payload": {"attempt": 1, "lease_owner": "worker-1"}
}
```

## Monitoring
Use `OrchestrationService.monitoring_summary()` to get platform status:
- Active runs and tasks
- Circuit breaker states
- Task type distribution
- Resource utilization

## Health Checks
- `recover_stalled_tasks()` - Auto-recovers heartbeat-expired tasks
- `is_circuit_open()` - Check if agent is disabled
- `latest_checkpoint()` - Get last successful checkpoint
