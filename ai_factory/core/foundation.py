from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_factory.core.discovery import load_benchmark_registry
from ai_factory.core.instances.models import UserLevel


InterfaceId = Literal["cli", "tui", "web", "desktop"]
ExtensionKind = Literal["training_method", "evaluation_suite", "deployment_target"]


class InterfaceSurface(BaseModel):
    id: InterfaceId
    label: str
    entrypoint: str
    backend_contract: str
    description: str
    status: Literal["ready", "stub", "planned"] = "ready"


class ExperienceTier(BaseModel):
    id: UserLevel
    label: str
    description: str
    visible_controls: list[str] = Field(default_factory=list)
    recommended_modes: list[str] = Field(default_factory=list)
    safe_defaults: list[str] = Field(default_factory=list)


class ExtensionPoint(BaseModel):
    id: str
    kind: ExtensionKind
    label: str
    description: str
    supported_instance_types: list[str] = Field(default_factory=list)
    source: Literal["built_in", "benchmark_registry"] = "built_in"
    maturity: Literal["ready", "beta", "placeholder"] = "ready"
    config_hint: str | None = None
    future_ready: bool = False


class FoundationCatalog(BaseModel):
    interfaces: list[InterfaceSurface] = Field(default_factory=list)
    experience_tiers: list[ExperienceTier] = Field(default_factory=list)
    extension_points: list[ExtensionPoint] = Field(default_factory=list)


def _interfaces() -> list[InterfaceSurface]:
    return [
        InterfaceSurface(
            id="cli",
            label="CLI",
            entrypoint="ai_factory.cli:main",
            backend_contract="PlatformContainer -> InstanceManager -> LifecycleStateManager",
            description="Primary control surface for managed lifecycle operations, automation, and operator workflows.",
            status="ready",
        ),
        InterfaceSurface(
            id="tui",
            label="TUI",
            entrypoint="ai_factory.tui:run_tui",
            backend_contract="PlatformContainer -> InstanceManager -> LifecycleStateManager",
            description="Interactive terminal dashboard for active jobs, logs, metrics, and branch detail.",
            status="ready",
        ),
        InterfaceSurface(
            id="web",
            label="Web Dashboard",
            entrypoint="inference.app.main:app + frontend/",
            backend_contract="FastAPI InstanceService -> InstanceManager -> LifecycleStateManager",
            description="Browser-based control center for monitoring, branch management, evaluation, and inference.",
            status="ready",
        ),
        InterfaceSurface(
            id="desktop",
            label="Desktop App",
            entrypoint="desktop/main.js",
            backend_contract="Desktop shell -> FastAPI/CLI backend contracts",
            description="Electron shell that wraps the same web/API control center for macOS-oriented local operation.",
            status="stub",
        ),
    ]


def _experience_tiers() -> list[ExperienceTier]:
    return [
        ExperienceTier(
            id="beginner",
            label="Beginner",
            description="Guided workflows with safe defaults, minimal overrides, and click-to-run progression.",
            visible_controls=["templates", "guided actions", "status", "logs"],
            recommended_modes=["single"],
            safe_defaults=["local execution", "dry-run deployment hooks", "bounded sub-agents"],
        ),
        ExperienceTier(
            id="hobbyist",
            label="Hobbyist",
            description="Balanced control with common tuning knobs, follow-up workflows, and manageable remote options.",
            visible_controls=["templates", "hyperparameters", "remote profiles", "recommendations"],
            recommended_modes=["single", "local_parallel", "hybrid"],
            safe_defaults=["guardrailed overrides", "limited concurrency", "managed evaluation loop"],
        ),
        ExperienceTier(
            id="dev",
            label="Developer",
            description="Full-fidelity lifecycle control including raw configs, backend selection, and architecture-level decisions.",
            visible_controls=["raw configs", "command overrides", "agent policies", "publish hooks"],
            recommended_modes=["single", "local_parallel", "cloud_parallel", "hybrid"],
            safe_defaults=["no forced UI guardrails", "explicit control-plane visibility"],
        ),
    ]


def _training_extensions() -> list[ExtensionPoint]:
    return [
        ExtensionPoint(
            id="supervised",
            kind="training_method",
            label="Supervised Training",
            description="Standard managed training and instruction-style fine-tuning against curated datasets.",
            supported_instance_types=["train", "finetune"],
            config_hint="configs/train.yaml",
        ),
        ExtensionPoint(
            id="unsupervised",
            kind="training_method",
            label="Unsupervised Training",
            description="Pretraining-style dataset iteration for continued base-model adaptation.",
            supported_instance_types=["train"],
            config_hint="configs/train.yaml",
            maturity="beta",
        ),
        ExtensionPoint(
            id="lora",
            kind="training_method",
            label="LoRA",
            description="Parameter-efficient adapter tuning using the existing training stack.",
            supported_instance_types=["finetune"],
            config_hint="configs/finetune.yaml",
        ),
        ExtensionPoint(
            id="qlora",
            kind="training_method",
            label="QLoRA",
            description="Quantized LoRA path for resource-efficient local and remote iterations.",
            supported_instance_types=["finetune"],
            config_hint="configs/finetune.yaml",
        ),
        ExtensionPoint(
            id="full_finetune",
            kind="training_method",
            label="Full Finetune",
            description="Full-parameter update path for larger iteration budgets and deeper specialization.",
            supported_instance_types=["finetune"],
            config_hint="configs/finetune.yaml",
            maturity="beta",
        ),
        ExtensionPoint(
            id="rlhf",
            kind="training_method",
            label="RLHF Placeholder",
            description="Reserved extension point for future preference and reinforcement learning workflows.",
            supported_instance_types=["finetune"],
            maturity="placeholder",
            future_ready=True,
        ),
    ]


def _deployment_extensions() -> list[ExtensionPoint]:
    return [
        ExtensionPoint(
            id="huggingface",
            kind="deployment_target",
            label="Hugging Face",
            description="Publish exported artifacts to a Hub repository.",
            supported_instance_types=["deploy"],
            config_hint="configs/deploy.yaml",
        ),
        ExtensionPoint(
            id="ollama",
            kind="deployment_target",
            label="Ollama",
            description="Package or register a model for Ollama-based local serving.",
            supported_instance_types=["deploy"],
            config_hint="configs/deploy.yaml",
        ),
        ExtensionPoint(
            id="lmstudio",
            kind="deployment_target",
            label="LM Studio",
            description="Export artifacts for LM Studio import flows and local desktop inference.",
            supported_instance_types=["deploy"],
            config_hint="configs/deploy.yaml",
            maturity="beta",
        ),
        ExtensionPoint(
            id="openai_compatible_api",
            kind="deployment_target",
            label="OpenAI-Compatible API",
            description="Register or publish an endpoint that exposes an OpenAI-style inference surface.",
            supported_instance_types=["deploy"],
            config_hint="configs/deploy.yaml",
        ),
        ExtensionPoint(
            id="custom_api",
            kind="deployment_target",
            label="Custom API",
            description="Call an arbitrary API target as part of a managed deployment step.",
            supported_instance_types=["deploy"],
            config_hint="configs/deploy.yaml",
            maturity="beta",
        ),
    ]


def _evaluation_extensions(repo_root: Path) -> list[ExtensionPoint]:
    extensions = [
        ExtensionPoint(
            id="custom_dataset",
            kind="evaluation_suite",
            label="Custom Dataset Evaluation",
            description="Run managed evaluations against user-provided or domain-specific datasets.",
            supported_instance_types=["evaluate"],
            config_hint="configs/eval.yaml",
        )
    ]
    registry_path = repo_root / "evaluation" / "benchmarks" / "registry.yaml"
    for benchmark in load_benchmark_registry(registry_path):
        benchmark_id = str(benchmark.get("id") or "benchmark")
        extensions.append(
            ExtensionPoint(
                id=f"benchmark:{benchmark_id}",
                kind="evaluation_suite",
                label=str(benchmark.get("title") or benchmark_id.replace("_", " ").title()),
                description=str(
                    benchmark.get("description")
                    or f"Benchmark suite registered in {registry_path.relative_to(repo_root)}."
                ),
                supported_instance_types=["evaluate"],
                source="benchmark_registry",
                config_hint="configs/eval.yaml",
            )
        )
    return extensions


def build_foundation_catalog(repo_root: str | Path | None = None) -> FoundationCatalog:
    resolved_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    return FoundationCatalog(
        interfaces=_interfaces(),
        experience_tiers=_experience_tiers(),
        extension_points=[
            *_training_extensions(),
            *_evaluation_extensions(resolved_root),
            *_deployment_extensions(),
        ],
    )
