"""Mathematics domain configuration."""

from typing import Literal

from pydantic import BaseModel, Field

from ai_factory.core.model_scales import default_foundation_model_ref
from ai_factory.core.schemas import MetricSpec


class MathDomainConfig(BaseModel):
    """Configuration for the mathematics domain."""

    name: Literal["mathematics"] = "mathematics"
    version: str = "1.0.0"
    description: str = "Mathematical reasoning and calculus domain"

    # Sub-domains within mathematics
    subdomains: list[str] = Field(
        default_factory=lambda: ["calculus", "algebra", "geometry", "olympiad", "statistics", "linear_algebra"]
    )

    # Supported dataset families
    dataset_families: list[str] = Field(
        default_factory=lambda: [
            "derivatives",
            "integrals",
            "limits_series",
            "multivariable",
            "odes_optimization",
            "olympiad_reasoning",
        ]
    )

    # Domain-specific metrics
    metrics: list[MetricSpec] = Field(default_factory=list)

    # Default models for this domain
    default_models: list[str] = Field(
        default_factory=lambda: [default_foundation_model_ref("2b"), default_foundation_model_ref("12b")]
    )

    # Specialized evaluation benchmarks
    benchmarks: list[str] = Field(
        default_factory=lambda: ["mathematics_benchmark", "calculus_specialist", "olympiad_reasoning"]
    )
