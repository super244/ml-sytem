"""Mathematics domain configuration."""

from typing import List, Literal
from pydantic import BaseModel, Field

from ai_factory.core.schemas import DatasetSpec, MetricSpec


class MathDomainConfig(BaseModel):
    """Configuration for the mathematics domain."""
    
    name: Literal["mathematics"] = "mathematics"
    version: str = "1.0.0"
    description: str = "Mathematical reasoning and calculus domain"
    
    # Sub-domains within mathematics
    subdomains: List[str] = Field(
        default_factory=lambda: [
            "calculus",
            "algebra", 
            "geometry",
            "olympiad",
            "statistics",
            "linear_algebra"
        ]
    )
    
    # Supported dataset families
    dataset_families: List[str] = Field(
        default_factory=lambda: [
            "derivatives",
            "integrals", 
            "limits_series",
            "multivariable",
            "odes_optimization",
            "olympiad_reasoning"
        ]
    )
    
    # Domain-specific metrics
    metrics: List[MetricSpec] = Field(default_factory=list)
    
    # Default models for this domain
    default_models: List[str] = Field(
        default_factory=lambda: [
            "Qwen2.5-Math-1.5B-Instruct",
            "Qwen2.5-Math-7B-Instruct"
        ]
    )
    
    # Specialized evaluation benchmarks
    benchmarks: List[str] = Field(
        default_factory=lambda: [
            "mathematics_benchmark",
            "calculus_specialist",
            "olympiad_reasoning"
        ]
    )
