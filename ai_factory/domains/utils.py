"""Utility functions for domain management."""

from typing import List, Dict, Any
from pathlib import Path

from .mathematics.config import MathDomainConfig
from .mathematics.datasets import MathDatasetRegistry
from .mathematics.evaluation import MathEvaluationSuite
from .mathematics.training import MathTrainingProfiles


def list_available_domains() -> List[MathDomainConfig]:
    """List all available domains."""
    return [
        MathDomainConfig()
    ]


def get_domain_info(domain_name: str) -> Dict[str, Any]:
    """Get information about a specific domain."""
    if domain_name == "mathematics":
        config = MathDomainConfig()
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "subdomains": config.subdomains,
            "dataset_families": config.dataset_families,
            "benchmarks": config.benchmarks
        }
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
