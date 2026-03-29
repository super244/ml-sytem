"""Utility functions for domain management."""

from typing import Any

from .mathematics.config import MathDomainConfig


def list_available_domains() -> list[MathDomainConfig]:
    """List all available domains."""
    return [
        MathDomainConfig()
    ]


def get_domain_info(domain_name: str) -> dict[str, Any]:
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
