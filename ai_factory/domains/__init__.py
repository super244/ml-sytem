"""Domain-specific modules for AI-Factory.

This package contains domain-specific implementations for different
AI applications and use cases. Each domain provides:

- Domain-specific datasets and data processing
- Specialized training configurations
- Domain-appropriate evaluation metrics
- Custom inference interfaces

Available domains:
- mathematics: Mathematical reasoning and calculus
- code: Code generation and programming tasks
- reasoning: General reasoning and logic puzzles
- creative: Creative writing and content generation
"""

from .mathematics import (
    MathDatasetRegistry,
    MathDomainConfig,
    MathEvaluationSuite,
    MathTrainingProfiles,
)
from .utils import get_domain_info, list_available_domains

__all__ = [
    "MathDomainConfig",
    "MathDatasetRegistry",
    "MathEvaluationSuite",
    "MathTrainingProfiles",
    "list_available_domains",
    "get_domain_info",
]
