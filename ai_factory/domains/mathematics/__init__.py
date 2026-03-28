"""Mathematics domain for AI-Factory.

This domain provides specialized capabilities for mathematical reasoning,
including calculus, algebra, olympiad problems, and advanced mathematics.
"""

from .config import MathDomainConfig
from .datasets import MathDatasetRegistry
from .evaluation import MathEvaluationSuite
from .training import MathTrainingProfiles

__all__ = [
    "MathDomainConfig",
    "MathDatasetRegistry", 
    "MathEvaluationSuite",
    "MathTrainingProfiles",
]
