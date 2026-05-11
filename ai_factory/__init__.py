"""Shared utilities for the ai-factory monorepo.

This package hosts cross-cutting types (schemas), hashing/fingerprints, and
artifact/manifest helpers used by the orchestration layer and inference stack.
"""

from ai_factory.version import VERSION

__version__ = VERSION

__all__ = ["VERSION", "__version__"]
