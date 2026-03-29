"""CLI interface implementation."""

from __future__ import annotations

from pathlib import Path

from ai_factory.cli import main as legacy_main
from ai_factory.core.platform.container import build_platform_container


class CLIInterface:
    """Unified CLI interface for AI-Factory."""

    def __init__(self, repo_root: Path | None = None, artifacts_dir: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
        self.container = build_platform_container(
            repo_root=self.repo_root,
            artifacts_dir=self.artifacts_dir,
        )

    def run(self, args: list | None = None) -> None:
        """Run the CLI interface."""
        import sys

        original_argv = sys.argv[:]
        if args is not None:
            sys.argv = ["ai-factory"] + args
        try:
            legacy_main()
        finally:
            sys.argv = original_argv
