"""CLI interface implementation."""

from typing import Optional
from pathlib import Path

from ai_factory.core.platform.container import build_platform_container
from ai_factory.cli import main as legacy_main


class CLIInterface:
    """Unified CLI interface for AI-Factory."""
    
    def __init__(self, repo_root: Optional[Path] = None, artifacts_dir: Optional[Path] = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
        self.container = build_platform_container(
            repo_root=self.repo_root,
            artifacts_dir=self.artifacts_dir
        )
    
    def run(self, args: Optional[list] = None) -> None:
        """Run the CLI interface."""
        # Delegate to existing CLI implementation
        # This maintains compatibility while providing the new interface structure
        import sys
        if args is not None:
            sys.argv = ["ai-factory"] + args
        legacy_main()
