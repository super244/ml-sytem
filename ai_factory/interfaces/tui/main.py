"""TUI interface implementation."""

from pathlib import Path

from ai_factory.core.platform.container import build_platform_container
from ai_factory.tui import run_tui as legacy_run_tui


class TUIInterface:
    """Unified TUI interface for AI-Factory."""
    
    def __init__(self, repo_root: Path | None = None, artifacts_dir: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
        self.container = build_platform_container(
            repo_root=self.repo_root,
            artifacts_dir=self.artifacts_dir
        )
    
    def run(self, refresh_seconds: float = 2.0) -> None:
        """Run the TUI interface."""
        # Delegate to existing TUI implementation
        legacy_run_tui(
            repo_root=str(self.repo_root),
            artifacts_dir=str(self.artifacts_dir),
            refresh_seconds=refresh_seconds
        )
