"""Desktop interface implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class DesktopInterface:
    """Unified desktop interface for AI-Factory.

    Electron remains the default shell, while the native macOS scaffold
    provides a direct path to a SwiftUI-based application.
    """

    def __init__(self, repo_root: Path | None = None, artifacts_dir: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
        self.desktop_dir = self.repo_root / "desktop"
        self.native_dir = self.desktop_dir / "macos" / "AIFactoryNative"

    def native_available(self) -> bool:
        return self.native_dir.exists()

    def _should_use_native_shell(self) -> bool:
        return (
            os.environ.get("AI_FACTORY_DESKTOP_NATIVE") == "1"
            and sys.platform == "darwin"
            and self.native_available()
        )

    def build_native(self) -> None:
        if not self.native_available():
            raise FileNotFoundError(f"Native macOS scaffold not found: {self.native_dir}")
        subprocess.run(["swift", "build"], cwd=self.native_dir, check=True)

    def run_native(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("The native macOS desktop shell can only run on macOS.")
        if not self.native_available():
            raise FileNotFoundError(f"Native macOS scaffold not found: {self.native_dir}")
        subprocess.run(["swift", "run"], cwd=self.native_dir, check=True)

    def run(self) -> None:
        """Run the desktop application.

        By default this launches the Electron shell. Set
        ``AI_FACTORY_DESKTOP_NATIVE=1`` on macOS to run the SwiftUI shell
        instead.
        """
        if not self.desktop_dir.exists():
            raise FileNotFoundError(f"Desktop directory not found: {self.desktop_dir}")

        if self._should_use_native_shell():
            self.run_native()
            return

        node_modules = self.desktop_dir / "node_modules"
        if not node_modules.exists():
            print("Installing desktop dependencies...")
            subprocess.run(["npm", "install"], cwd=self.desktop_dir, check=True)

        subprocess.run(["npm", "start"], cwd=self.desktop_dir, check=True)

    def build(self) -> None:
        """Build the desktop application for distribution."""
        if not self.desktop_dir.exists():
            raise FileNotFoundError(f"Desktop directory not found: {self.desktop_dir}")

        subprocess.run(["npm", "run", "build"], cwd=self.desktop_dir, check=True)
