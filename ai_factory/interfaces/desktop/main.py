"""Desktop interface implementation."""

from pathlib import Path
import subprocess


class DesktopInterface:
    """Unified desktop interface for AI-Factory."""
    
    def __init__(self, repo_root: Path | None = None, artifacts_dir: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
        self.desktop_dir = self.repo_root / "desktop"
    
    def run(self) -> None:
        """Run the desktop application."""
        if not self.desktop_dir.exists():
            raise FileNotFoundError(f"Desktop directory not found: {self.desktop_dir}")
        
        # Check if Node.js dependencies are installed
        node_modules = self.desktop_dir / "node_modules"
        if not node_modules.exists():
            print("Installing desktop dependencies...")
            subprocess.run(["npm", "install"], cwd=self.desktop_dir)
        
        # Run the Electron application
        subprocess.run(["npm", "start"], cwd=self.desktop_dir)
    
    def build(self) -> None:
        """Build the desktop application for distribution."""
        if not self.desktop_dir.exists():
            raise FileNotFoundError(f"Desktop directory not found: {self.desktop_dir}")
        
        subprocess.run(["npm", "run", "build"], cwd=self.desktop_dir)
