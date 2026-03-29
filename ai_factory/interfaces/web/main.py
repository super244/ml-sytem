"""Web interface implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class WebInterface:
    """Unified web interface for AI-Factory."""

    def __init__(self, repo_root: Path | None = None, artifacts_dir: Path | None = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"

    @property
    def backend_url(self) -> str:
        return "http://127.0.0.1:8000"

    @property
    def frontend_url(self) -> str:
        return "http://127.0.0.1:3000/workspace"

    def run_backend(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
        """Run the FastAPI backend server."""
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "inference.app.main:app",
            "--host", host,
            "--port", str(port)
        ]
        if reload:
            cmd.append("--reload")

        subprocess.run(cmd, cwd=self.repo_root, check=True)
    
    def run_frontend(self, port: int = 3000) -> None:
        """Run the Next.js frontend development server."""
        frontend_dir = self.repo_root / "frontend"
        if not frontend_dir.exists():
            raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")

        env = os.environ.copy()
        env["NEXT_PUBLIC_API_BASE_URL"] = self.backend_url

        cmd = ["npm", "run", "dev", "--", "--port", str(port)]
        subprocess.run(cmd, cwd=frontend_dir, env=env, check=True)
