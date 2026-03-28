"""Web interface implementation."""

from typing import Optional
from pathlib import Path
import subprocess
import sys


class WebInterface:
    """Unified web interface for AI-Factory."""
    
    def __init__(self, repo_root: Optional[Path] = None, artifacts_dir: Optional[Path] = None):
        self.repo_root = repo_root or Path.cwd()
        self.artifacts_dir = artifacts_dir or self.repo_root / "artifacts"
    
    def run_backend(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
        """Run the FastAPI backend server."""
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "inference.app.main:app",
            "--host", host,
            "--port", str(port)
        ]
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, cwd=self.repo_root)
    
    def run_frontend(self, port: int = 3000) -> None:
        """Run the Next.js frontend development server."""
        frontend_dir = self.repo_root / "frontend"
        if not frontend_dir.exists():
            raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")
        
        # Set environment variable for API base URL
        env = {"NEXT_PUBLIC_API_BASE_URL": "http://127.0.0.1:8000"}
        
        cmd = ["npm", "run", "dev", "--", "--port", str(port)]
        subprocess.run(cmd, cwd=frontend_dir, env=env)
