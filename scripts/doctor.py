#!/usr/bin/env python3
"""AI-Factory system health check script."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Checking {description}...")
    try:
        subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description}: {e.stderr.strip()}")
        return False


def main():
    """Run system health checks."""
    print("🏭 AI-Factory System Health Check")
    print("=" * 40)

    checks = []

    # Core imports
    checks.append(("python -c 'import ai_factory'", "Core imports"))

    # CLI functionality
    checks.append(("python -m ai_factory.cli --help", "CLI functionality"))

    # Test collection
    checks.append(("python -m pytest --collect-only -q", "Test discovery"))

    # Code formatting
    checks.append(("ruff check --quiet", "Code linting"))

    # Type checking (optional)
    checks.append(("mypy ai_factory/ --no-error-summary", "Type checking"))

    # Frontend dependencies
    frontend_path = Path("frontend")
    if frontend_path.exists():
        checks.append(("cd frontend && npm list --depth=0 >/dev/null", "Frontend dependencies"))

    # Configuration files
    config_files = [
        "pyproject.toml",
        "configs/finetune.yaml",
        "training/configs/profiles/baseline_qlora.yaml",
        "inference/configs/model_registry.yaml",
    ]

    for config_file in config_files:
        if Path(config_file).exists():
            checks.append((f"test -f {config_file}", f"Config file: {config_file}"))

    # Run checks
    passed = 0
    total = len(checks)

    for cmd, desc in checks:
        if run_command(cmd, desc):
            passed += 1
        print()

    # Summary
    print("=" * 40)
    print(f"Health Check: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All systems operational!")
        return 0
    else:
        print("⚠️  Some issues detected. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
