#!/usr/bin/env python3
"""Collect core quality metrics for the improvement plan."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def get_test_coverage() -> float | None:
    report_path = Path("artifacts") / "metrics" / "coverage.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["pytest", "--cov=ai_factory", "--cov-report", f"json:{report_path}", "-q"])
    if result.returncode != 0 or not report_path.exists():
        return None
    payload = json.loads(report_path.read_text())
    totals = payload.get("totals") or {}
    value = totals.get("percent_covered")
    return float(value) if isinstance(value, (int, float)) else None


def get_security_issue_count() -> int | None:
    report_path = Path("artifacts") / "metrics" / "bandit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["bandit", "-r", "ai_factory", "-f", "json", "-o", str(report_path)])
    if result.returncode not in {0, 1} or not report_path.exists():
        return None
    payload = json.loads(report_path.read_text())
    return len(payload.get("results") or [])


def main() -> None:
    metrics: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_coverage_percent": get_test_coverage(),
        "security_issues": get_security_issue_count(),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
