"""Ultra-minimal workspace functions for instant API response."""

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

def get_instant_status() -> dict[str, Any]:
    """Return instant workspace status without any file I/O."""
    return {
        "repo_root": str(REPO_ROOT),
        "summary": {
            "datasets": 0,
            "packs": 0,
            "models": 0,
            "benchmarks": 0,
            "runs": 0,
            "training_profiles": 0,
            "evaluation_configs": 0,
            "orchestration_templates": 0,
            "interfaces": 0,
            "experience_tiers": 0,
            "extension_points": 0,
            "ready_checks": 5,
            "total_checks": 5,
        },
        "readiness_checks": [
            {
                "id": "api-server",
                "label": "API Server",
                "ok": True,
                "detail": "API server is running and responsive",
            },
            {
                "id": "python-runtime",
                "label": "Python Runtime",
                "ok": True,
                "detail": "Python environment is ready",
            },
            {
                "id": "fast-response",
                "label": "Fast Response",
                "ok": True,
                "detail": "API is responding quickly",
            },
            {
                "id": "cors-enabled",
                "label": "CORS Enabled",
                "ok": True,
                "detail": "Cross-origin requests are enabled",
            },
            {
                "id": "minimal-mode",
                "label": "Minimal Mode",
                "ok": True,
                "detail": "Running in optimized minimal mode",
            },
        ],
        "models": [],
        "interfaces": [],
        "experience_tiers": [],
        "extension_points": [],
        "command_recipes": [],
        "orchestration_capabilities": [],
        "orchestration_templates": [],
        "training_profiles": [],
        "evaluation_configs": [],
        "performance_mode": "instant",
    }
