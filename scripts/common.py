from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_root() -> Path:
    return REPO_ROOT


def active_python() -> str:
    return sys.executable


def run_step(label: str, command: list[str], cwd: Path | None = None) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[ai-factory] {label}: {rendered}")
    subprocess.run(command, cwd=str(cwd or REPO_ROOT), check=True)


def emit_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for nested_key, nested_value in value.items():
                print(f"  - {nested_key}: {nested_value}")
            continue
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
            continue
        print(f"{key}: {value}")
