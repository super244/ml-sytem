from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Path | None:
    base = Path(checkpoints_dir)
    if not base.exists():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir() and _CHECKPOINT_PATTERN.match(path.name)]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float, str]:
        match = _CHECKPOINT_PATTERN.match(path.name)
        step = int(match.group(1)) if match else -1
        return (step, path.stat().st_mtime, path.name)

    return max(candidates, key=sort_key)


def resolve_resume_checkpoint(
    checkpoint_dir: str | Path,
    explicit_checkpoint: str | None = None,
    resume_from_latest: bool = False,
) -> str | None:
    if explicit_checkpoint:
        return explicit_checkpoint
    if not resume_from_latest:
        return None
    latest = find_latest_checkpoint(checkpoint_dir)
    return str(latest) if latest is not None else None
