from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileResponseCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def stats(self) -> dict[str, Any]:
        files = list(self.root.glob("*.json"))
        return {
            "entries": len(files),
            "size_bytes": sum(file.stat().st_size for file in files),
            "root": str(self.root),
        }
