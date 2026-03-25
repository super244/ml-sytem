from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any


class JsonlTelemetryLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> str:
        event_id = uuid.uuid4().hex[:12]
        body = {"event_id": event_id, "event_type": event_type, **payload}
        with self.path.open("a") as handle:
            handle.write(json.dumps(body, ensure_ascii=False) + "\n")
        return event_id
