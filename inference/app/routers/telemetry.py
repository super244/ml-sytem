import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_DIR = REPO_ROOT / "data" / "telemetry"


class FlagTelemetryRequest(BaseModel):
    prompt: str
    assistant_output: str
    expected_output: str
    model_variant: str
    latency_s: float | None = None


@router.post("/flag")
async def flag_telemetry(request: FlagTelemetryRequest) -> dict[str, Any]:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TELEMETRY_DIR / "flagged.jsonl"
    record = {
        "timestamp": time.time(),
        "prompt": request.prompt,
        "assistant_output": request.assistant_output,
        "expected_output": request.expected_output,
        "model_variant": request.model_variant,
        "latency_s": request.latency_s,
    }
    with open(out_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "ok", "message": "Telemetry flagged successfully."}
