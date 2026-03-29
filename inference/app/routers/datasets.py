import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from inference.app.dependencies import get_metadata_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_DIR = REPO_ROOT / "data" / "telemetry"

class SynthesizeRequest(BaseModel):
    seed_prompt: str
    num_variants: int
    model_variant: str

@router.get("")
def list_datasets() -> dict[str, Any]:
    return get_metadata_service().dataset_dashboard()

@router.get("/telemetry")
def get_telemetry_backlog() -> dict[str, Any]:
    flagged_file = TELEMETRY_DIR / "flagged.jsonl"
    results = []
    if flagged_file.exists():
        with open(flagged_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    results.append(data)
                except Exception:
                    pass
    return {"telemetry": results}

@router.post("/synthesize")
def synthesize_dataset(req: SynthesizeRequest) -> dict[str, Any]:
    # In a full product, this would dispatch an orchestrator task.
    # For architecture speed, we simulate the start of an LLM generation.
    job_id = f"synth-{uuid.uuid4().hex[:8]}"
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"Synthesizing {req.num_variants} rows using {req.model_variant}...",
        "estimated_time_s": req.num_variants * 1.5
    }
