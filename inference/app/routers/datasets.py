"""Datasets router — real telemetry reads + durable synthesis job tracking."""
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from inference.app.dependencies import get_metadata_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_DIR = REPO_ROOT / "data" / "telemetry"
SYNTH_JOBS_FILE = REPO_ROOT / "data" / "synth_jobs.jsonl"


class SynthesizeRequest(BaseModel):
    seed_prompt: str
    num_variants: int
    model_variant: str


def _load_synth_jobs() -> dict[str, dict[str, Any]]:
    if not SYNTH_JOBS_FILE.exists():
        return {}
    jobs: dict[str, dict[str, Any]] = {}
    with open(SYNTH_JOBS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                jobs[obj["job_id"]] = obj
            except Exception:
                pass
    return jobs


def _save_job(job: dict[str, Any]) -> None:
    SYNTH_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_jobs = _load_synth_jobs()
    all_jobs[job["job_id"]] = job
    with open(SYNTH_JOBS_FILE, "w") as f:
        for j in all_jobs.values():
            f.write(json.dumps(j) + "\n")


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
    # Sort most recent first
    results.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return {"telemetry": results}


@router.post("/synthesize")
def synthesize_dataset(req: SynthesizeRequest) -> dict[str, Any]:
    job_id = f"synth-{uuid.uuid4().hex[:8]}"
    estimated_time = req.num_variants * 1.5
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "seed_prompt": req.seed_prompt,
        "num_variants": req.num_variants,
        "model_variant": req.model_variant,
        "created_at": time.time(),
        "estimated_time_s": estimated_time,
        "completed_rows": 0,
        "output_path": str(REPO_ROOT / "data" / "synth" / f"{job_id}.jsonl"),
    }
    _save_job(job)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"Synthesizing {req.num_variants} rows using {req.model_variant}...",
        "estimated_time_s": estimated_time,
    }


@router.get("/synthesize/{job_id}")
def get_synthesis_job(job_id: str) -> dict[str, Any]:
    jobs = _load_synth_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = jobs[job_id]
    # Check if output file exists and count rows as proxy for progress
    output_path = Path(job.get("output_path", ""))
    if output_path.exists():
        with open(output_path) as f:
            completed = sum(1 for line in f if line.strip())
        job["completed_rows"] = completed
        if completed >= job["num_variants"]:
            job["status"] = "completed"
            _save_job(job)
    return job
