"""Datasets router — real telemetry reads + durable synthesis job tracking."""

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ai_factory.core.orchestration.sqlite import SqliteControlPlane
from ai_factory.core.platform.settings import get_platform_settings
from inference.app.config import get_settings
from inference.app.dependencies import get_metadata_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_db() -> SqliteControlPlane:
    settings = get_settings()
    platform_settings = get_platform_settings(repo_root=settings.repo_root)
    return SqliteControlPlane(platform_settings.control_db_path)


class SynthesizeRequest(BaseModel):
    seed_prompt: str
    num_variants: int
    model_variant: str


def _telemetry_record_id(record: dict[str, Any]) -> str:
    # Remove 'id' if present to ensure consistent hashing for the same payload
    payload_to_hash = {k: v for k, v in record.items() if k not in ("id", "status", "actioned_at")}
    payload = json.dumps(payload_to_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


@router.get("")
def list_datasets(service: Any = Depends(get_metadata_service)) -> dict[str, Any]:
    try:
        return service.dataset_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Datasets service unavailable: {str(exc)}") from exc


@router.get("/telemetry")
def get_telemetry_backlog(db: SqliteControlPlane = Depends(_get_db)) -> dict[str, Any]:
    try:
        
        records = db.list_telemetry(status="flagged")
        return {"telemetry": records}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Telemetry backlog unavailable: {str(exc)}") from exc


@router.post("/telemetry/{record_id}/promote")
def promote_telemetry_record(record_id: str, db: SqliteControlPlane = Depends(_get_db)) -> dict[str, Any]:
    try:
        
        updated = db.update_telemetry_status(record_id, status="promoted", actioned_at=time.time())
        if not updated:
            raise HTTPException(status_code=404, detail=f"Telemetry record '{record_id}' not found")
        return {
            "status": "promoted",
            "record": updated,
            "destination": "db:promoted",
            "message": "Telemetry record promoted into the curation queue.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to promote telemetry: {str(exc)}") from exc


@router.post("/telemetry/{record_id}/discard")
def discard_telemetry_record(record_id: str, db: SqliteControlPlane = Depends(_get_db)) -> dict[str, Any]:
    try:
        
        updated = db.update_telemetry_status(record_id, status="discarded", actioned_at=time.time())
        if not updated:
            raise HTTPException(status_code=404, detail=f"Telemetry record '{record_id}' not found")
        return {
            "status": "discarded",
            "record": updated,
            "destination": "db:discarded",
            "message": "Telemetry record discarded from the active backlog.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to discard telemetry: {str(exc)}") from exc


@router.post("/synthesize")
def synthesize_dataset(req: SynthesizeRequest, db: SqliteControlPlane = Depends(_get_db)) -> dict[str, Any]:
    try:
        
        job_id = f"synth-{uuid.uuid4().hex[:8]}"
        estimated_time = req.num_variants * 1.5
        settings = get_settings()
        output_path = str(Path(settings.repo_root) / "data" / "synth" / f"{job_id}.jsonl")
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "seed_prompt": req.seed_prompt,
            "num_variants": req.num_variants,
            "model_variant": req.model_variant,
            "created_at": time.time(),
            "estimated_time_s": estimated_time,
            "completed_rows": 0,
            "output_path": output_path,
        }
        db.upsert_synth_job(job)
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Synthesizing {req.num_variants} rows using {req.model_variant}...",
            "estimated_time_s": estimated_time,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Synthesis unavailable: {str(exc)}") from exc


@router.get("/synthesize/{job_id}")
def get_synthesis_job(job_id: str, db: SqliteControlPlane = Depends(_get_db)) -> dict[str, Any]:
    try:
        
        job = db.get_synth_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        
        # Check if output file exists and count rows as proxy for progress
        output_path = Path(job.get("output_path", ""))
        if output_path.exists():
            with open(output_path) as f:
                completed = sum(1 for line in f if line.strip())
            job["completed_rows"] = completed
            if completed >= job["num_variants"] and job["status"] != "completed":
                job["status"] = "completed"
                db.upsert_synth_job(job)
        return job
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Synthesis job retrieval unavailable: {str(exc)}") from exc

