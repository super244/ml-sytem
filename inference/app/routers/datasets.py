"""Datasets router — real telemetry reads + durable synthesis job tracking."""

import hashlib
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
PROMOTED_TELEMETRY_FILE = TELEMETRY_DIR / "promoted.jsonl"
DISCARDED_TELEMETRY_FILE = TELEMETRY_DIR / "discarded.jsonl"


class SynthesizeRequest(BaseModel):
    seed_prompt: str
    num_variants: int
    model_variant: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _telemetry_record_id(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _annotate_telemetry_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["id"] = _telemetry_record_id(record)
    return payload


def _flagged_records_with_ids() -> list[dict[str, Any]]:
    flagged_file = TELEMETRY_DIR / "flagged.jsonl"
    records = [_annotate_telemetry_record(record) for record in _load_jsonl(flagged_file)]
    records.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return records


def _move_flagged_record(record_id: str, *, destination: Path, status: str) -> dict[str, Any]:
    flagged_file = TELEMETRY_DIR / "flagged.jsonl"
    raw_flagged = _load_jsonl(flagged_file)
    remaining: list[dict[str, Any]] = []
    matched: dict[str, Any] | None = None

    for record in raw_flagged:
        current_id = _telemetry_record_id(record)
        if matched is None and current_id == record_id:
            matched = _annotate_telemetry_record(record)
            continue
        remaining.append(record)

    if matched is None:
        raise HTTPException(status_code=404, detail=f"Telemetry record '{record_id}' not found")

    _write_jsonl(flagged_file, remaining)
    archived = _load_jsonl(destination)
    archived.append(
        {
            **{key: value for key, value in matched.items() if key != "id"},
            "status": status,
            "actioned_at": time.time(),
        }
    )
    _write_jsonl(destination, archived)
    return matched


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
    return {"telemetry": _flagged_records_with_ids()}


@router.post("/telemetry/{record_id}/promote")
def promote_telemetry_record(record_id: str) -> dict[str, Any]:
    record = _move_flagged_record(
        record_id,
        destination=PROMOTED_TELEMETRY_FILE,
        status="promoted",
    )
    return {
        "status": "promoted",
        "record": record,
        "destination": str(PROMOTED_TELEMETRY_FILE.relative_to(REPO_ROOT)),
        "message": "Telemetry record promoted into the curation queue.",
    }


@router.post("/telemetry/{record_id}/discard")
def discard_telemetry_record(record_id: str) -> dict[str, Any]:
    record = _move_flagged_record(
        record_id,
        destination=DISCARDED_TELEMETRY_FILE,
        status="discarded",
    )
    return {
        "status": "discarded",
        "record": record,
        "destination": str(DISCARDED_TELEMETRY_FILE.relative_to(REPO_ROOT)),
        "message": "Telemetry record discarded from the active backlog.",
    }


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
