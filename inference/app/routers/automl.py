"""AutoML sweep orchestration — durable, file-backed sweep store."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from inference.app.config import get_settings

router = APIRouter(prefix="/automl", tags=["automl"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEPS_FILE = REPO_ROOT / "data" / "automl" / "sweeps.jsonl"


def _ensure_demo_mode() -> None:
    if not get_settings().demo_mode:
        raise HTTPException(
            status_code=503,
            detail="AutoML sweep simulation is disabled outside AI_FACTORY_DEMO_MODE=1.",
        )


def _load_sweeps() -> dict[str, dict[str, Any]]:
    if not SWEEPS_FILE.exists():
        return {}
    store: dict[str, dict[str, Any]] = {}
    with open(SWEEPS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                store[obj["id"]] = obj
            except Exception:
                pass
    return store


def _persist_sweep(sweep: dict[str, Any]) -> None:
    SWEEPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_sweeps = _load_sweeps()
    all_sweeps[sweep["id"]] = sweep
    with open(SWEEPS_FILE, "w") as f:
        for s in all_sweeps.values():
            f.write(json.dumps(s) + "\n")


class SearchSpace(BaseModel):
    learning_rate: list[float] = [1e-5, 1e-4, 1e-3]
    batch_size: list[int] = [8, 16, 32]
    warmup_ratio: list[float] = [0.03, 0.06, 0.1]
    lora_rank: list[int] = [4, 8, 16, 32]


class LaunchSweepRequest(BaseModel):
    name: str
    base_model: str
    strategy: str = "bayesian"  # bayesian | grid | random
    num_trials: int = 10
    search_space: SearchSpace = SearchSpace()


async def _sweep_worker(sweep_id: str, num_trials: int, search_space: SearchSpace) -> None:
    combinations = list(
        itertools.product(
            search_space.learning_rate, search_space.batch_size, search_space.warmup_ratio, search_space.lora_rank
        )
    )

    for i in range(num_trials):
        await asyncio.sleep(1.0)  # Simulate workload progression

        sweep = _load_sweeps().get(sweep_id)
        if not sweep:
            break

        lr, bs, wr, rank = combinations[i % len(combinations)]

        # Deterministic simulation of a trial
        base_loss = 0.8 + (i * 0.01)
        if lr < 1e-4:
            base_loss += 0.15
        if rank >= 16:
            base_loss -= 0.08
        final_loss = max(0.2, base_loss)
        accuracy = min(0.99, 1.0 - final_loss * 0.6)

        trial = {
            "trial_id": f"trial-{i:02d}",
            "status": "completed",
            "params": {
                "learning_rate": lr,
                "batch_size": bs,
                "warmup_ratio": wr,
                "lora_rank": rank,
            },
            "metrics": {
                "final_loss": round(final_loss, 4),
                "accuracy": round(accuracy, 4),
                "perplexity": round(math.exp(final_loss), 3),
            },
            "duration_s": 60 + (i * 10),
        }

        sweep["trials"].append(trial)
        sweep["completed_trials"] = len(sweep["trials"])

        best = sweep.get("best_trial")
        if not best or trial["metrics"]["final_loss"] < best["metrics"]["final_loss"]:
            sweep["best_trial"] = trial

        if sweep["completed_trials"] >= sweep["num_trials"]:
            sweep["status"] = "completed"

        _persist_sweep(sweep)


@router.get("/sweeps")
def list_sweeps() -> dict[str, Any]:
    sweeps = sorted(_load_sweeps().values(), key=lambda item: float(item.get("created_at", 0.0)), reverse=True)
    return {
        "status": "available",
        "write_enabled": get_settings().demo_mode,
        "sweeps": sweeps,
    }


@router.post("/sweeps")
def launch_sweep(req: LaunchSweepRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    _ensure_demo_mode()
    sweep_id = f"sweep-{uuid.uuid4().hex[:8]}"

    sweep: dict[str, Any] = {
        "id": sweep_id,
        "name": req.name,
        "base_model": req.base_model,
        "strategy": req.strategy,
        "status": "running",
        "num_trials": req.num_trials,
        "completed_trials": 0,
        "created_at": time.time(),
        "best_trial": None,
        "trials": [],
    }
    _persist_sweep(sweep)

    background_tasks.add_task(_sweep_worker, sweep_id, req.num_trials, req.search_space)

    return sweep


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict[str, Any]:
    sweeps = _load_sweeps()
    if sweep_id not in sweeps:
        raise HTTPException(status_code=404, detail=f"Sweep '{sweep_id}' not found")
    return sweeps[sweep_id]
