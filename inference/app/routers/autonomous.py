from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/experiments/autonomous", tags=["autonomous"])

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_FILE = REPO_ROOT / "data" / "autonomous" / "experiments.jsonl"


def _load_experiments() -> dict[str, dict[str, Any]]:
    if not EXPERIMENTS_FILE.exists():
        return {}
    store: dict[str, dict[str, Any]] = {}
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                store[obj["experiment_id"]] = obj
            except Exception:
                pass
    return store


def _save_experiment(exp: dict[str, Any]) -> None:
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_exps = _load_experiments()
    all_exps[exp["experiment_id"]] = exp
    with open(EXPERIMENTS_FILE, "w") as f:
        for e in all_exps.values():
            f.write(json.dumps(e) + "\n")


async def _experiment_worker(experiment_id: str) -> None:
    stages = ["running", "analyzing", "completed"]
    for stage in stages:
        await asyncio.sleep(2.0)  # Simulate workload progression
        exps = _load_experiments()
        if experiment_id in exps:
            exps[experiment_id]["status"] = stage
            _save_experiment(exps[experiment_id])


class AutonomousExperimentRequest(BaseModel):
    experiment_name: str = Field(..., description="The name of the autonomous experiment")
    goal: str = Field(..., description="The primary goal or objective of the experiment")
    parameters: dict[str, str | int | float | bool] | None = Field(
        default=None, description="Optional parameters for the experiment"
    )


class AutonomousExperimentResponse(BaseModel):
    experiment_id: str = Field(..., description="The unique identifier for the started experiment")
    status: str = Field(..., description="The current status of the experiment")
    message: str = Field(..., description="Status message regarding the experiment initiation")


@router.post("/run", response_model=AutonomousExperimentResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_autonomous_experiment(
    request: AutonomousExperimentRequest, background_tasks: BackgroundTasks
) -> AutonomousExperimentResponse:
    """
    Start a new autonomous AI-driven experiment.
    """
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

    exp_data = {
        "experiment_id": experiment_id,
        "experiment_name": request.experiment_name,
        "goal": request.goal,
        "parameters": request.parameters or {},
        "status": "accepted",
    }
    _save_experiment(exp_data)

    background_tasks.add_task(_experiment_worker, experiment_id)

    return AutonomousExperimentResponse(
        experiment_id=experiment_id,
        status="accepted",
        message=f"Autonomous experiment '{request.experiment_name}' started successfully.",
    )


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    exps = _load_experiments()
    if experiment_id not in exps:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exps[experiment_id]
