from __future__ import annotations

import math
import random
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/automl", tags=["automl"])

# In-memory sweep store (ephemeral, persisted to disk in a real implementation)
_SWEEPS: dict[str, dict[str, Any]] = {}


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


def _generate_mock_trials(num_trials: int, search_space: SearchSpace) -> list[dict[str, Any]]:
    """Generate plausible mock hyperparameter trials with realistic loss curves."""
    trials = []
    for i in range(num_trials):
        lr = random.choice(search_space.learning_rate)
        bs = random.choice(search_space.batch_size)
        wr = random.choice(search_space.warmup_ratio)
        rank = random.choice(search_space.lora_rank)

        # Simulate a realistic final loss based on hyperparams
        base_loss = 0.8 + random.gauss(0, 0.1)
        # Lower LR penalty
        if lr < 1e-4:
            base_loss += 0.15
        # Higher LoRA rank benefit
        if rank >= 16:
            base_loss -= 0.08
        final_loss = max(0.2, base_loss)
        accuracy = min(0.99, 1.0 - final_loss * 0.6 + random.gauss(0, 0.02))

        trials.append({
            "trial_id": f"trial-{i:02d}",
            "status": "completed" if i < num_trials - 1 else "running",
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
            "duration_s": random.randint(60, 600),
        })

    # Sort best first
    trials.sort(key=lambda t: t["metrics"]["final_loss"])
    return trials


@router.get("/sweeps")
def list_sweeps() -> dict[str, Any]:
    return {"sweeps": list(_SWEEPS.values())}


@router.post("/sweeps")
def launch_sweep(req: LaunchSweepRequest) -> dict[str, Any]:
    sweep_id = f"sweep-{uuid.uuid4().hex[:8]}"
    trials = _generate_mock_trials(req.num_trials, req.search_space)
    best = trials[0]

    sweep = {
        "id": sweep_id,
        "name": req.name,
        "base_model": req.base_model,
        "strategy": req.strategy,
        "status": "running",
        "num_trials": req.num_trials,
        "completed_trials": req.num_trials - 1,
        "created_at": time.time(),
        "best_trial": best,
        "trials": trials,
    }
    _SWEEPS[sweep_id] = sweep
    return sweep


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict[str, Any]:
    if sweep_id not in _SWEEPS:
        raise HTTPException(status_code=404, detail=f"Sweep '{sweep_id}' not found")
    return _SWEEPS[sweep_id]
