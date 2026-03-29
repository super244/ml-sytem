import random
import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/agents", tags=["agents"])

MOCK_AGENTS = [
    {
        "id": "agent-data-01",
        "name": "Data Curator",
        "role": "Synthesizes and filters noisy dataset rows.",
        "model": "gpt-4o",
        "status": "active",
        "uptime_s": 3600,
        "tokens_used": 145000,
    },
    {
        "id": "agent-train-01",
        "name": "Optimization Agent",
        "role": "Monitors loss curves and triggers hyperparam sweeps.",
        "model": "claude-3-sonnet",
        "status": "sleeping",
        "uptime_s": 86400,
        "tokens_used": 45000,
    },
    {
        "id": "agent-eval-01",
        "name": "Red-Team Evaluator",
        "role": "Adversarially attacks the model to find failure modes.",
        "model": "qwen-2-72b",
        "status": "active",
        "uptime_s": 1200,
        "tokens_used": 89000,
    }
]

LOG_POOL = [
    "[Data Curator] Flagged row #402 for low perplexity.",
    "[Optimization Agent] Loss plateau detected at epoch 4. Suggesting learning rate decay.",
    "[Red-Team Evaluator] Generated 50 adversarial prompts targeting logic failures.",
    "[Data Curator] Synthesizing 500 new logic puzzles...",
    "[Optimization Agent] Sleeping until next validation step.",
    "[Red-Team Evaluator] Model successfully defended against jailbreak attempt #14.",
    "[Data Curator] Deduplicating the JSONL dataset cache.",
]

@router.get("/swarm")
def get_swarm_status() -> dict[str, Any]:
    return {"swarm": MOCK_AGENTS}

@router.get("/logs")
def get_swarm_logs(limit: int = 10) -> dict[str, Any]:
    # Generate mock logs with recent timestamps
    now = time.time()
    logs = []
    for i in range(limit):
        logs.append({
            "timestamp": now - (i * random.uniform(2, 10)),
            "message": random.choice(LOG_POOL),
            "level": "info",
        })
    # Sort chronologically
    logs.sort(key=lambda x: x["timestamp"])  # type: ignore
    return {"logs": logs}
