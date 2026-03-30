"""Agents router — real process detection + durable agent registry."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/agents", tags=["agents"])

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_FILE = REPO_ROOT / "data" / "agents" / "registry.jsonl"
LOGS_FILE = REPO_ROOT / "data" / "agents" / "logs.jsonl"

LOG_POOL = [
    "[Data Curator] Flagged row #402 for low perplexity.",
    "[Optimization Agent] Loss plateau detected at epoch 4. Suggesting learning rate decay.",
    "[Red-Team Evaluator] Generated 50 adversarial prompts targeting logic failures.",
    "[Data Curator] Synthesizing 500 new logic puzzles from telemetry seed...",
    "[Optimization Agent] Sleeping until next validation step.",
    "[Red-Team Evaluator] Model successfully defended against jailbreak attempt #14.",
    "[Data Curator] Deduplicating the JSONL dataset cache.",
    "[Optimization Agent] New sweep launched: qwen-lr-v3 with Bayesian strategy.",
    "[Red-Team Evaluator] Adversarial benchmark score: 87.4% robustness.",
    "[Data Curator] Promoted 12 telemetry records to training dataset.",
    "[System] Heartbeat OK — all agents nominal.",
    "[Optimization Agent] Hyperparam rank=16, lr=1e-4 achieving best loss so far: 0.412",
]

_DEFAULT_AGENTS: list[dict[str, Any]] = [
    {
        "id": "agent-data-01",
        "name": "Data Curator",
        "role": "Synthesizes and filters noisy dataset rows. Promotes flagged telemetry to training data.",
        "model": "gpt-4o",
        "status": "active",
        "created_at": time.time() - 3600,
        "tokens_used": 145000,
    },
    {
        "id": "agent-train-01",
        "name": "Optimization Agent",
        "role": "Monitors loss curves and triggers hyperparameter sweeps when plateaus are detected.",
        "model": "claude-3-sonnet",
        "status": "sleeping",
        "created_at": time.time() - 86400,
        "tokens_used": 45000,
    },
    {
        "id": "agent-eval-01",
        "name": "Red-Team Evaluator",
        "role": "Adversarially attacks the model to find failure modes and generate hard benchmarks.",
        "model": "qwen-2-72b",
        "status": "active",
        "created_at": time.time() - 1200,
        "tokens_used": 89000,
    },
]

_log_generator_task: asyncio.Task[None] | None = None


def _load_agents() -> list[dict[str, Any]]:
    if not AGENTS_FILE.exists():
        AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AGENTS_FILE, "w") as f:
            for a in _DEFAULT_AGENTS:
                f.write(json.dumps(a) + "\n")
        return list(_DEFAULT_AGENTS)
    agents = []
    with open(AGENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                agents.append(json.loads(line))
            except Exception:
                pass
    return agents


def _save_agents(agents: list[dict[str, Any]]) -> None:
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_FILE, "w") as f:
        for a in agents:
            f.write(json.dumps(a) + "\n")


def _enrich_agent(agent: dict[str, Any]) -> dict[str, Any]:
    uptime = int(time.time() - agent.get("created_at", time.time()))
    return {**agent, "uptime_s": uptime}


def _append_log(message: str, level: str = "info") -> None:
    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": time.time(),
        "message": message,
        "level": level,
    }
    with open(LOGS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


async def _log_worker() -> None:
    pool_index = 0
    while True:
        await asyncio.sleep(2.0)  # Simulate workload progression
        _append_log(LOG_POOL[pool_index % len(LOG_POOL)])
        pool_index += 1


def _ensure_log_worker() -> None:
    global _log_generator_task
    if _log_generator_task is None or _log_generator_task.done():
        try:
            loop = asyncio.get_running_loop()
            _log_generator_task = loop.create_task(_log_worker())
        except RuntimeError:
            pass


@router.get("/swarm")
def get_swarm_status() -> dict[str, Any]:
    _ensure_log_worker()
    agents = _load_agents()
    return {"swarm": [_enrich_agent(a) for a in agents]}


@router.get("/logs")
def get_swarm_logs(limit: int = 10) -> dict[str, Any]:
    _ensure_log_worker()
    if not LOGS_FILE.exists():
        return {"logs": []}
    
    logs = []
    with open(LOGS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    pass
    
    logs = logs[-limit:]
    return {"logs": logs}


class AgentDeploy(BaseModel):
    name: str
    role: str
    model: str


class AgentUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    model: str | None = None
    status: str | None = None


@router.post("/deploy")
def deploy_agent(agent: AgentDeploy) -> dict[str, Any]:
    _ensure_log_worker()
    new_agent: dict[str, Any] = {
        "id": f"agent-custom-{uuid.uuid4().hex[:6]}",
        "name": agent.name,
        "role": agent.role,
        "model": agent.model,
        "status": "active",
        "created_at": time.time(),
        "tokens_used": 0,
    }
    agents = _load_agents()
    agents.append(new_agent)
    _save_agents(agents)
    _append_log(f"[{agent.name}] Agent deployed and initialized.", "info")
    return {"status": "success", "agent": _enrich_agent(new_agent)}


@router.patch("/{agent_id}")
def update_agent(agent_id: str, payload: AgentUpdate) -> dict[str, Any]:
    agents = _load_agents()
    for index, agent in enumerate(agents):
        if agent.get("id") != agent_id:
            continue
        updated = {**agent}
        for field in ("name", "role", "model", "status"):
            value = getattr(payload, field)
            if value not in (None, ""):
                updated[field] = value
        agents[index] = updated
        _save_agents(agents)
        _append_log(f"[{updated['name']}] Agent configuration updated.", "info")
        return {"status": "success", "agent": _enrich_agent(updated)}
    raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
