import asyncio
import math
import random
import time
import hashlib
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import select
from ai_factory.database import async_session_factory
from ai_factory.models.job import TrainingJob
from ai_factory.models.dataset import Dataset
from ai_factory.models.model_registry import ModelCheckpoint
from ai_factory.models.cluster import ClusterNode
from ai_factory.models.agent import AgentRecord, AgentDecisionRecord
from ai_factory.models.automl import AutoMLSearch, AutoMLRun
from ai_factory.models.lineage import LineageNode, LineageEdge
from ai_factory.api.websocket import manager
from ai_factory.schemas.websocket import (
    GPUTelemetryMessage,
    JobUpdateMessage,
    JobCompleteMessage,
    JobFailedMessage,
    ClusterUpdateMessage,
    AgentDecisionMessage,
    LogLineMessage,
    AutoMLUpdateMessage,
)
from ai_factory.schemas.cluster import GPUMetric, NodeStatus


NODE_DEFINITIONS = [
    {
        "id": "node-1",
        "name": "mac-studio-01",
        "type": "apple_silicon",
        "status": "online",
        "gpus": [{"index": 0, "name": "M2 Ultra", "utilization": 82.0, "vram_used_gb": 18.0, "vram_total_gb": 24.0, "temperature_celsius": 71.0, "power_draw_watts": 95.0, "memory_bandwidth_gbps": 200.0}],
        "cpu_utilization": 45.0,
        "ram_used_gb": 28.0,
        "ram_total_gb": 64.0,
        "network_rx_mbps": 120.5,
        "network_tx_mbps": 45.2,
        "active_jobs": ["job-2845"],
        "cost_per_hour": None,
    },
    {
        "id": "node-2",
        "name": "linux-rig-01",
        "type": "local_gpu",
        "status": "online",
        "gpus": [
            {"index": 0, "name": "NVIDIA A100 80GB", "utilization": 97.0, "vram_used_gb": 76.0, "vram_total_gb": 80.0, "temperature_celsius": 83.0, "power_draw_watts": 350.0, "memory_bandwidth_gbps": 2039.0},
            {"index": 1, "name": "NVIDIA A100 80GB", "utilization": 91.0, "vram_used_gb": 74.0, "vram_total_gb": 80.0, "temperature_celsius": 79.0, "power_draw_watts": 320.0, "memory_bandwidth_gbps": 2039.0},
        ],
        "cpu_utilization": 72.0,
        "ram_used_gb": 48.0,
        "ram_total_gb": 128.0,
        "network_rx_mbps": 850.0,
        "network_tx_mbps": 230.0,
        "active_jobs": ["job-2847", "job-2846"],
        "cost_per_hour": None,
    },
    {
        "id": "node-3",
        "name": "ec2-p4d-01",
        "type": "ssh_remote",
        "status": "online",
        "gpus": [{"index": 0, "name": "NVIDIA A100 40GB", "utilization": 12.0, "vram_used_gb": 2.0, "vram_total_gb": 40.0, "temperature_celsius": 34.0, "power_draw_watts": 65.0, "memory_bandwidth_gbps": 1555.0}],
        "cpu_utilization": 8.0,
        "ram_used_gb": 12.0,
        "ram_total_gb": 96.0,
        "network_rx_mbps": 50.0,
        "network_tx_mbps": 15.0,
        "active_jobs": [],
        "cost_per_hour": 3.06,
    },
    {
        "id": "node-4",
        "name": "lambda-A100-02",
        "type": "ssh_remote",
        "status": "online",
        "gpus": [{"index": 0, "name": "NVIDIA A100 80GB", "utilization": 38.0, "vram_used_gb": 31.0, "vram_total_gb": 80.0, "temperature_celsius": 58.0, "power_draw_watts": 180.0, "memory_bandwidth_gbps": 2039.0}],
        "cpu_utilization": 22.0,
        "ram_used_gb": 18.0,
        "ram_total_gb": 64.0,
        "network_rx_mbps": 200.0,
        "network_tx_mbps": 80.0,
        "active_jobs": ["job-2844"],
        "cost_per_hour": 2.14,
    },
]

JOB_DEFINITIONS = [
    {"id": "job-2847", "name": "run-2847-bayesian-lr-search", "type": "lora_finetune", "status": "running", "base_model": "meta-llama/Meta-Llama-3.1-8B", "current_step": 4821, "total_steps": 10000, "current_loss": 0.3421, "node_id": "node-2", "gpu_utilization": [91.0, 88.0], "vram_used_gb": 74.0, "vram_total_gb": 80.0},
    {"id": "job-2846", "name": "run-2846-dpo-alignment", "type": "dpo", "status": "running", "base_model": "mistralai/Mistral-7B-v0.1", "current_step": 7200, "total_steps": 10000, "current_loss": 0.1893, "node_id": "node-2", "gpu_utilization": [85.0], "vram_used_gb": 22.0, "vram_total_gb": 24.0},
    {"id": "job-2845", "name": "run-2845-math-sft", "type": "lora_finetune", "status": "running", "base_model": "meta-llama/Meta-Llama-3.1-8B", "current_step": 9100, "total_steps": 10000, "current_loss": 0.2104, "node_id": "node-1", "gpu_utilization": [78.0, 82.0], "vram_used_gb": 71.0, "vram_total_gb": 80.0},
    {"id": "job-2844", "name": "run-2844-code-lora", "type": "lora_finetune", "status": "queued", "base_model": "codellama/CodeLlama-13b-hf", "current_step": 0, "total_steps": 15000, "current_loss": None, "node_id": "node-4", "gpu_utilization": [], "vram_used_gb": 0.0, "vram_total_gb": 80.0},
]

DATASET_DEFINITIONS = [
    {
        "id": "ds-math-v3",
        "name": "math-reasoning-v3",
        "domain": "math",
        "status": "ready",
        "sample_count": 50000,
        "quality_score_mean": 0.982,
        "quality_score_p10": 0.91,
        "quality_score_p90": 0.99,
        "size_mb": 245.6,
        "content_hash": hashlib.sha256(b"math-reasoning-v3").hexdigest(),
        "pipeline_config_hash": "cfg-" + hashlib.sha256(b"math-pipe").hexdigest()[:12],
        "git_sha": "a1b2c3d",
        "pack_summary_json": {
            "source_distribution": {"synthetic": 0.7, "curated": 0.3},
            "filter_pass_rates": {"perplexity": 0.92, "dedup": 0.88, "toxicity": 0.99},
            "quality_histogram": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.22, 0.17, 0.12],
            "dedup_removed": 4200,
            "toxicity_removed": 120,
            "total_before_filter": 58000,
            "total_after_filter": 50000,
        },
    },
    {
        "id": "ds-code-v2",
        "name": "code-instruct-v2",
        "domain": "code",
        "status": "ready",
        "sample_count": 35000,
        "quality_score_mean": 0.957,
        "quality_score_p10": 0.88,
        "quality_score_p90": 0.98,
        "size_mb": 189.3,
        "content_hash": hashlib.sha256(b"code-instruct-v2").hexdigest(),
        "pipeline_config_hash": "cfg-" + hashlib.sha256(b"code-pipe").hexdigest()[:12],
        "git_sha": "e5f6g7h",
        "pack_summary_json": {
            "source_distribution": {"synthetic": 0.6, "curated": 0.4},
            "filter_pass_rates": {"perplexity": 0.94, "dedup": 0.85, "toxicity": 0.99},
            "quality_histogram": [0.02, 0.03, 0.04, 0.06, 0.09, 0.14, 0.19, 0.20, 0.14, 0.09],
            "dedup_removed": 5200,
            "toxicity_removed": 80,
            "total_before_filter": 42000,
            "total_after_filter": 35000,
        },
    },
    {
        "id": "ds-reason-v4",
        "name": "general-reasoning-v4",
        "domain": "reasoning",
        "status": "ready",
        "sample_count": 120000,
        "quality_score_mean": 0.913,
        "quality_score_p10": 0.82,
        "quality_score_p90": 0.96,
        "size_mb": 612.8,
        "content_hash": hashlib.sha256(b"general-reasoning-v4").hexdigest(),
        "pipeline_config_hash": "cfg-" + hashlib.sha256(b"reason-pipe").hexdigest()[:12],
        "git_sha": "i9j0k1l",
        "pack_summary_json": {
            "source_distribution": {"synthetic": 0.5, "curated": 0.3, "augmented": 0.2},
            "filter_pass_rates": {"perplexity": 0.89, "dedup": 0.82, "toxicity": 0.98},
            "quality_histogram": [0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.17, 0.15, 0.10, 0.07],
            "dedup_removed": 18000,
            "toxicity_removed": 2400,
            "total_before_filter": 155000,
            "total_after_filter": 120000,
        },
    },
]

MODEL_DEFINITIONS = [
    {"id": "mdl-1", "name": "llama-3.1-8b-math-v1", "base_model": "meta-llama/Meta-Llama-3.1-8B", "training_type": "lora", "size_gb": 4.2, "eval_scores": {"mmlu": 61.2, "gsm8k": 78.4, "humaneval": 43.8}, "children_count": 2, "deployed": False, "dataset_hash": hashlib.sha256(b"math-reasoning-v3").hexdigest(), "parent_model_id": None},
    {"id": "mdl-2", "name": "mistral-7b-code-v1", "base_model": "mistralai/Mistral-7B-v0.1", "training_type": "lora", "size_gb": 3.8, "eval_scores": {"mmlu": 58.8, "humaneval": 52.1, "gsm8k": 71.2}, "children_count": 0, "deployed": False, "dataset_hash": hashlib.sha256(b"code-instruct-v2").hexdigest(), "parent_model_id": None},
    {"id": "mdl-3", "name": "llama-3.1-8b-general-v1", "base_model": "meta-llama/Meta-Llama-3.1-8B", "training_type": "full_finetune", "size_gb": 15.2, "eval_scores": {"mmlu": 63.4, "gsm8k": 69.1, "humaneval": 38.2}, "children_count": 1, "deployed": True, "dataset_hash": hashlib.sha256(b"general-reasoning-v4").hexdigest(), "parent_model_id": None},
    {"id": "mdl-4", "name": "codellama-13b-instruct-v1", "base_model": "codellama/CodeLlama-13b-hf", "training_type": "dpo", "size_gb": 7.1, "eval_scores": {"mmlu": 55.1, "humaneval": 67.3, "gsm8k": 44.8}, "children_count": 0, "deployed": False, "dataset_hash": hashlib.sha256(b"code-instruct-v2").hexdigest(), "parent_model_id": "mdl-2"},
]

AGENT_DEFINITIONS = [
    {"id": "evaluator-01", "type": "evaluator", "status": "active", "decisions_today": 8},
    {"id": "pruner-01", "type": "pruner", "status": "active", "decisions_today": 5},
    {"id": "promoter-01", "type": "promoter", "status": "active", "decisions_today": 2},
]

DECISION_SEED = [
    {"agent_id": "pruner-01", "agent_type": "pruner", "action": "pruned", "target_id": "run-2841", "reasoning": "Loss plateau detected at 2000 steps. Δloss < 0.001 for 500 steps. Killing to free resources.", "metrics_snapshot": {"loss": 0.4521, "loss_delta": 0.0003, "step": 2000}},
    {"agent_id": "evaluator-01", "agent_type": "evaluator", "action": "scored", "target_id": "run-2839", "reasoning": "MMLU: 61.2% · HumanEval: 43.8% · GSM8K: 78.4% → Composite: 0.71", "metrics_snapshot": {"mmlu": 61.2, "humaneval": 43.8, "gsm8k": 78.4, "composite": 0.71}},
    {"agent_id": "promoter-01", "agent_type": "promoter", "action": "promoted", "target_id": "run-2839", "reasoning": "Composite score 0.71 exceeds threshold 0.65. Promoting to DPO stage.", "metrics_snapshot": {"composite": 0.71, "threshold": 0.65}},
    {"agent_id": "evaluator-01", "agent_type": "evaluator", "action": "scored", "target_id": "run-2838", "reasoning": "MMLU: 54.1% · HumanEval: 31.2% · GSM8K: 62.8% → Composite: 0.52", "metrics_snapshot": {"mmlu": 54.1, "humaneval": 31.2, "gsm8k": 62.8, "composite": 0.52}},
    {"agent_id": "pruner-01", "agent_type": "pruner", "action": "pruned", "target_id": "run-2836", "reasoning": "Loss divergence detected. Current loss 2.41 exceeds baseline 0.89 by 170%.", "metrics_snapshot": {"loss": 2.41, "baseline": 0.89}},
    {"agent_id": "evaluator-01", "agent_type": "evaluator", "action": "scored", "target_id": "run-2835", "reasoning": "MMLU: 58.8% · HumanEval: 52.1% · GSM8K: 71.2% → Composite: 0.64", "metrics_snapshot": {"mmlu": 58.8, "humaneval": 52.1, "gsm8k": 71.2, "composite": 0.64}},
]

AUTOML_SEARCH_DEFINITION = {
    "id": "search-bayesian-01",
    "strategy": "bayesian",
    "status": "running",
    "search_space": {
        "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 5e-4},
        "lora_rank": {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
        "lora_alpha": {"type": "categorical", "choices": [8, 16, 32, 64]},
        "warmup_steps": {"type": "int_uniform", "low": 50, "high": 500},
        "batch_size": {"type": "categorical", "choices": [2, 4, 8]},
    },
    "best_loss": 0.2104,
    "best_config": {"learning_rate": 2.4e-4, "lora_rank": 16, "lora_alpha": 32, "warmup_steps": 100, "batch_size": 4},
}


def _generate_samples(domain: str, count: int = 50) -> list[dict]:
    samples = []
    sources = {"math": "GSM8K-synth", "code": "HumanEval-synth", "reasoning": "COT-synth"}
    contents = {
        "math": [
            "Solve: If a train travels at 60 mph for 2.5 hours, how far does it go?\nAnswer: 150 miles",
            "Calculate the integral of x^2 from 0 to 3.\nAnswer: 9",
            "What is the probability of rolling a sum of 7 with two dice?\nAnswer: 6/36 = 1/6",
        ],
        "code": [
            "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left+right)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid+1\n        else: right = mid-1\n    return -1",
        ],
        "reasoning": [
            "Step 1: Identify the premises.\nStep 2: Apply logical deduction.\nStep 3: Draw conclusion based on evidence.",
            "Let's think step by step. First, we need to understand the relationship between the variables.",
        ],
    }
    for i in range(count):
        samples.append({
            "id": f"sample-{domain}-{i}",
            "content": random.choice(contents.get(domain, contents["reasoning"])),
            "quality_score": round(random.uniform(0.7, 1.0), 3),
            "perplexity": round(random.uniform(5.0, 25.0), 2),
            "domain": domain,
            "source": sources.get(domain, "mixed-synth"),
        })
    return samples


def _generate_automl_runs(search_id: str, count: int = 47) -> list[dict]:
    runs = []
    for i in range(count):
        if i < 3:
            status = "promoted"
        elif i < 8:
            status = "running"
        elif i < 31:
            status = "completed"
        else:
            status = "pruned"

        lr = round(random.uniform(1e-5, 5e-4), 6)
        rank = random.choice([4, 8, 16, 32, 64])
        alpha = random.choice([8, 16, 32, 64])

        eval_loss = None
        composite_score = None
        training_minutes = None
        step_pruned = None
        prune_reason = None

        if status in ("completed", "promoted"):
            eval_loss = round(random.uniform(0.2, 0.4), 4)
            composite_score = round(random.uniform(0.5, 0.85), 3)
            training_minutes = round(random.uniform(10, 120), 1)
        elif status == "pruned":
            eval_loss = round(random.uniform(0.5, 1.2), 4)
            step_pruned = random.randint(500, 3000)
            prune_reason = random.choice(["Loss plateau", "Loss divergence", "ASHA halving"])
            training_minutes = round(random.uniform(5, 45), 1)
        elif status == "running":
            eval_loss = round(random.uniform(0.3, 0.6), 4)
            training_minutes = round(random.uniform(5, 30), 1)

        runs.append({
            "id": f"aml-{str(i+1).zfill(3)}",
            "search_id": search_id,
            "status": status,
            "hyperparams": {"lr": lr, "rank": rank, "alpha": alpha},
            "eval_loss": eval_loss,
            "composite_score": composite_score,
            "training_minutes": training_minutes,
            "step_pruned": step_pruned,
            "prune_reason": prune_reason,
            "job_id": f"job-aml-{i+1}" if status == "running" else None,
        })
    return runs


def _generate_lineage_data() -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "ln-base-llama", "type": "base_model", "label": "Meta-Llama-3.1-8B", "metadata_json": {"source": "huggingface", "params": "8B"}},
        {"id": "ln-base-mistral", "type": "base_model", "label": "Mistral-7B-v0.1", "metadata_json": {"source": "huggingface", "params": "7B"}},
        {"id": "ln-base-codellama", "type": "base_model", "label": "CodeLlama-13b-hf", "metadata_json": {"source": "huggingface", "params": "13B"}},
        {"id": "ln-ds-math", "type": "dataset", "label": "math-reasoning-v3", "metadata_json": {"samples": 50000, "domain": "math"}},
        {"id": "ln-ds-code", "type": "dataset", "label": "code-instruct-v2", "metadata_json": {"samples": 35000, "domain": "code"}},
        {"id": "ln-ds-reason", "type": "dataset", "label": "general-reasoning-v4", "metadata_json": {"samples": 120000, "domain": "reasoning"}},
        {"id": "ln-ckpt-mdl1", "type": "checkpoint", "label": "llama-3.1-8b-math-v1", "metadata_json": {"eval_mmlu": 61.2, "eval_gsm8k": 78.4}},
        {"id": "ln-ckpt-mdl2", "type": "checkpoint", "label": "mistral-7b-code-v1", "metadata_json": {"eval_mmlu": 58.8, "eval_humaneval": 52.1}},
        {"id": "ln-ckpt-mdl3", "type": "checkpoint", "label": "llama-3.1-8b-general-v1", "metadata_json": {"eval_mmlu": 63.4}},
        {"id": "ln-ckpt-mdl4", "type": "checkpoint", "label": "codellama-13b-instruct-v1", "metadata_json": {"eval_humaneval": 67.3}},
        {"id": "ln-deploy-mdl3", "type": "deployment", "label": "prod-general-v1", "metadata_json": {"endpoint": "/api/v1/inference"}},
    ]
    edges = [
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl1", "target_id": "ln-base-llama", "label": "derived from"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl1", "target_id": "ln-ds-math", "label": "trained on"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl2", "target_id": "ln-base-mistral", "label": "derived from"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl2", "target_id": "ln-ds-code", "label": "trained on"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl3", "target_id": "ln-base-llama", "label": "derived from"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl3", "target_id": "ln-ds-reason", "label": "trained on"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl4", "target_id": "ln-ckpt-mdl2", "label": "derived from"},
        {"id": str(uuid4()), "source_id": "ln-ckpt-mdl4", "target_id": "ln-ds-code", "label": "trained on"},
        {"id": str(uuid4()), "source_id": "ln-deploy-mdl3", "target_id": "ln-ckpt-mdl3", "label": "deployed from"},
    ]
    return nodes, edges


async def _table_has_rows(session, model_class) -> bool:
    result = await session.execute(select(model_class).limit(1))
    return result.scalar_one_or_none() is not None


async def seed_database():
    async with async_session_factory() as session:

        now = datetime.now(timezone.utc)

        if not await _table_has_rows(session, ClusterNode):
            for nd in NODE_DEFINITIONS:
                node = ClusterNode(
                    id=nd["id"],
                    name=nd["name"],
                    type=nd["type"],
                    status=nd["status"],
                    gpus_json=nd["gpus"],
                    cpu_utilization=nd["cpu_utilization"],
                    ram_used_gb=nd["ram_used_gb"],
                    ram_total_gb=nd["ram_total_gb"],
                    network_rx_mbps=nd["network_rx_mbps"],
                    network_tx_mbps=nd["network_tx_mbps"],
                    active_jobs=nd["active_jobs"],
                    cost_per_hour=nd["cost_per_hour"],
                    last_seen=now,
                )
                session.add(node)

        if not await _table_has_rows(session, TrainingJob):
            for jd in JOB_DEFINITIONS:
                loss_history = []
                step_history = []
                if jd["current_step"] > 0:
                    for s in range(0, jd["current_step"], max(jd["current_step"] // 200, 1)):
                        loss = 2.5 * math.exp(-s / 2000) + random.gauss(0, 0.02)
                        loss_history.append(round(max(loss, 0.05), 4))
                        step_history.append(s)
                logs_tail = [
                    {"id": f"log-{jd['id']}-{i}", "level": random.choice(["INFO", "METRIC", "INFO"]), "message": f"Training step {jd['current_step']}/{jd['total_steps']}", "source": "trainer", "timestamp": now.isoformat(), "job_id": jd["id"]}
                    for i in range(min(10, max(1, jd["current_step"] // 100)))
                ]
                job = TrainingJob(
                    id=jd["id"],
                    name=jd["name"],
                    type=jd["type"],
                    status=jd["status"],
                    base_model=jd["base_model"],
                    config_json={"learning_rate": 2.4e-4, "epochs": 3, "batch_size": 4, "lora_rank": 16, "lora_alpha": 32},
                    current_step=jd["current_step"],
                    total_steps=jd["total_steps"],
                    current_loss=jd["current_loss"],
                    node_id=jd["node_id"],
                    gpu_utilization=jd["gpu_utilization"],
                    vram_used_gb=jd["vram_used_gb"],
                    vram_total_gb=jd["vram_total_gb"],
                    loss_history=loss_history[-200:],
                    step_history=step_history[-200:],
                    logs_tail=logs_tail[-100:],
                    started_at=now - timedelta(hours=random.randint(1, 6)),
                    created_at=now - timedelta(hours=random.randint(1, 6)),
                )
                session.add(job)

        if not await _table_has_rows(session, Dataset):
            for dd in DATASET_DEFINITIONS:
                samples = _generate_samples(dd["domain"], 50)
                ds = Dataset(
                    id=dd["id"],
                    name=dd["name"],
                    domain=dd["domain"],
                    status=dd["status"],
                    sample_count=dd["sample_count"],
                    quality_score_mean=dd["quality_score_mean"],
                    quality_score_p10=dd["quality_score_p10"],
                    quality_score_p90=dd["quality_score_p90"],
                    size_mb=dd["size_mb"],
                    content_hash=dd["content_hash"],
                    pipeline_config_hash=dd["pipeline_config_hash"],
                    git_sha=dd["git_sha"],
                    pack_summary_json=dd["pack_summary_json"],
                    samples_json=samples,
                    created_at=now - timedelta(hours=random.randint(1, 72)),
                )
                session.add(ds)

        if not await _table_has_rows(session, ModelCheckpoint):
            for md in MODEL_DEFINITIONS:
                model = ModelCheckpoint(
                    id=md["id"],
                    name=md["name"],
                    base_model=md["base_model"],
                    training_type=md["training_type"],
                    size_gb=md["size_gb"],
                    eval_scores=md["eval_scores"],
                    children_count=md["children_count"],
                    deployed=md["deployed"],
                    dataset_hash=md["dataset_hash"],
                    parent_model_id=md["parent_model_id"],
                    created_at=now - timedelta(hours=random.randint(1, 48)),
                )
                session.add(model)

        if not await _table_has_rows(session, AgentRecord):
            for ad in AGENT_DEFINITIONS:
                agent = AgentRecord(
                    id=ad["id"],
                    type=ad["type"],
                    status=ad["status"],
                    decisions_today=ad["decisions_today"],
                    last_action_at=now - timedelta(minutes=random.randint(5, 60)),
                )
                session.add(agent)

        if not await _table_has_rows(session, AgentDecisionRecord):
            for i, dd in enumerate(DECISION_SEED):
                decision = AgentDecisionRecord(
                    id=str(uuid4()),
                    agent_id=dd["agent_id"],
                    agent_type=dd["agent_type"],
                    action=dd["action"],
                    target_id=dd["target_id"],
                    reasoning=dd["reasoning"],
                    metrics_snapshot=dd["metrics_snapshot"],
                    timestamp=now - timedelta(minutes=(i + 1) * 15),
                )
                session.add(decision)

        if not await _table_has_rows(session, AutoMLSearch):
            search = AutoMLSearch(
                id=AUTOML_SEARCH_DEFINITION["id"],
                strategy=AUTOML_SEARCH_DEFINITION["strategy"],
                status=AUTOML_SEARCH_DEFINITION["status"],
                search_space=AUTOML_SEARCH_DEFINITION["search_space"],
                best_loss=AUTOML_SEARCH_DEFINITION["best_loss"],
                best_config=AUTOML_SEARCH_DEFINITION["best_config"],
                created_at=now - timedelta(hours=12),
            )
            session.add(search)

        if not await _table_has_rows(session, AutoMLRun):
            for run_data in _generate_automl_runs(AUTOML_SEARCH_DEFINITION["id"]):
                run = AutoMLRun(
                    id=run_data["id"],
                    search_id=run_data["search_id"],
                    status=run_data["status"],
                    hyperparams=run_data["hyperparams"],
                    eval_loss=run_data["eval_loss"],
                    composite_score=run_data["composite_score"],
                    training_minutes=run_data["training_minutes"],
                    step_pruned=run_data["step_pruned"],
                    prune_reason=run_data["prune_reason"],
                    job_id=run_data["job_id"],
                )
                session.add(run)

        if not await _table_has_rows(session, LineageNode):
            lineage_nodes, lineage_edges = _generate_lineage_data()
            for ln in lineage_nodes:
                session.add(LineageNode(**ln))
            for le in lineage_edges:
                session.add(LineageEdge(**le))

        await session.commit()


async def _gpu_telemetry_loop():
    while True:
        try:
            async with async_session_factory() as session:
                result = await session.execute(select(ClusterNode))
                nodes = result.scalars().all()
                now_ts = time.time() * 1000

                for node in nodes:
                    gpus = node.gpus_json or []
                    updated_gpus = []
                    for g in gpus:
                        gpu = dict(g)
                        gpu["utilization"] = max(0, min(100, gpu["utilization"] + random.gauss(0, 3)))
                        gpu["vram_used_gb"] = max(0, min(gpu["vram_total_gb"], gpu["vram_used_gb"] + random.gauss(0, 0.5)))
                        gpu["temperature_celsius"] = max(30, min(95, gpu["temperature_celsius"] + random.gauss(0, 1.5)))
                        gpu["power_draw_watts"] = max(50, gpu["power_draw_watts"] + random.gauss(0, 10))
                        updated_gpus.append(gpu)

                    node.gpus_json = updated_gpus
                    node.cpu_utilization = max(0, min(100, node.cpu_utilization + random.gauss(0, 2)))
                    node.last_seen = datetime.now(timezone.utc)

                    gpu_metrics = [GPUMetric(**g) for g in updated_gpus]
                    msg = GPUTelemetryMessage(
                        node_id=node.id,
                        timestamp=now_ts,
                        gpus=gpu_metrics,
                    )
                    await manager.broadcast(msg)

                await session.commit()
        except Exception as e:
            print(f"GPU telemetry error: {e}")
        await asyncio.sleep(2)


async def _job_update_loop():
    while True:
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(TrainingJob).where(TrainingJob.status == "running")
                )
                jobs = result.scalars().all()
                now_ts = time.time() * 1000

                for job in jobs:
                    step_increment = random.randint(5, 20)
                    job.current_step = min(job.current_step + step_increment, job.total_steps)
                    progress = job.current_step / max(job.total_steps, 1)

                    new_loss = 2.5 * math.exp(-job.current_step / 2000) + random.gauss(0, 0.02)
                    new_loss = max(new_loss, 0.05)
                    job.current_loss = round(new_loss, 4)

                    loss_history = list(job.loss_history or [])
                    step_history = list(job.step_history or [])
                    loss_history.append(job.current_loss)
                    step_history.append(job.current_step)
                    job.loss_history = loss_history[-200:]
                    job.step_history = step_history[-200:]

                    loss_delta = None
                    if len(loss_history) >= 2:
                        loss_delta = round(loss_history[-1] - loss_history[-2], 5)

                    remaining = job.total_steps - job.current_step
                    eta_seconds = int(remaining * 0.5) if remaining > 0 else 0

                    if job.current_step >= job.total_steps:
                        job.status = "completed"
                        job.completed_at = datetime.now(timezone.utc)

                        complete_msg = JobCompleteMessage(
                            job_id=job.id,
                            timestamp=now_ts,
                            final_loss=job.current_loss or 0.1,
                            eval_scores={"mmlu": round(random.uniform(55, 70), 1), "gsm8k": round(random.uniform(60, 85), 1), "humaneval": round(random.uniform(35, 55), 1)},
                            checkpoint_path=f"/data/checkpoints/{job.id}/ckpt-{job.total_steps}",
                            model_id=f"mdl-{job.id}",
                        )
                        await manager.broadcast(complete_msg)
                    elif random.random() < 0.005:
                        job.status = "failed"
                        job.completed_at = datetime.now(timezone.utc)

                        fail_msg = JobFailedMessage(
                            job_id=job.id,
                            timestamp=now_ts,
                            error="CUDA OOM: out of memory on device 0",
                            step_failed_at=job.current_step,
                        )
                        await manager.broadcast(fail_msg)

                    msg = JobUpdateMessage(
                        job_id=job.id,
                        timestamp=now_ts,
                        progress=round(progress, 4),
                        current_step=job.current_step,
                        current_loss=job.current_loss,
                        loss_delta=loss_delta,
                        eta_seconds=eta_seconds,
                        status=job.status,
                    )
                    await manager.broadcast(msg)

                await session.commit()
        except Exception as e:
            print(f"Job update error: {e}")
        await asyncio.sleep(5)


async def _cluster_update_loop():
    while True:
        try:
            async with async_session_factory() as session:
                result = await session.execute(select(ClusterNode))
                nodes = result.scalars().all()
                now_ts = time.time() * 1000

                node_statuses = []
                for node in nodes:
                    gpus = [GPUMetric(**g) for g in (node.gpus_json or [])]
                    node_statuses.append(NodeStatus(
                        id=node.id,
                        name=node.name,
                        type=node.type,
                        status=node.status,
                        gpus=gpus,
                        cpu_utilization=node.cpu_utilization,
                        ram_used_gb=node.ram_used_gb,
                        ram_total_gb=node.ram_total_gb,
                        network_rx_mbps=node.network_rx_mbps,
                        network_tx_mbps=node.network_tx_mbps,
                        active_jobs=node.active_jobs or [],
                        cost_per_hour=node.cost_per_hour,
                        last_seen=node.last_seen,
                    ))

                msg = ClusterUpdateMessage(nodes=node_statuses, timestamp=now_ts)
                await manager.broadcast(msg)
        except Exception as e:
            print(f"Cluster update error: {e}")
        await asyncio.sleep(10)


async def _agent_decision_loop():
    tick = 0
    actions = ["scored", "pruned", "promoted", "flagged"]
    agent_types = [
        ("evaluator-01", "evaluator"),
        ("pruner-01", "pruner"),
        ("promoter-01", "promoter"),
    ]
    reasonings = [
        "Loss plateau detected. Δloss < 0.001 for 500 steps.",
        "MMLU: {mmlu:.1f}% · HumanEval: {he:.1f}% → Composite: {comp:.2f}",
        "Composite score {comp:.2f} exceeds threshold 0.65. Promoting.",
        "Loss divergence detected. Current loss exceeds baseline.",
        "Performance regression on GSM8K benchmark.",
    ]

    while True:
        try:
            tick += 1
            if tick % 3 == 0:
                agent_id, agent_type = random.choice(agent_types)
                action = random.choice(actions)
                mmlu = random.uniform(50, 70)
                he = random.uniform(30, 60)
                comp = random.uniform(0.45, 0.85)
                reasoning = random.choice(reasonings).format(mmlu=mmlu, he=he, comp=comp)

                msg = AgentDecisionMessage(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    action=action,
                    target_id=f"run-{random.randint(2830, 2850)}",
                    reasoning=reasoning,
                    timestamp=time.time() * 1000,
                )
                await manager.broadcast(msg)

                try:
                    async with async_session_factory() as session:
                        decision = AgentDecisionRecord(
                            id=str(uuid4()),
                            agent_id=agent_id,
                            agent_type=agent_type,
                            action=action,
                            target_id=msg.target_id,
                            reasoning=reasoning,
                            metrics_snapshot={"mmlu": round(mmlu, 1), "humaneval": round(he, 1), "composite": round(comp, 2)},
                            timestamp=datetime.now(timezone.utc),
                        )
                        session.add(decision)
                        await session.commit()
                except Exception:
                    pass

        except Exception as e:
            print(f"Agent decision error: {e}")
        await asyncio.sleep(15)


async def _log_line_loop():
    sources = ["trainer", "evaluator", "agent:pruner-01", "agent:evaluator-01", "cluster", "scheduler", "metrics"]
    levels = ["INFO", "INFO", "INFO", "METRIC", "WARN", "AGENT"]
    messages_templates = [
        "Training step {step}/{total} completed",
        "loss={loss:.4f} lr=2.4e-4 grad_norm={gn:.2f}",
        "Checkpoint saved: ckpt-{step}",
        "Analyzing loss curve for {target}...",
        "GPU temperature {temp}°C exceeds soft limit",
        "Eval checkpoint queued: {target}-ckpt-{step}",
        "Gradient accumulation adjusted: 4 → 8",
        "Resource allocation updated for node {node}",
    ]

    while True:
        try:
            step = random.randint(1000, 10000)
            msg = LogLineMessage(
                source=random.choice(sources),
                level=random.choice(levels),
                message=random.choice(messages_templates).format(
                    step=step,
                    total=10000,
                    loss=random.uniform(0.1, 0.5),
                    gn=random.uniform(0.5, 2.0),
                    target=f"run-{random.randint(2830, 2850)}",
                    temp=random.randint(70, 90),
                    node=random.choice(["linux-rig-01", "mac-studio-01", "ec2-p4d-01"]),
                ),
                job_id=random.choice(["job-2847", "job-2846", "job-2845", None]),
                timestamp=time.time() * 1000,
            )
            await manager.broadcast(msg)
        except Exception as e:
            print(f"Log line error: {e}")
        await asyncio.sleep(3)


async def _automl_update_loop():
    while True:
        try:
            now_ts = time.time() * 1000
            msg = AutoMLUpdateMessage(
                search_id="search-bayesian-01",
                run_id=f"aml-{random.randint(1, 47):03d}",
                status=random.choice(["running", "completed", "pruned"]),
                eval_loss=round(random.uniform(0.2, 0.8), 4),
                composite_score=round(random.uniform(0.4, 0.85), 3),
                timestamp=now_ts,
            )
            await manager.broadcast(msg)
        except Exception as e:
            print(f"AutoML update error: {e}")
        await asyncio.sleep(8)


async def start_simulation():
    await seed_database()
    asyncio.create_task(_gpu_telemetry_loop())
    asyncio.create_task(_job_update_loop())
    asyncio.create_task(_cluster_update_loop())
    asyncio.create_task(_agent_decision_loop())
    asyncio.create_task(_log_line_loop())
    asyncio.create_task(_automl_update_loop())
