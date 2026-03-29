from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def notebook_metadata(title: str) -> dict[str, Any]:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
        "atlas_title": title,
    }


def write_notebook(path: Path, metadata: dict[str, Any], cells: list[dict[str, Any]]) -> None:
    payload = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


COMMON_SETUP = """
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path.cwd()
if (ROOT / "notebooks").exists():
    REPO_ROOT = ROOT
else:
    REPO_ROOT = ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT
"""


def tutorial_intro(title: str, goals: list[str]) -> dict[str, Any]:
    return markdown_cell(
        f"""
# {title}

This notebook is part of the Atlas Math Lab research workbench.

Goals:
{chr(10).join(f"- {goal}" for goal in goals)}
"""
    )


NOTEBOOK_SPECS: dict[str, tuple[str, list[dict[str, Any]]]] = {
    "00_dataset_landscape.ipynb": (
        "Dataset Landscape",
        [
            tutorial_intro(
                "Dataset Landscape",
                [
                    "Inspect the catalog of synthetic and public dataset families.",
                    "Review dataset sizes, topics, and preview examples.",
                    "Understand how packs map onto training and benchmark slices.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from data.catalog import load_catalog, load_pack_summary

catalog = load_catalog(REPO_ROOT / "data" / "catalog.json")
pack_summary = load_pack_summary(REPO_ROOT / "data" / "processed" / "pack_summary.json")
catalog["summary"], pack_summary
"""
            ),
            code_cell(
                """
import pandas as pd

df = pd.DataFrame(catalog["datasets"])
df[["id", "kind", "family", "topic", "num_rows", "size_bytes"]].sort_values(["kind", "id"])
"""
            ),
        ],
    ),
    "01_calculus_generator_lab.ipynb": (
        "Calculus Generator Lab",
        [
            tutorial_intro(
                "Calculus Generator Lab",
                [
                    "Inspect synthesis configs and family-specific generators.",
                    "Preview a small generated dataset before materializing a full pack.",
                    "Understand the pedagogical focus of each synthetic family.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
import yaml
from data.synthesis import DatasetSpec, generate_records

config = yaml.safe_load((REPO_ROOT / "data" / "configs" / "generation.yaml").read_text())
spec = DatasetSpec(**config["dataset_specs"][0])
rows = generate_records(spec, target_size_bytes=24 * 1024, seed=123)
rows[:2]
"""
            ),
        ],
    ),
    "02_transformer_exploration.ipynb": (
        "Transformer Exploration",
        [
            tutorial_intro(
                "Transformer Exploration",
                [
                    "Inspect tokenizer behavior and chat formatting.",
                    "Validate training configs before a full run.",
                    "Measure parameter counts for LoRA/QLoRA profiles.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from training.src.config import load_experiment_config
from training.src.modeling import load_tokenizer

config = load_experiment_config(REPO_ROOT / "training" / "configs" / "profiles" / "failure_aware.yaml")
tokenizer = load_tokenizer(config)
tokenizer("Evaluate int_0^1 x e^(x^2) dx.", add_special_tokens=False)
"""
            ),
        ],
    ),
    "03_base_vs_finetuned_inference.ipynb": (
        "Base vs Finetuned Inference",
        [
            tutorial_intro(
                "Base vs Finetuned Inference",
                [
                    "Run local side-by-side generations for the same prompt.",
                    "Inspect reranking, candidate agreement, and verifier metadata.",
                    "Use the same generation engine exposed by the API.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from inference.app.config import get_settings
from inference.app.generation import GenerationParameters, MathGenerator
from inference.app.model_loader import MathModelRegistry, load_registry_from_yaml
from inference.app.prompts import load_prompt_presets

settings = get_settings()
generator = MathGenerator(
    MathModelRegistry(load_registry_from_yaml(settings.model_registry_path)),
    prompt_presets=load_prompt_presets(settings.prompt_library_path),
)

question = "Evaluate int_0^1 x * e^(x^2) dx."
generator.generate(GenerationParameters(question=question, model_variant="base", num_samples=1))
"""
            ),
        ],
    ),
    "04_evaluation_win_case_browser.ipynb": (
        "Evaluation Win Case Browser",
        [
            tutorial_intro(
                "Evaluation Win Case Browser",
                [
                    "Load evaluation outputs and filter for specialist win cases.",
                    "Inspect per-example failure taxonomy and verifier metadata.",
                    "Turn good and bad examples into follow-up training hypotheses.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from ai_factory.core.io import read_jsonl

    rows = read_jsonl(REPO_ROOT / "evaluation" / "results" / "latest" / "per_example.jsonl")
    win_cases = [
        row for row in rows
        if row.get("primary", {}).get("correct") and not row.get("secondary", {}).get("correct")
    ]
    len(win_cases), win_cases[:1]
"""
            ),
        ],
    ),
    "05_lora_experiment_board.ipynb": (
        "LoRA Experiment Board",
        [
            tutorial_intro(
                "LoRA Experiment Board",
                [
                    "Compare profile configs across adapters, runtimes, and datasets.",
                    "Summarize trainable ratios and optimization settings.",
                    "Use the board to decide which experiment family to run next.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
import pandas as pd
from pathlib import Path
import yaml

profile_dir = REPO_ROOT / "training" / "configs" / "profiles"
rows = []
for path in sorted(profile_dir.glob("*.yaml")):
    payload = yaml.safe_load(path.read_text())
    rows.append({"profile": path.stem, "run_name": payload["run_name"], "seed": payload["seed"]})
pd.DataFrame(rows)
"""
            ),
        ],
    ),
    "06_public_dataset_normalization.ipynb": (
        "Public Dataset Normalization",
        [
            tutorial_intro(
                "Public Dataset Normalization",
                [
                    "Inspect public registry entries and normalization rules.",
                    "Preview how public data maps into the canonical schema.",
                    "Audit lineage and filter logic before normalization.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from data.adapters.base import load_public_registry

registry = load_public_registry(REPO_ROOT / "data" / "public" / "registry.yaml")
[(entry.id, entry.path, entry.filters) for entry in registry]
"""
            ),
        ],
    ),
    "07_dataset_quality_audit.ipynb": (
        "Dataset Quality Audit",
        [
            tutorial_intro(
                "Dataset Quality Audit",
                [
                    "Audit quality scores, reasoning styles, and verification coverage.",
                    "Identify low-quality or contaminated slices before training.",
                    "Review per-source and per-topic distributions.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from ai_factory.core.io import read_jsonl
from data.quality.stats import compute_record_stats

rows = read_jsonl(REPO_ROOT / "data" / "processed" / "normalized_all.jsonl")
compute_record_stats(rows)
"""
            ),
        ],
    ),
    "08_prompt_optimization_lab.ipynb": (
        "Prompt Optimization Lab",
        [
            tutorial_intro(
                "Prompt Optimization Lab",
                [
                    "Compare preset styles and solver modes on a fixed prompt set.",
                    "Capture which prompt styles improve verification-friendly reasoning.",
                    "Prepare prompt studies before heavier model retraining.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from inference.app.prompts import load_prompt_presets

presets = load_prompt_presets(REPO_ROOT / "inference" / "configs" / "prompt_presets.yaml")
{key: preset.description for key, preset in presets.items()}
"""
            ),
        ],
    ),
    "09_reranking_self_consistency.ipynb": (
        "Reranking and Self Consistency",
        [
            tutorial_intro(
                "Reranking and Self Consistency",
                [
                    "Inspect candidate-level agreement and verification scores.",
                    "Analyze how reranking changes final answers.",
                    "Prototype alternative best-of-N heuristics.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from ai_factory.core.answers import candidate_agreement, choose_best_candidate

toy_candidates = [
    {"text": "Final Answer: 2", "final_answer": "2", "verification_score": 1.0},
    {"text": "Final Answer: 2", "final_answer": "2", "verification_score": 0.8},
    {"text": "Final Answer: 3", "final_answer": "3", "verification_score": 0.2},
]
candidate_agreement(toy_candidates), choose_best_candidate(toy_candidates)
"""
            ),
        ],
    ),
    "10_verifier_analysis.ipynb": (
        "Verifier Analysis",
        [
            tutorial_intro(
                "Verifier Analysis",
                [
                    "Inspect typed step checks and verification outcomes.",
                    "Review formatting failures, arithmetic slips, and no-answer cases.",
                    "Understand how verifier-aware prompting affects outputs.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from ai_factory.core.answers import verify_prediction

verify_prediction(
    prediction_text="We compute carefully. Final Answer: 3",
    reference_answer="3",
    step_checks=[{"kind": "substring", "value": "carefully"}],
)
"""
            ),
        ],
    ),
    "11_benchmark_slice_analysis.ipynb": (
        "Benchmark Slice Analysis",
        [
            tutorial_intro(
                "Benchmark Slice Analysis",
                [
                    "Review benchmark registries and slice metadata.",
                    "Compare topic and difficulty coverage across benchmark packs.",
                    "Prepare leaderboard-ready slice summaries.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from evaluation.benchmark_registry import load_benchmark_registry

load_benchmark_registry(REPO_ROOT / "evaluation" / "benchmarks" / "registry.yaml")
"""
            ),
        ],
    ),
    "12_error_driven_retraining.ipynb": (
        "Error Driven Retraining",
        [
            tutorial_intro(
                "Error Driven Retraining Workflow",
                [
                    "Mine evaluation failures into replay examples.",
                    "Audit the replay pack before incorporating it into training.",
                    "Close the loop from evaluation to data curation.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from ai_factory.core.io import read_jsonl
from data.quality.mining import select_failure_cases

eval_rows = read_jsonl(REPO_ROOT / "evaluation" / "results" / "latest" / "per_example.jsonl")
select_failure_cases(eval_rows, limit=5)
"""
            ),
        ],
    ),
    "13_run_artifact_explorer.ipynb": (
        "Run Artifact Explorer",
        [
            tutorial_intro(
                "Run Artifact Explorer",
                [
                    "Inspect run manifests, metrics, and packaged model outputs.",
                    "Compare training runs by profile name and metrics.",
                    "Understand the artifact layout under artifacts/runs and artifacts/models.",
                ],
            ),
            code_cell(COMMON_SETUP),
            code_cell(
                """
from training.src.comparison import load_run_summary
from pathlib import Path

run_dirs = sorted((REPO_ROOT / "artifacts" / "runs").glob("*")) if (REPO_ROOT / "artifacts" / "runs").exists() else []
load_run_summary(run_dirs[0]) if run_dirs else {"runs": []}
"""
            ),
        ],
    ),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, cells) in NOTEBOOK_SPECS.items():
        output_path = NOTEBOOK_DIR / filename
        write_notebook(output_path, notebook_metadata(title), cells)
        print(f"Updated {output_path}")


if __name__ == "__main__":
    main()
