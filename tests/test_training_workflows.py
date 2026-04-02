from __future__ import annotations

from pathlib import Path

import yaml

from ai_factory.core.io import write_jsonl
from evaluation.generated_benchmark import create_temp_model_registry, generate_benchmark_records
from training.src import workflows


def test_normalize_huggingface_dataset_reference_accepts_url():
    payload = workflows.normalize_huggingface_dataset_reference(
        "https://huggingface.co/datasets/openai/gsm8k?split=test&revision=main"
    )

    assert payload["path"] == "openai/gsm8k"
    assert payload["split"] == "test"
    assert payload["revision"] == "main"


def test_discover_private_categories_from_custom_root(tmp_path: Path):
    write_jsonl(
        tmp_path / "custom_alpha.jsonl",
        [{"question": "Q1", "solution": "S1", "difficulty": "easy", "topic": "calculus"}],
    )
    write_jsonl(
        tmp_path / "custom_beta.jsonl",
        [{"question": "Q2", "solution": "S2", "difficulty": "hard", "topic": "algebra"}],
    )
    (tmp_path / "custom_beta.manifest.json").write_text("{}")

    categories = workflows.discover_private_categories(tmp_path)

    assert categories == ["custom_alpha", "custom_beta"]


def test_build_workflow_corpus_uses_private_categories(tmp_path: Path, monkeypatch):
    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    source_a = custom_root / "custom_alpha.jsonl"
    source_b = custom_root / "custom_beta.jsonl"
    write_jsonl(
        source_a,
        [
            {"id": f"a-{idx}", "question": f"QA{idx}", "solution": f"SA{idx}", "difficulty": "hard", "topic": "calculus"}
            for idx in range(8)
        ],
    )
    write_jsonl(
        source_b,
        [
            {"id": f"b-{idx}", "question": f"QB{idx}", "solution": f"SB{idx}", "difficulty": "easy", "topic": "algebra"}
            for idx in range(8)
        ],
    )
    monkeypatch.setattr(workflows, "REPO_ROOT", tmp_path)

    result = workflows.build_workflow_corpus(
        workflow_name="unit_supervised",
        run_name="unit-run",
        private_categories=["custom_alpha", "custom_beta"],
        custom_root=custom_root,
        seed=11,
        eval_ratio=0.2,
        test_ratio=0.1,
    )

    assert Path(result["train_file"]).exists()
    assert Path(result["eval_file"]).exists()
    assert Path(result["test_file"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert [item["id"] for item in result["source_specs"]] == ["custom_alpha", "custom_beta"]


def test_build_training_config_payload_adjusts_training_mode():
    qlora_payload = workflows.build_training_config_payload(
        workflow_name="supervised",
        run_name="qlora-run",
        base_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        train_file="/tmp/train.jsonl",
        eval_file="/tmp/eval.jsonl",
        test_file="/tmp/test.jsonl",
        pack_manifest="/tmp/manifest.json",
        method="qlora",
    )
    full_payload = workflows.build_training_config_payload(
        workflow_name="supervised",
        run_name="full-run",
        base_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        train_file="/tmp/train.jsonl",
        eval_file="/tmp/eval.jsonl",
        test_file="/tmp/test.jsonl",
        pack_manifest="/tmp/manifest.json",
        method="full",
    )

    assert qlora_payload["model"]["use_4bit"] is True
    assert qlora_payload["adapter"]["method"] == "qlora"
    assert full_payload["model"]["use_full_precision"] is True
    assert full_payload["model"]["use_4bit"] is False
    assert full_payload["adapter"]["method"] == "full"


def test_generate_benchmark_records_returns_requested_count():
    records = generate_benchmark_records(question_count=24, seed=5)

    assert len(records) == 24
    assert len({record["question"] for record in records}) == 24
    assert all(record["final_answer"] for record in records)


def test_create_temp_model_registry_writes_yaml(tmp_path: Path):
    registry_path = create_temp_model_registry(
        tmp_path / "registry.yaml",
        model_name="eval_model",
        base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        adapter_path="artifacts/models/demo/latest",
    )

    payload = yaml.safe_load(registry_path.read_text())

    assert payload["models"][0]["name"] == "eval_model"
    assert payload["models"][0]["adapter_path"] == "artifacts/models/demo/latest"
