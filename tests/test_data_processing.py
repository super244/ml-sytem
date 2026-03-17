from pathlib import Path

from ai_factory.core.schemas import DatasetBuildInfo
from data.builders.corpus_builder import normalize_record
from data.builders.pack_registry import build_derived_packs


def test_normalize_record_accepts_legacy_step_strings():
    normalized = normalize_record(
        {
            "question": "Evaluate int_0^1 x dx.",
            "solution": "The antiderivative is x^2/2. Final Answer: 1/2",
            "step_checks": "x^2/2||1/2",
            "difficulty": "hard",
        },
        default_source="unit_test",
    )
    assert normalized is not None
    assert normalized["quality_score"] > 0
    assert len(normalized["step_checks"]) == 2


def test_build_derived_packs(tmp_path: Path):
    rows = [
        {
            "id": "a",
            "question": "Q",
            "solution": "S",
            "difficulty": "hard",
            "topic": "calculus",
            "source": "custom_derivative_mastery",
            "pack_id": "core_train_mix",
            "reasoning_style": "chain_of_thought",
            "step_checks": [{"kind": "substring", "value": "x"}],
            "dataset_split": "test",
        }
    ]
    summaries = build_derived_packs(rows, tmp_path, build=DatasetBuildInfo(build_id="unit-test"))
    ids = {item["id"] for item in summaries}
    assert "calculus_hard_pack" in ids
    assert "benchmark_holdout_pack" in ids
