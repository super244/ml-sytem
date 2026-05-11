from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_factory.core.hashing import normalize_text, sha256_file, sha256_text, stable_question_fingerprint
from ai_factory.core.io import load_json, read_jsonl, write_json, write_jsonl, write_markdown
from ai_factory.core.reports import bullet_list, markdown_table, write_markdown_report
from ai_factory.core.tokens import approximate_token_count, estimate_generation_cost_usd

# ---------------------------------------------------------------------------
# io.py
# ---------------------------------------------------------------------------


def test_load_json_missing_file_returns_default(tmp_path: Path) -> None:
    result = load_json(tmp_path / "nonexistent.json", default={"fallback": True})
    assert result == {"fallback": True}


def test_load_json_missing_file_returns_none_by_default(tmp_path: Path) -> None:
    assert load_json(tmp_path / "nonexistent.json") is None


def test_write_and_load_json_roundtrip(tmp_path: Path) -> None:
    payload = {"key": "value", "num": 42}
    path = tmp_path / "sub" / "data.json"
    write_json(path, payload)
    assert path.exists()
    loaded = load_json(path)
    assert loaded == payload


def test_write_jsonl_and_read_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"b": 2}, {"c": 3}]
    path = tmp_path / "data.jsonl"
    write_jsonl(path, rows)
    loaded = read_jsonl(path)
    assert loaded == rows


def test_read_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    result = read_jsonl(tmp_path / "missing.jsonl")
    assert result == []


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_markdown(path, "# Hello\n\nWorld")
    assert path.read_text().startswith("# Hello")


def test_write_markdown_strips_trailing_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_markdown(path, "content   \n\n\n")
    assert path.read_text() == "content\n"


# ---------------------------------------------------------------------------
# hashing.py
# ---------------------------------------------------------------------------


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_text_none_returns_empty() -> None:
    assert normalize_text(None) == ""


def test_normalize_text_empty_returns_empty() -> None:
    assert normalize_text("") == ""


def test_stable_question_fingerprint_is_deterministic() -> None:
    fp1 = stable_question_fingerprint("What is 2+2?")
    fp2 = stable_question_fingerprint("What is 2+2?")
    assert fp1 == fp2
    assert len(fp1) == 16


def test_stable_question_fingerprint_case_insensitive() -> None:
    fp1 = stable_question_fingerprint("Hello World")
    fp2 = stable_question_fingerprint("hello world")
    assert fp1 == fp2


def test_sha256_text_returns_hex_string() -> None:
    result = sha256_text("hello")
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_sha256_file_matches_sha256_text(tmp_path: Path) -> None:
    content = "test content for hashing"
    path = tmp_path / "file.txt"
    path.write_text(content)
    file_hash = sha256_file(path)
    text_hash = sha256_text(content)
    assert file_hash == text_hash


# ---------------------------------------------------------------------------
# tokens.py
# ---------------------------------------------------------------------------


def test_approximate_token_count_empty_returns_zero() -> None:
    assert approximate_token_count(None) == 0
    assert approximate_token_count("") == 0


def test_approximate_token_count_estimates_from_words() -> None:
    count = approximate_token_count("hello world foo bar")
    assert count >= 1


def test_estimate_generation_cost_usd_returns_none_when_costs_missing() -> None:
    result = estimate_generation_cost_usd(100, 50, None, None)
    assert result is None


def test_estimate_generation_cost_usd_computes_correctly() -> None:
    result = estimate_generation_cost_usd(1_000_000, 1_000_000, 1.0, 2.0)
    assert result == pytest.approx(3.0)


def test_estimate_generation_cost_usd_zero_tokens() -> None:
    result = estimate_generation_cost_usd(0, 0, 1.0, 2.0)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reports.py
# ---------------------------------------------------------------------------


def test_markdown_table_generates_correct_format() -> None:
    table = markdown_table(["Name", "Score"], [["Alice", "95"], ["Bob", "87"]])
    assert "| Name | Score |" in table
    assert "| --- | --- |" in table
    assert "| Alice | 95 |" in table


def test_bullet_list_generates_correct_format() -> None:
    result = bullet_list(["item one", "item two", "item three"])
    assert result == "- item one\n- item two\n- item three"


def test_write_markdown_report_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_markdown_report(path, "Test Report", [("Section A", "Content A"), ("Section B", "Content B")])
    text = path.read_text()
    assert "# Test Report" in text
    assert "## Section A" in text
    assert "Content A" in text


# ---------------------------------------------------------------------------
# error_taxonomy.py
# ---------------------------------------------------------------------------


def test_cluster_failures_counts_error_types() -> None:
    from ai_factory.core.error_taxonomy import cluster_failures

    results = [
        {"primary": {"correct": False, "error_type": "arithmetic"}},
        {"primary": {"correct": False, "error_type": "arithmetic"}},
        {"primary": {"correct": False, "error_type": "logic"}},
        {"primary": {"correct": True, "error_type": "arithmetic"}},
    ]
    counts = cluster_failures(results, "primary")
    assert counts["arithmetic"] == 2
    assert counts["logic"] == 1


def test_cluster_failures_skips_correct() -> None:
    from ai_factory.core.error_taxonomy import cluster_failures

    results = [{"primary": {"correct": True, "error_type": "arithmetic"}}]
    counts = cluster_failures(results, "primary")
    assert counts == {}


def test_summarize_failure_taxonomy_returns_primary_and_secondary() -> None:
    from ai_factory.core.error_taxonomy import summarize_failure_taxonomy

    results = [
        {
            "primary": {"correct": False, "error_type": "arithmetic"},
            "secondary": {"correct": False, "error_type": "sign"},
        },
    ]
    summary = summarize_failure_taxonomy(results)
    assert "primary" in summary
    assert "secondary" in summary
    assert summary["primary"]["arithmetic"] == 1


# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------


def test_load_catalog_returns_empty_structure_when_missing(tmp_path: Path) -> None:
    from ai_factory.core.datasets import load_catalog

    result = load_catalog(tmp_path / "nonexistent.json")
    assert result["datasets"] == []
    assert result["generated_at"] is None


def test_inspect_json_asset_flags_git_lfs_pointer(tmp_path: Path) -> None:
    from ai_factory.core.datasets import inspect_json_asset

    pointer = tmp_path / "catalog.json"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:1234567890abcdef\nsize 42\n")

    status = inspect_json_asset(pointer)
    assert status["ok"] is False
    assert status["kind"] == "git_lfs_pointer"


def test_inspect_json_asset_flags_invalid_json(tmp_path: Path) -> None:
    from ai_factory.core.datasets import inspect_json_asset

    broken = tmp_path / "catalog.json"
    broken.write_text("{not json")

    status = inspect_json_asset(broken)
    assert status["ok"] is False
    assert status["kind"] == "invalid_json"


def test_load_pack_summary_returns_empty_when_missing(tmp_path: Path) -> None:
    from ai_factory.core.datasets import load_pack_summary

    result = load_pack_summary(tmp_path / "nonexistent.json")
    assert result == {"packs": []}


def test_list_catalog_entries_filters_by_kind(tmp_path: Path) -> None:
    from ai_factory.core.datasets import list_catalog_entries

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"id": "a", "kind": "synthetic"},
                    {"id": "b", "kind": "public"},
                    {"id": "c", "kind": "synthetic"},
                ]
            }
        )
    )
    entries = list_catalog_entries(kind="synthetic", path=catalog_path)
    assert len(entries) == 2
    assert all(e["kind"] == "synthetic" for e in entries)


def test_compute_record_stats_basic() -> None:
    from ai_factory.core.datasets import compute_record_stats

    records = [
        {"difficulty": "hard", "topic": "calculus", "source": "custom", "quality_score": 0.9},
        {"difficulty": "easy", "topic": "algebra", "source": "public", "quality_score": 0.7},
    ]
    stats = compute_record_stats(records)
    assert stats["num_records"] == 2
    assert stats["difficulty_counts"]["hard"] == 1
    assert stats["avg_quality_score"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# artifacts.py
# ---------------------------------------------------------------------------


def test_prepare_run_layout_creates_directories(tmp_path: Path) -> None:
    from ai_factory.core.artifacts import prepare_run_layout

    layout = prepare_run_layout(tmp_path, "test-run", explicit_run_id="test-run-001")
    assert layout.run_id == "test-run-001"
    assert layout.logs_dir.exists()
    assert layout.reports_dir.exists()
    assert layout.checkpoints_dir.exists()


def test_detect_run_env_returns_run_env() -> None:
    from ai_factory.core.artifacts import detect_run_env

    env = detect_run_env()
    assert hasattr(env, "python")
    assert hasattr(env, "platform")


def test_ensure_latest_pointer_writes_json(tmp_path: Path) -> None:
    from ai_factory.core.artifacts import ensure_latest_pointer

    pointer = tmp_path / "LATEST_RUN.json"
    target = tmp_path / "runs" / "my-run"
    target.mkdir(parents=True)
    ensure_latest_pointer(pointer, target, metadata={"run_id": "my-run"})
    data = json.loads(pointer.read_text())
    assert "target_dir" in data
    assert data["run_id"] == "my-run"


# ---------------------------------------------------------------------------
# benchmark_registry.py
# ---------------------------------------------------------------------------


def test_load_benchmark_registry_returns_list(tmp_path: Path) -> None:
    from ai_factory.core.discovery import load_benchmark_registry

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks:\n  - id: math500\n    path: data/math500.jsonl\n")
    result = load_benchmark_registry(registry_path)
    assert len(result) == 1
    assert result[0]["id"] == "math500"


def test_resolve_benchmark_file_by_id(tmp_path: Path) -> None:
    from ai_factory.core.discovery import resolve_benchmark_file

    benchmark_path = tmp_path / "data" / "math500.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("[]\n")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks:\n  - id: math500\n    path: data/math500.jsonl\n")
    path, entry = resolve_benchmark_file(registry_path, benchmark_id="math500")
    assert path == "data/math500.jsonl"
    assert entry["id"] == "math500"


def test_resolve_benchmark_file_by_path(tmp_path: Path) -> None:
    from ai_factory.core.discovery import resolve_benchmark_file

    benchmark_path = tmp_path / "custom" / "bench.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("[]\n")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks: []\n")
    path, entry = resolve_benchmark_file(registry_path, benchmark_file="custom/bench.jsonl")
    assert path == "custom/bench.jsonl"


def test_resolve_benchmark_file_rejects_missing_path(tmp_path: Path) -> None:
    from ai_factory.core.discovery import resolve_benchmark_file

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks: []\n")

    with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
        resolve_benchmark_file(registry_path, benchmark_file=str(tmp_path / "missing.jsonl"))


def test_resolve_benchmark_file_raises_for_unknown_id(tmp_path: Path) -> None:
    from ai_factory.core.discovery import resolve_benchmark_file

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks: []\n")
    with pytest.raises(KeyError):
        resolve_benchmark_file(registry_path, benchmark_id="nonexistent")


def test_resolve_benchmark_file_by_id_from_nested_registry_supports_repo_relative_path(tmp_path: Path) -> None:
    from ai_factory.core.discovery import resolve_benchmark_file

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\nversion = '0.0.0'\n")
    benchmark_path = tmp_path / "data" / "processed" / "eval.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("[]\n")
    registry_path = tmp_path / "evaluation" / "benchmarks" / "registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("benchmarks:\n  - id: core_eval\n    path: data/processed/eval.jsonl\n")

    path, entry = resolve_benchmark_file(registry_path, benchmark_id="core_eval")

    assert path == "data/processed/eval.jsonl"
    assert entry["id"] == "core_eval"
