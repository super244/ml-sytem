from __future__ import annotations

from pathlib import Path

from evaluation.benchmark_registry import load_benchmark_registry, resolve_benchmark_file


def test_load_benchmark_registry_normalizes_entries(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmarks" / "math.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("[]\n")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks:\n  - id: math500\n    path: benchmarks/math.jsonl\n")

    registry = load_benchmark_registry(registry_path)

    assert registry[0]["id"] == "math500"
    assert registry[0]["resolved_path"].endswith("benchmarks/math.jsonl")
    assert registry[0]["exists"] is True


def test_resolve_benchmark_file_returns_metadata(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "custom" / "bench.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("[]\n")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("benchmarks: []\n")

    path, entry = resolve_benchmark_file(registry_path, benchmark_file="custom/bench.jsonl")

    assert path == "custom/bench.jsonl"
    assert entry["resolved_path"].endswith("custom/bench.jsonl")
    assert entry["exists"] is True
