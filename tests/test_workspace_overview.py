from pathlib import Path

from inference.app.workspace import build_workspace_overview


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_build_workspace_overview_discovers_profiles_and_commands(tmp_path: Path):
    _write(tmp_path / "data" / "catalog.json", '{"summary": {"num_datasets": 2}}')
    _write(
        tmp_path / "data" / "processed" / "pack_summary.json",
        '{"packs": [{"id": "verification_pack"}]}',
    )
    _write(tmp_path / "data" / "processed" / "manifest.json", '{"schema_version": "v2"}')
    _write(
        tmp_path / "evaluation" / "benchmarks" / "registry.yaml",
        (
            "benchmarks:\n"
            "  - id: benchmark_holdout\n"
            "    title: Holdout\n"
            "    path: data.jsonl\n"
            "    description: Test\n"
            "    tags: []\n"
        ),
    )
    _write(
        tmp_path / "training" / "configs" / "profiles" / "baseline_qlora.yaml",
        "name: baseline\n",
    )
    _write(tmp_path / "evaluation" / "configs" / "base_vs_finetuned.yaml", "models: {}\n")
    _write(
        tmp_path / "inference" / "configs" / "model_registry.yaml",
        "models:\n  - name: base\n    base_model: Qwen/Qwen2.5-Math-1.5B-Instruct\n",
    )

    overview = build_workspace_overview(tmp_path)

    assert overview["summary"]["datasets"] == 2
    assert overview["summary"]["packs"] == 1
    assert overview["summary"]["models"] == 1
    assert overview["summary"]["training_profiles"] == 1
    assert overview["summary"]["evaluation_configs"] == 1
    assert any(recipe["id"] == "refresh-lab" for recipe in overview["command_recipes"])
