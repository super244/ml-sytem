from pathlib import Path

from inference.app.workspace import build_workspace_overview


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_build_workspace_overview_discovers_profiles_and_commands(tmp_path: Path) -> None:
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
        tmp_path / "configs" / "finetune.yaml",
        (
            "instance:\n"
            "  type: finetune\n"
            "experience:\n"
            "  level: hobbyist\n"
            "orchestration_mode: hybrid\n"
            "subsystem:\n"
            "  config_ref: training/configs/profiles/baseline_qlora.yaml\n"
        ),
    )
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
    assert overview["summary"]["orchestration_templates"] == 1
    assert any(recipe["id"] == "refresh-lab" for recipe in overview["command_recipes"])
    assert any(recipe["command"] == "ai-factory refresh-lab" for recipe in overview["command_recipes"])
    assert any(recipe["command"] == "ai-factory doctor --json" for recipe in overview["command_recipes"])
    assert any(capability["id"] == "feedback-loop" for capability in overview["orchestration_capabilities"])
    assert overview["orchestration_templates"][0]["orchestration_mode"] == "hybrid"
    assert overview["summary"]["interfaces"] == 4
    assert any(surface["id"] == "desktop" for surface in overview["interfaces"])
    assert any(tier["id"] == "beginner" for tier in overview["experience_tiers"])
    assert any(extension["id"] == "qlora" for extension in overview["extension_points"])
    assert any(extension["id"] == "benchmark:benchmark_holdout" for extension in overview["extension_points"])


def test_build_workspace_overview_reports_invalid_catalog_asset(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "catalog.json",
        "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 10\n",
    )
    _write(tmp_path / "data" / "processed" / "pack_summary.json", '{"packs": []}')
    _write(tmp_path / "data" / "processed" / "manifest.json", '{"schema_version": "v2"}')
    _write(tmp_path / "evaluation" / "benchmarks" / "registry.yaml", "benchmarks: []\n")
    _write(tmp_path / "inference" / "configs" / "model_registry.yaml", "models: []\n")

    overview = build_workspace_overview(tmp_path)

    catalog_check = next(item for item in overview["readiness_checks"] if item["id"] == "dataset-catalog")
    assert catalog_check["ok"] is False
    assert overview["status"] == "degraded"
    assert any("git lfs pull" in error for error in overview["errors"])
