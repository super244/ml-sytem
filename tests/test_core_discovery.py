import builtins
import importlib
import json
import sys
from pathlib import Path

from ai_factory.core.discovery import latest_training_run, list_training_runs


def _write_run_manifest(run_dir: Path, *, run_id: str, run_name: str, created_at: str) -> None:
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_name,
                "profile_name": "baseline",
                "created_at": created_at,
                "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            }
        )
    )


def test_list_training_runs_surfaces_created_at(tmp_path: Path):
    run_dir = tmp_path / "runs" / "baseline-run"
    _write_run_manifest(
        run_dir,
        run_id="baseline-20260324-120000",
        run_name="baseline",
        created_at="2026-03-24T12:00:00+00:00",
    )

    runs = list_training_runs(str(tmp_path))

    assert runs[0]["created_at"] == "2026-03-24T12:00:00+00:00"


def test_latest_training_run_prefers_manifest_created_at_over_output_dir(tmp_path: Path):
    _write_run_manifest(
        tmp_path / "runs" / "zeta-old",
        run_id="zeta-20260324-100000",
        run_name="zeta-old",
        created_at="2026-03-24T10:00:00+00:00",
    )
    _write_run_manifest(
        tmp_path / "runs" / "alpha-new",
        run_id="alpha-20260324-110000",
        run_name="alpha-new",
        created_at="2026-03-24T11:00:00+00:00",
    )

    latest = latest_training_run(list_training_runs(str(tmp_path)))

    assert latest is not None
    assert latest["run_name"] == "alpha-new"


def test_latest_training_run_falls_back_to_run_id_timestamp(tmp_path: Path):
    _write_run_manifest(
        tmp_path / "runs" / "zeta-newer-id",
        run_id="zeta-20260324-120000",
        run_name="zeta-newer-id",
        created_at="",
    )
    _write_run_manifest(
        tmp_path / "runs" / "alpha-older-id",
        run_id="alpha-20260324-110000",
        run_name="alpha-older-id",
        created_at="",
    )

    latest = latest_training_run(list_training_runs(str(tmp_path)))

    assert latest is not None
    assert latest["run_name"] == "zeta-newer-id"


def test_latest_training_run_parses_embedded_timestamp_in_run_id(tmp_path: Path):
    _write_run_manifest(
        tmp_path / "runs" / "zeta-release-20260324-120000-build-7",
        run_id="zeta-release-20260324-120000-build-7",
        run_name="zeta-release",
        created_at="",
    )
    _write_run_manifest(
        tmp_path / "runs" / "alpha-release-20260324-110000-build-9",
        run_id="alpha-release-20260324-110000-build-9",
        run_name="alpha-release",
        created_at="",
    )

    latest = latest_training_run(list_training_runs(str(tmp_path)))

    assert latest is not None
    assert latest["run_name"] == "zeta-release"


def test_list_training_runs_skips_invalid_manifests(tmp_path: Path):
    good_run = tmp_path / "runs" / "good-run"
    _write_run_manifest(
        good_run,
        run_id="good-20260324-120000",
        run_name="good-run",
        created_at="2026-03-24T12:00:00+00:00",
    )
    bad_run = tmp_path / "runs" / "bad-run"
    bad_manifest = bad_run / "manifests" / "run_manifest.json"
    bad_manifest.parent.mkdir(parents=True, exist_ok=True)
    bad_manifest.write_text("{not valid json")

    runs = list_training_runs(str(tmp_path))

    assert len(runs) == 1
    assert runs[0]["run_name"] == "good-run"


def test_importing_ai_factory_core_stays_light_without_sympy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sympy" or name.startswith("sympy."):
            raise ModuleNotFoundError("No module named 'sympy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("ai_factory.core", None)
    sys.modules.pop("ai_factory.core.answers", None)

    core_module = importlib.import_module("ai_factory.core")

    assert core_module.normalize_text(" a \n b ") == "a b"
