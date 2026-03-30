import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ai_factory.core.discovery import (
    latest_training_run,
    list_training_runs,
    load_benchmark_registry,
)
from data.catalog import load_catalog, load_pack_summary

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None  # type: ignore[assignment]
    RICH_AVAILABLE = False


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_rich_table(title: str, columns: list[str], rows: list[list[str]], style: str = "cyan") -> None:
    if RICH_AVAILABLE:
        table = Table(title=title)
        for col in columns:
            table.add_column(col, style=style)
        for row in rows:
            table.add_row(*row)
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * len(title))
        header = " | ".join(columns)
        print(header)
        print("-" * len(header))
        for row in rows:
            print(" | ".join(row))


def run_step(label: str, command: list[str], cwd: Path | None = None) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"Running {label}...", total=None)
            subprocess.run(command, cwd=str(cwd or Path.cwd()), check=True)
        console.print(f"[green]✓[/green] {label}")
    else:
        print(f"[ai-factory] {label}: {rendered}")
        subprocess.run(command, cwd=str(cwd or Path.cwd()), check=True)


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def cmd_api_smoke(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    checks: dict[str, str] = {}
    details: dict[str, Any] = {}
    failed = False

    paths = ["/v1/health", "/v1/status", "/v1/models", "/v1/datasets", "/v1/benchmarks", "/v1/runs"]
    for path in paths:
        endpoint = f"{base_url}{path}"
        try:
            payload = _request_json("GET", endpoint)
            checks[path] = "ok"
            if path == "/v1/status":
                details["status_title"] = payload.get("title")
                details["status_version"] = payload.get("version")
            elif path == "/v1/models":
                details["num_models"] = len(payload.get("models", []))
            elif path == "/v1/datasets":
                details["num_datasets"] = payload.get("summary", {}).get("num_datasets")
            elif path == "/v1/benchmarks":
                details["num_benchmarks"] = len(payload.get("benchmarks", []))
            elif path == "/v1/runs":
                details["num_runs"] = len(payload.get("runs", []))
        except urllib.error.URLError as exc:
            checks[path] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:
            checks[path] = f"failed: {exc}"
            failed = True

    if not args.skip_verify:
        endpoint = f"{base_url}/v1/verify"
        try:
            payload = _request_json(
                "POST",
                endpoint,
                {
                    "reference_answer": "1/2",
                    "candidate_answer": "1/2",
                    "prediction_text": "Final Answer: 1/2",
                    "step_checks": [],
                },
            )
            checks["/v1/verify"] = "ok"
            details["verify_equivalent"] = payload.get("equivalent")
            details["verify_error_type"] = payload.get("error_type")
        except urllib.error.URLError as exc:
            checks["/v1/verify"] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:
            checks["/v1/verify"] = f"failed: {exc}"
            failed = True

    if args.include_generate:
        endpoint = f"{base_url}/v1/generate"
        try:
            payload = _request_json(
                "POST",
                endpoint,
                {
                    "question": "Evaluate \\int_0^1 x dx.",
                    "model_variant": "finetuned",
                    "compare_to_base": False,
                    "compare_to_model": None,
                    "prompt_preset": "atlas_rigorous",
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 256,
                    "show_reasoning": True,
                    "difficulty_target": "easy",
                    "num_samples": 1,
                    "use_calculator": True,
                    "solver_mode": "rigorous",
                    "output_format": "text",
                    "use_cache": False,
                },
            )
            checks["/v1/generate"] = "ok"
            details["generated_final_answer"] = payload.get("final_answer")
        except urllib.error.URLError as exc:
            checks["/v1/generate"] = f"failed: {exc.reason}"
            failed = True
        except Exception as exc:
            checks["/v1/generate"] = f"failed: {exc}"
            failed = True

    payload = {"base_url": base_url, "checks": checks, "details": details}
    if args.json:
        _emit_json(payload)
    else:
        rows = [[path, status] for path, status in checks.items()]
        _print_rich_table("API Smoke Check", ["Endpoint", "Status"], rows, style="green")
        detail_rows = [[k, str(v)] for k, v in details.items()]
        _print_rich_table("Details", ["Key", "Value"], detail_rows, style="blue")

    if failed:
        sys.exit(1)


def cmd_latest_run(args: argparse.Namespace) -> None:
    repo_root = Path.cwd()
    runs = list_training_runs(str(repo_root / "artifacts"))
    latest = latest_training_run(runs)
    if latest is None:
        if args.json:
            _emit_json({"status": "no_runs_found"})
        else:
            if RICH_AVAILABLE:
                console.print(Panel("[red]No training runs found in artifacts directory.[/red]"))
            else:
                print("No training runs found in artifacts directory.")
        return

    payload = {
        "run_name": latest.get("run_name"),
        "profile_name": latest.get("profile_name"),
        "base_model": latest.get("base_model"),
        "output_dir": latest.get("output_dir"),
        "trainable_ratio": latest.get("model_report", {}).get("trainable_ratio"),
        "eval_loss": latest.get("metrics", {}).get("eval_loss"),
    }
    if args.json:
        _emit_json(payload)
    else:
        rows = [[str(k), str(v)] for k, v in payload.items()]
        _print_rich_table("Latest Run Summary", ["Metric", "Value"], rows, style="magenta")


def cmd_refresh_lab(args: argparse.Namespace) -> None:
    python = sys.executable
    if not args.skip_generate:
        run_step(
            "Generate synthetic datasets",
            [python, "data/generator/generate_calculus_datasets.py", "--config", "data/configs/generation.yaml"],
        )
    run_step(
        "Prepare processed corpus",
        [python, "data/prepare_dataset.py", "--config", "data/configs/processing.yaml"],
    )
    run_step(
        "Validate processed corpus",
        [
            python,
            "data/tools/validate_dataset.py",
            "--input",
            "data/processed/train.jsonl",
            "--manifest",
            "data/processed/manifest.json",
        ],
    )
    if not args.skip_notebooks:
        run_step("Refresh notebook lab", [python, "notebooks/build_notebooks.py"])
    if not args.skip_train_dry_run:
        run_step("Training dry-run", [python, "-m", "training.train", "--config", args.profile, "--dry-run"])
    if not args.skip_tests:
        run_step("Pytest suite", [python, "-m", "pytest"])

    if RICH_AVAILABLE:
        console.print(Panel("[bold green]Workspace refresh complete.[/bold green]"))
    else:
        print("[ai-factory] Workspace refresh complete.")


def cmd_doctor(args: argparse.Namespace) -> None:
    root = Path(getattr(args, "root", None) or Path.cwd())
    catalog = load_catalog(root / "data" / "catalog.json")
    packs = load_pack_summary(root / "data" / "processed" / "pack_summary.json").get("packs", [])
    runs = list_training_runs(str(root / "artifacts"))
    benchmarks = load_benchmark_registry(root / "evaluation" / "benchmarks" / "registry.yaml")
    frontend_ready = (root / "frontend" / "node_modules").exists()

    recommended_next_steps = [
        "python scripts/refresh_lab.py",
        "uvicorn inference.app.main:app --reload",
        "python scripts/api_smoke.py",
    ]
    if runs:
        recommended_next_steps.append("python scripts/latest_run.py")
    else:
        recommended_next_steps.append(
            "python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run"
        )
    if not frontend_ready:
        recommended_next_steps.append("cd frontend && npm install")

    packages = {
        "yaml": has_package("yaml"),
        "pytest": has_package("pytest"),
        "torch": has_package("torch"),
        "transformers": has_package("transformers"),
        "datasets": has_package("datasets"),
        "sympy": has_package("sympy"),
    }

    payload: dict[str, Any] = {
        "repo_root": str(root),
        "python_packages": packages,
        "data": {
            "catalog_present": (root / "data" / "catalog.json").exists(),
            "processed_manifest_present": (root / "data" / "processed" / "manifest.json").exists(),
            "num_catalog_datasets": catalog.get("summary", {}).get("num_datasets", 0),
            "num_derived_packs": len(packs),
        },
        "artifacts": {
            "num_runs": len(runs),
            "latest_run": (latest_training_run(runs) or {}).get("run_name"),
        },
        "evaluation": {
            "num_benchmarks": len(benchmarks),
        },
        "frontend": {
            "package_json_present": (root / "frontend" / "package.json").exists(),
            "node_modules_present": frontend_ready,
        },
        "recommended_next_steps": recommended_next_steps,
    }

    if args.json:
        from ai_factory.cli import _render_ready_summary
        from inference.app.workspace import build_workspace_overview
        workspace_payload = build_workspace_overview(root)
        if args.json:
             _emit_json({**payload, **workspace_payload})
        else:
             _emit_json(payload)
        return

    from ai_factory.cli import _print_section, _render_ready_summary
    from inference.app.workspace import build_workspace_overview
    workspace_payload = build_workspace_overview(root)

    if RICH_AVAILABLE:
        # First display the workspace readiness summary cleanly
        readiness = workspace_payload.get("readiness_checks") or []
        ready = sum(1 for c in readiness if c.get("ok"))
        total = len(readiness)
        
        console.print(Panel(f"[bold cyan]Workspace readiness: {ready}/{total}[/bold cyan]"))
        table = Table(title="Readiness Checks")
        table.add_column("Status", style="bold")
        table.add_column("Check")
        table.add_column("Detail")
        for check in readiness:
            ok = check.get("ok")
            mark = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
            table.add_row(mark, check.get("label", check.get("id", "?")), check.get("detail", ""))
        console.print(table)
        
        # Display the rest of the doctor information
        pkg_rows = [[pkg, "[green]Yes[/green]" if ok else "[red]No[/red]"] for pkg, ok in packages.items()]
        _print_rich_table("Python Packages", ["Package", "Installed"], pkg_rows)

        data_rows = [
            ["Catalog Present", str(payload["data"]["catalog_present"])],
            ["Manifest Present", str(payload["data"]["processed_manifest_present"])],
            ["Catalog Datasets", str(payload["data"]["num_catalog_datasets"])],
            ["Derived Packs", str(payload["data"]["num_derived_packs"])],
        ]
        _print_rich_table("Data Assets", ["Asset", "Status/Count"], data_rows)

        artifact_rows = [
            ["Total Runs", str(payload["artifacts"]["num_runs"])],
            ["Latest Run", str(payload["artifacts"]["latest_run"])],
            ["Total Benchmarks", str(payload["evaluation"]["num_benchmarks"])],
        ]
        _print_rich_table("Artifacts & Eval", ["Metric", "Value"], artifact_rows)

        steps_text = "\n".join(f"- {step}" for step in recommended_next_steps)
        console.print(Panel(steps_text, title="[bold magenta]Recommended Next Steps[/bold magenta]"))
    else:
        _render_ready_summary(workspace_payload)
        _print_section("Quick recipes")
        for recipe in (workspace_payload.get("command_recipes") or [])[:5]:
            print(f"  {recipe.get('title', '?')}: {recipe.get('command', '')}")
        print("\nPython Packages:")
        for k, v in packages.items():
            print(f"  {k}: {v}")
        print("\nData:")
        for k, v in payload["data"].items():
            print(f"  {k}: {v}")
        print("\nArtifacts & Eval:")
        print(f"  Runs: {payload['artifacts']['num_runs']}")
        print(f"  Latest: {payload['artifacts']['latest_run']}")
        print(f"  Benchmarks: {payload['evaluation']['num_benchmarks']}")
        print("\nRecommended Next Steps:")
        for step in recommended_next_steps:
            print(f"  - {step}")
