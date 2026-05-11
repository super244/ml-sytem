"""AI-Factory quick-start script for local dry-run orchestration."""

from __future__ import annotations

from pathlib import Path

from ai_factory.core.platform.container import build_platform_container


def main() -> None:
    container = build_platform_container(repo_root=Path.cwd(), artifacts_dir=Path("artifacts"))
    control = container.control_service

    print("AI-Factory quick start")
    print("======================")

    created = control.create_instance("examples/orchestration/train.yaml", start=False)
    print(f"Created training instance: {created.id} ({created.name})")
    print(f"Status: {created.status}")
    print(f"Orchestration run: {created.orchestration_run_id}")

    runs = control.list_orchestration_runs()
    print(f"Runs tracked: {len(runs)}")
    print("Tip: run `ai-factory start <instance-id>` (or create with start=True) to execute.")


if __name__ == "__main__":
    main()
