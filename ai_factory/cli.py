from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ai_factory.core.config.loader import load_orchestration_config, save_cloud_profile
from ai_factory.core.control.service import FactoryControlService
from ai_factory.core.instances.models import (
    ArchitectureSpec,
    EnvironmentSpec,
    EvaluationSuiteSpec,
    InstanceManifest,
    LifecycleProfile,
    PortForward,
)
from ai_factory.core.platform.container import build_platform_container
from ai_factory.titan import detect_titan_status, write_hardware_markdown

_STATUS_ICONS = {
    "running": "*",
    "completed": "+",
    "failed": "!",
    "pending": "~",
}


def _build_control_service(args: argparse.Namespace) -> FactoryControlService:
    return build_platform_container(
        repo_root=args.repo_root,
        artifacts_dir=args.artifacts_dir,
    ).control_service


def _render_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if isinstance(payload, list):
        for item in payload:
            print(item)
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for nested_key, nested_value in value.items():
                    print(f"  - {nested_key}: {nested_value}")
            else:
                print(f"{key}: {value}")
        return
    print(payload)


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"
    if value is None:
        return "n/a"
    return str(value)


def _format_progress(manifest: InstanceManifest) -> str:
    progress = manifest.progress
    if progress is None:
        return "n/a"
    if isinstance(progress.percent, (int, float)):
        return f"{progress.stage} {progress.percent * 100:.0f}%"
    return progress.stage


def _format_instance_row(manifest: InstanceManifest) -> str:
    stage = manifest.lifecycle.stage or (manifest.progress.stage if manifest.progress else None)
    updated = manifest.updated_at[:19].replace("T", " ") if manifest.updated_at else "n/a"
    icon = _STATUS_ICONS.get(manifest.status, " ")
    return (
        f"  {icon} {manifest.id:<18} {manifest.type:<10} {manifest.status:<10} env={manifest.environment.kind:<5} "
        f"stage={stage or 'n/a':<10} progress={_format_progress(manifest):<14} updated={updated} name={manifest.name}"
    )


def _render_instance_report(manifest: InstanceManifest) -> None:
    _print_section(f"Instance {manifest.id}")
    print(f"name: {manifest.name}")
    print(f"type/status: {manifest.type} / {manifest.status}")
    print(f"environment: {manifest.environment.kind}")
    if manifest.environment.kind == "cloud":
        endpoint = manifest.environment.host or "profile-managed"
        identity = manifest.environment.user or "n/a"
        print(f"cloud endpoint: {identity}@{endpoint}:{manifest.environment.port}")
        if manifest.environment.remote_repo_root:
            print(f"remote repo: {manifest.environment.remote_repo_root}")
    print(f"lifecycle stage: {manifest.lifecycle.stage or 'n/a'}")
    print(f"origin: {manifest.lifecycle.origin or 'n/a'}")
    print(f"learning mode: {manifest.lifecycle.learning_mode or 'n/a'}")
    if manifest.lifecycle.source_model:
        print(f"source model: {manifest.lifecycle.source_model}")
    if manifest.progress:
        print(f"progress: {_format_progress(manifest)}")
        if manifest.progress.status_message:
            print(f"progress note: {manifest.progress.status_message}")
    print(f"updated: {manifest.updated_at}")
    print(f"orchestration run: {manifest.orchestration_run_id or 'n/a'}")
    if manifest.decision is not None:
        print(f"decision: {manifest.decision.action} ({manifest.decision.rule})")
        if manifest.decision.explanation:
            print(f"decision note: {manifest.decision.explanation}")
    if manifest.metrics_summary:
        _print_section("Metrics")
        for key in sorted(manifest.metrics_summary):
            print(f"  {key}: {_format_metric_value(manifest.metrics_summary[key])}")
    if manifest.task_summary:
        _print_section("Task summary")
        for key in sorted(manifest.task_summary):
            print(f"  {key}: {_format_metric_value(manifest.task_summary[key])}")
    if manifest.recommendations:
        _print_section("Recommendations")
        for recommendation in manifest.recommendations[:5]:
            print(f"  - {recommendation.action} | priority={recommendation.priority} | {recommendation.reason}")
    if manifest.error is not None:
        _print_section("Error")
        print(f"  {manifest.error.code}: {manifest.error.message}")

    _print_section("Next steps")
    _print_next_steps(manifest)


def _print_next_steps(manifest: InstanceManifest) -> None:
    iid = manifest.id
    if manifest.status == "failed":
        print(f"  retry:    ai-factory retry {iid}")
        print(f"  logs:     ai-factory logs {iid} --follow")
        return
    if manifest.status == "running":
        print(f"  watch:    ai-factory watch {iid}")
        print(f"  logs:     ai-factory logs {iid} --follow")
        return
    if manifest.status == "completed":
        if manifest.type in {"train", "finetune"}:
            print(f"  evaluate: ai-factory evaluate {iid}")
            print(f"  infer:    ai-factory inference {iid}")
            print(f"  deploy:   ai-factory deploy {iid} --target ollama")
        elif manifest.type == "evaluate":
            if manifest.decision:
                action = manifest.decision.action
                if action == "deploy":
                    print(f"  deploy:   ai-factory deploy {iid} --target ollama")
                elif action in {"finetune", "retrain"}:
                    print(f"  iterate:  ai-factory action {iid} {action}")
                elif action == "open_inference":
                    print(f"  infer:    ai-factory inference {iid}")
            print(f"  compare:  ai-factory compare {iid}")
        elif manifest.type == "deploy":
            print(f"  status:   ai-factory status {iid}")
        elif manifest.type == "inference":
            print(f"  status:   ai-factory status {iid}")
        if manifest.recommendations:
            rec = manifest.recommendations[0]
            print(f"  top rec:  ai-factory action {iid} {rec.action}")
        return
    print(f"  status:   ai-factory status {iid}")


def _render_workspace_overview(payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    readiness = payload.get("readiness_checks") or []
    recipes = payload.get("command_recipes") or []
    templates = payload.get("orchestration_templates") or []
    training_profiles = payload.get("training_profiles") or []
    evaluation_configs = payload.get("evaluation_configs") or []
    interfaces = payload.get("interfaces") or []
    capabilities = payload.get("orchestration_capabilities") or []

    _print_section("Workspace overview")
    print(f"repo root: {payload.get('repo_root', 'n/a')}")
    print(
        "summary: "
        + ", ".join(
            f"{label}={summary.get(label, 0)}"
            for label in (
                "datasets",
                "packs",
                "models",
                "benchmarks",
                "runs",
                "training_profiles",
                "evaluation_configs",
                "orchestration_templates",
            )
        )
    )
    ready = summary.get("ready_checks", 0)
    total = summary.get("total_checks", 0)
    print(f"readiness: {ready}/{total} checks ready")
    if readiness:
        _print_section("Readiness checks")
        for check in readiness:
            state = "ok" if check.get("ok") else "needs setup"
            print(f"  [{state:>11}] {check.get('label', check.get('id', 'check'))}: {check.get('detail', '')}")
    if interfaces:
        _print_section("Interfaces")
        for surface in interfaces:
            print(
                f"  - {surface.get('label', surface.get('id', 'surface'))}: "
                f"{surface.get('entrypoint', 'n/a')} ({surface.get('status', 'unknown')})"
            )
    if capabilities:
        _print_section("Capabilities")
        for capability in capabilities[:6]:
            print(f"  - {capability.get('title', capability.get('id', 'capability'))}: {capability.get('detail', '')}")
    if recipes:
        _print_section("Top recipes")
        for recipe in recipes[:5]:
            print(f"  - {recipe.get('title', recipe.get('id', 'recipe'))}: {recipe.get('command', '')}")
    if templates:
        _print_section("Managed templates")
        for template in templates[:4]:
            print(
                f"  - {template.get('title', template.get('id', 'template'))} "
                f"[{template.get('instance_type', 'n/a')}] {template.get('path', 'n/a')}"
            )
    if training_profiles:
        _print_section("Training profiles")
        for profile in training_profiles[:4]:
            print(f"  - {profile.get('title', profile.get('id', 'profile'))}: {profile.get('train_command', '')}")
    if evaluation_configs:
        _print_section("Evaluation configs")
        for config in evaluation_configs[:4]:
            print(f"  - {config.get('title', config.get('id', 'config'))}: {config.get('run_command', '')}")


def _render_ready_summary(payload: dict[str, Any]) -> None:
    readiness = payload.get("readiness_checks") or []
    ready = sum(1 for c in readiness if c.get("ok"))
    total = len(readiness)
    print(f"Workspace readiness: {ready}/{total}")
    for check in readiness:
        ok = check.get("ok")
        mark = "OK" if ok else "MISSING"
        print(f"  [{mark:>7}] {check.get('label', check.get('id', '?'))}")
        if not ok:
            print(f"           {check.get('detail', '')}")
    if ready == total:
        print("\nAll checks passed. Workspace is ready.")
    else:
        print(f"\n{total - ready} check(s) need attention before full operation.")


def _render_titan_status(payload: dict[str, Any]) -> None:
    print(f"Hardware: {payload['silicon']}")
    print(f"Mode: {payload['mode']}")
    print(f"Backend: {payload['backend']}")
    print(f"Bandwidth: {payload.get('bandwidth_gbps') or 'n/a'} GB/s")
    print(f"GPU: {payload.get('gpu_name') or 'none'}")
    print(f"Unified memory: {payload.get('unified_memory_gb') or 'n/a'} GB")
    print(f"CPU fallback threads: {payload['cpu_fallback_threads']}")
    print(f"Silent mode cap: {payload['gpu_cap_pct']}%")


def _render_compare_summary(left: InstanceManifest, right: InstanceManifest) -> None:
    _print_section("Run comparison")
    print(f"  left:  {left.id} ({left.name}) [{left.status}]")
    print(f"  right: {right.id} ({right.name}) [{right.status}]")

    all_keys = sorted(set(left.metrics_summary) | set(right.metrics_summary))
    if not all_keys:
        print("\n  No metrics available to compare.")
        return

    _print_section("Metric deltas")
    print(f"  {'metric':<30} {'left':>12} {'right':>12} {'delta':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12}")
    for key in all_keys:
        lv = left.metrics_summary.get(key)
        rv = right.metrics_summary.get(key)
        lf = _format_metric_value(lv)
        rf = _format_metric_value(rv)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            delta = rv - lv
            df = f"{delta:+.4f}"
        else:
            df = "n/a"
        print(f"  {key:<30} {lf:>12} {rf:>12} {df:>12}")


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    response = input(f"{text}{suffix}: ").strip()
    return response or (default or "")


def _interactive_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "interactive", False) and sys.stdin.isatty())


def _new_config_path(args: argparse.Namespace) -> str | None:
    config = args.config
    return config or "configs/finetune.yaml"


def _deploy_target_from_config(config_path: str) -> str | None:
    try:
        config = load_orchestration_config(config_path)
        return config.subsystem.provider
    except Exception:
        return None


def _resolve_deploy_target(args: argparse.Namespace) -> str:
    if args.target:
        target = args.target
        return target
    config_target = _deploy_target_from_config(args.config)
    if config_target:
        return config_target
    raise SystemExit("ai-factory deploy requires --target or a subsystem.provider in the deploy config")


def _environment_from_args(args: argparse.Namespace) -> EnvironmentSpec | None:
    kind = args.environment
    if kind is None and args.command == "new":
        if _interactive_enabled(args):
            kind = _prompt("Environment (local/cloud)", "local")
        else:
            return None
    if not kind:
        return None
    if kind == "local":
        return EnvironmentSpec(kind="local")

    interactive = _interactive_enabled(args)
    host = args.cloud_host or (_prompt("Cloud host") if interactive else None)
    user = args.cloud_user or (_prompt("Cloud user") if interactive else None)
    key_path = args.cloud_key_path or (_prompt("SSH key path", "") if interactive else "")
    remote_repo_root = args.remote_repo_root or (
        _prompt("Remote repo root", "/tmp/ai-factory") if interactive else None
    )
    python_bin = args.python_bin or (_prompt("Remote python", "python3") if interactive else "python3")
    port_forwards = []
    for raw in getattr(args, "port_forwards", []) or []:
        local_port, remote_port, bind_host = _parse_port_forward(raw)
        port_forwards.append(PortForward(local_port=local_port, remote_port=remote_port, bind_host=bind_host))
    environment = EnvironmentSpec(
        kind="cloud",
        profile_name=args.cloud_profile,
        host=host,
        user=user,
        port=args.cloud_port,
        key_path=key_path or None,
        remote_repo_root=remote_repo_root,
        python_bin=python_bin,
        port_forwards=port_forwards,
    )
    if args.cloud_profile:
        save_cloud_profile(args.cloud_profile, environment)
    return environment


def _lifecycle_from_args(args: argparse.Namespace) -> LifecycleProfile | None:
    interactive = _interactive_enabled(args)
    origin = args.origin or (
        _prompt("Training origin (existing_model/from_scratch)", "existing_model") if interactive else None
    )
    learning_mode = args.learning_mode or (_prompt("Learning mode", "qlora") if interactive else None)
    architecture_family = args.architecture_family
    if origin == "from_scratch" and not architecture_family and interactive:
        architecture_family = _prompt("Architecture family", "transformer")
    evaluation_suite = args.evaluation_suite or (_prompt("Evaluation suite", "") if interactive else "")

    compare_to_models = list(getattr(args, "compare_models", []) or [])
    if not compare_to_models and interactive:
        compare_to = _prompt("Compare against model (optional)", "")
        if compare_to:
            compare_to_models = [compare_to]

    deployment_targets = list(getattr(args, "deployment_targets", []) or [])
    if not deployment_targets and interactive:
        deployment_target = _prompt("Primary deployment target", "")
        if deployment_target:
            deployment_targets = [deployment_target]

    if not any(
        [
            origin,
            learning_mode,
            architecture_family,
            args.source_model,
            evaluation_suite,
            deployment_targets,
            args.architecture_hidden_size,
            args.architecture_layers,
            args.architecture_heads,
            args.architecture_context,
        ]
    ):
        return None

    architecture = ArchitectureSpec(
        family=architecture_family or None,
        hidden_size=args.architecture_hidden_size,
        num_layers=args.architecture_layers,
        num_attention_heads=args.architecture_heads,
        max_position_embeddings=args.architecture_context,
    )
    suite = None
    if evaluation_suite:
        suite = EvaluationSuiteSpec(
            id=evaluation_suite,
            label=evaluation_suite.replace("_", " ").replace("-", " ").title(),
            benchmark_config=evaluation_suite if evaluation_suite.endswith(".yaml") else None,
            compare_to_models=compare_to_models,
        )
    lifecycle = LifecycleProfile(
        origin=origin,
        learning_mode=learning_mode,
        source_model=args.source_model or None,
        architecture=architecture,
        evaluation_suite=suite,
        deployment_targets=deployment_targets,
    )
    return lifecycle


def _manifest_payload(manifest: InstanceManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")


def _parse_port_forward(raw: str) -> tuple[int, int, str]:
    parts = [part.strip() for part in raw.split(":") if part.strip()]
    if len(parts) == 2:
        local_port, remote_port = parts
        bind_host = "127.0.0.1"
    elif len(parts) == 3:
        local_port, remote_port, bind_host = parts
    else:
        raise ValueError("port forwards must look like local_port:remote_port[:bind_host]")
    return int(local_port), int(remote_port), bind_host


def _tail_file(path: Path, *, initial: str = "") -> None:
    if initial:
        print(initial, end="")
    cursor = len(initial.encode("utf-8"))
    while True:
        text = path.read_text() if path.exists() else ""
        encoded = text.encode("utf-8")
        if len(encoded) > cursor:
            print(encoded[cursor:].decode("utf-8"), end="")
            cursor = len(encoded)
        time.sleep(0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai-factory", description="Unified instance control plane for AI-Factory.")
    parser.add_argument("--repo-root")
    parser.add_argument("--artifacts-dir")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_json = argparse.ArgumentParser(add_help=False)
    common_json.add_argument("--json", action="store_true")

    new_parser = subparsers.add_parser("new", parents=[common_json])
    new_parser.add_argument("--config")
    new_parser.add_argument("--environment", choices=["local", "cloud"])
    new_parser.add_argument("--cloud-profile")
    new_parser.add_argument("--cloud-host")
    new_parser.add_argument("--cloud-user")
    new_parser.add_argument("--cloud-port", type=int, default=22)
    new_parser.add_argument("--cloud-key-path")
    new_parser.add_argument("--remote-repo-root")
    new_parser.add_argument("--python-bin")
    new_parser.add_argument(
        "--port-forward",
        dest="port_forwards",
        action="append",
        help="Forward local_port:remote_port[:bind_host] for cloud instances. Repeatable.",
    )
    new_parser.add_argument("--name")
    new_parser.add_argument("--user-level", choices=["beginner", "hobbyist", "dev"])
    new_parser.add_argument("--origin", choices=["existing_model", "from_scratch"])
    new_parser.add_argument(
        "--learning-mode",
        choices=["supervised", "unsupervised", "rlhf", "dpo", "orpo", "ppo", "lora", "qlora", "full_finetune"],
    )
    new_parser.add_argument("--source-model")
    new_parser.add_argument("--architecture-family")
    new_parser.add_argument("--architecture-hidden-size", type=int)
    new_parser.add_argument("--architecture-layers", type=int)
    new_parser.add_argument("--architecture-heads", type=int)
    new_parser.add_argument("--architecture-context", type=int)
    new_parser.add_argument("--evaluation-suite")
    new_parser.add_argument("--compare-model", dest="compare_models", action="append")
    new_parser.add_argument(
        "--deployment-target",
        dest="deployment_targets",
        action="append",
        choices=["huggingface", "ollama", "lmstudio", "openai_compatible_api"],
    )
    new_parser.add_argument("--no-start", action="store_true")
    new_parser.add_argument("--interactive", action="store_true", default=True)

    subparsers.add_parser("list", parents=[common_json])
    list_parser = subparsers.choices["list"]
    list_parser.add_argument("--type", dest="instance_type")
    list_parser.add_argument("--status")
    list_parser.add_argument("--parent", dest="parent_instance_id")

    status_parser = subparsers.add_parser("status", parents=[common_json])
    status_parser.add_argument("instance_id")

    children_parser = subparsers.add_parser("children", parents=[common_json])
    children_parser.add_argument("instance_id")

    tasks_parser = subparsers.add_parser("tasks", parents=[common_json])
    tasks_parser.add_argument("target")

    events_parser = subparsers.add_parser("events", parents=[common_json])
    events_parser.add_argument("target")
    events_parser.add_argument("--limit", type=int)

    retry_parser = subparsers.add_parser("retry", parents=[common_json])
    retry_parser.add_argument("target")

    cancel_parser = subparsers.add_parser("cancel", parents=[common_json])
    cancel_parser.add_argument("target")

    watch_parser = subparsers.add_parser("watch", parents=[common_json])
    watch_parser.add_argument("target")
    watch_parser.add_argument("--timeout", type=float, default=30.0)

    recommendations_parser = subparsers.add_parser("recommendations", parents=[common_json])
    recommendations_parser.add_argument("instance_id")

    action_parser = subparsers.add_parser("action", parents=[common_json])
    action_parser.add_argument("instance_id")
    action_parser.add_argument("action")
    action_parser.add_argument("--config")
    action_parser.add_argument(
        "--target",
        choices=[
            "huggingface",
            "ollama",
            "lmstudio",
            "api",
            "custom_api",
            "openai_compatible_api",
        ],
    )
    action_parser.add_argument("--no-start", action="store_true")

    logs_parser = subparsers.add_parser("logs", parents=[common_json])
    logs_parser.add_argument("instance_id")
    logs_parser.add_argument("--follow", action="store_true")
    logs_parser.add_argument("--stream", choices=["stdout", "stderr", "both"], default="both")

    eval_parser = subparsers.add_parser("evaluate", parents=[common_json])
    eval_parser.add_argument("instance_id")
    eval_parser.add_argument("--config", default="configs/eval.yaml")
    eval_parser.add_argument("--no-start", action="store_true")

    deploy_parser = subparsers.add_parser("deploy", parents=[common_json])
    deploy_parser.add_argument("instance_id")
    deploy_parser.add_argument(
        "--target",
        choices=[
            "huggingface",
            "ollama",
            "lmstudio",
            "api",
            "custom_api",
            "openai_compatible_api",
        ],
    )
    deploy_parser.add_argument("--config", default="configs/deploy.yaml")
    deploy_parser.add_argument("--no-start", action="store_true")

    inference_parser = subparsers.add_parser("inference", parents=[common_json])
    inference_parser.add_argument("instance_id")
    inference_parser.add_argument("--config", default="configs/inference.yaml")
    inference_parser.add_argument("--no-start", action="store_true")

    tui_parser = subparsers.add_parser("tui")
    tui_parser.add_argument("--refresh-seconds", type=float, default=2.0)

    workspace_parser = subparsers.add_parser("workspace", parents=[common_json])
    workspace_parser.add_argument("--root")

    ready_parser = subparsers.add_parser("ready", parents=[common_json])
    ready_parser.add_argument("--root")

    compare_parser = subparsers.add_parser("compare", parents=[common_json])
    compare_parser.add_argument("left_instance_id")
    compare_parser.add_argument("right_instance_id", nargs="?")

    doctor_parser = subparsers.add_parser("doctor", parents=[common_json])
    doctor_parser.add_argument("--root")

    api_smoke_parser = subparsers.add_parser("api-smoke", parents=[common_json])
    api_smoke_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    api_smoke_parser.add_argument("--skip-verify", action="store_true")
    api_smoke_parser.add_argument("--include-generate", action="store_true")

    subparsers.add_parser("latest-run", parents=[common_json])

    refresh_lab_parser = subparsers.add_parser("refresh-lab")
    refresh_lab_parser.add_argument("--skip-generate", action="store_true")
    refresh_lab_parser.add_argument("--skip-notebooks", action="store_true")
    refresh_lab_parser.add_argument("--skip-train-dry-run", action="store_true")
    refresh_lab_parser.add_argument("--skip-tests", action="store_true")
    refresh_lab_parser.add_argument("--profile", default="training/configs/profiles/baseline_qlora.yaml")

    domain_parser = subparsers.add_parser("domain", parents=[common_json])
    domain_subparsers = domain_parser.add_subparsers(dest="domain_command", required=True)

    domain_list_parser = domain_subparsers.add_parser("list", parents=[common_json])
    domain_list_parser.add_argument("--show-details", action="store_true")

    domain_info_parser = domain_subparsers.add_parser("info", parents=[common_json])
    domain_info_parser.add_argument("domain_name")

    platform_parser = subparsers.add_parser("platform", parents=[common_json])
    platform_subparsers = platform_parser.add_subparsers(dest="platform_command", required=True)

    platform_subparsers.add_parser("status", parents=[common_json])

    platform_scale_parser = platform_subparsers.add_parser("scale", parents=[common_json])
    platform_scale_parser.add_argument("target_nodes", type=int)

    multi_train_parser = subparsers.add_parser("multi-train", parents=[common_json])
    multi_train_parser.add_argument("--domains", nargs="+", required=True)
    multi_train_parser.add_argument("--config", default="configs/multi_domain.yaml")
    multi_train_parser.add_argument("--no-start", action="store_true")

    titan_parser = subparsers.add_parser("titan", parents=[common_json])
    titan_subparsers = titan_parser.add_subparsers(dest="titan_command", required=True)
    titan_status_parser = titan_subparsers.add_parser("status", parents=[common_json])
    titan_status_parser.add_argument("--write-hardware-doc", action="store_true")
    titan_doc_parser = titan_subparsers.add_parser("hardware-doc", parents=[common_json])
    titan_doc_parser.add_argument("--path", default="HARDWARE.md")

    serve_parser = subparsers.add_parser("serve", parents=[common_json])
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tui":
        from ai_factory.tui import run_tui

        run_tui(
            repo_root=args.repo_root,
            artifacts_dir=args.artifacts_dir,
            refresh_seconds=args.refresh_seconds,
        )
        return

    if args.command in {"workspace", "ready"}:
        from inference.app.workspace import build_workspace_overview

        root = Path(getattr(args, "root", None) or ".") if hasattr(args, "root") and args.root else None
        payload = build_workspace_overview(root)
        if args.json:
            _render_payload(payload, as_json=True)
            return
        if args.command == "ready":
            _render_ready_summary(payload)
        else:
            _render_workspace_overview(payload)
        return

    if args.command in {"doctor", "api-smoke", "latest-run", "refresh-lab"}:
        from ai_factory import cli_scripts

        if args.command == "doctor":
            cli_scripts.cmd_doctor(args)
        elif args.command == "api-smoke":
            cli_scripts.cmd_api_smoke(args)
        elif args.command == "latest-run":
            cli_scripts.cmd_latest_run(args)
        elif args.command == "refresh-lab":
            cli_scripts.cmd_refresh_lab(args)
        return

    if args.command == "titan":
        status = detect_titan_status(args.repo_root)
        if args.titan_command == "hardware-doc":
            output_path = write_hardware_markdown(args.path, repo_root=args.repo_root)
            payload = {"status": "generated", "path": str(output_path), "mode": status["mode"]}
            _render_payload(payload, as_json=args.json)
            return
        if args.write_hardware_doc:
            write_hardware_markdown(repo_root=args.repo_root)
        if args.json:
            _render_payload(status, as_json=True)
            return
        _render_titan_status(status)
        return

    if args.command == "serve":
        import subprocess

        cmd = ["uvicorn", "inference.app.main:app", "--host", args.host, "--port", str(args.port)]
        if args.reload:
            cmd.append("--reload")
        print(f"Starting AI-Factory API server on {args.host}:{args.port}")
        subprocess.run(cmd)
        return

    control = _build_control_service(args)

    if args.command == "new":
        environment = _environment_from_args(args)
        user_level_override = args.user_level or (
            _prompt("Experience level (beginner/hobbyist/dev)", "hobbyist") if _interactive_enabled(args) else None
        )
        name_override = args.name or (_prompt("Instance name", "") if _interactive_enabled(args) else "")
        manifest = control.create_instance(
            _new_config_path(args),
            start=not args.no_start,
            environment_override=environment,
            name_override=name_override or None,
            user_level_override=user_level_override,
            lifecycle_override=_lifecycle_from_args(args),
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "list":
        manifests = control.list_instances(
            instance_type=args.instance_type,
            status=args.status,
            parent_instance_id=args.parent_instance_id,
        )
        if args.json:
            _render_payload([_manifest_payload(item) for item in manifests], as_json=True)
            return
        if not manifests:
            print("No instances found. Create one with: ai-factory new --config configs/finetune.yaml")
            return
        print(f"Instances ({len(manifests)}):")
        rows = [_format_instance_row(item) for item in manifests]
        _render_payload(rows, as_json=False)
        return

    if args.command == "status":
        manifest = control.get_instance(args.instance_id)
        if args.json:
            _render_payload(_manifest_payload(manifest), as_json=True)
            return
        _render_instance_report(manifest)
        return

    if args.command == "children":
        manifests = control.get_children(args.instance_id)
        _render_payload([_manifest_payload(item) for item in manifests], as_json=args.json)
        return

    if args.command == "tasks":
        payload = control.list_tasks(args.target)
        _render_payload(payload, as_json=args.json)
        return

    if args.command == "events":
        payload = control.list_orchestration_events(args.target, limit=args.limit)
        _render_payload(payload, as_json=args.json)
        return

    if args.command == "retry":
        manifest = control.retry_instance(args.target)
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "cancel":
        manifest = control.cancel_instance(args.target)
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "watch":
        payload = control.watch_instance(args.target, timeout_s=args.timeout)
        _render_payload(payload, as_json=True if args.json else False)
        return

    if args.command == "recommendations":
        manifest = control.get_instance(args.instance_id)
        if args.json:
            payload = [item.model_dump(mode="json") for item in manifest.recommendations]
            _render_payload(payload, as_json=True)
            return
        if not manifest.recommendations:
            print(f"No recommendations for {args.instance_id}.")
            return
        _print_section(f"Recommendations for {args.instance_id}")
        for idx, rec in enumerate(manifest.recommendations, 1):
            print(f"  {idx}. [{rec.action}] priority={rec.priority}")
            print(f"     {rec.reason}")
            if rec.config_path:
                print(f"     config: {rec.config_path}")
            print(f"     run: ai-factory action {args.instance_id} {rec.action}")
        return

    if args.command == "action":
        manifest = control.execute_action(
            args.instance_id,
            action=args.action,
            config_path=args.config,
            deployment_target=args.target,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "logs":
        logs = control.get_logs(args.instance_id).model_dump(mode="json")
        if args.json:
            _render_payload(logs, as_json=True)
            return
        if args.stream in {"stdout", "both"}:
            print("== stdout ==")
            print(logs["stdout"], end="")
        if args.stream in {"stderr", "both"}:
            if args.stream == "both":
                print("\n== stderr ==")
            print(logs["stderr"], end="")
        if args.follow:
            store = control.store
            path = (
                store.stdout_path(args.instance_id)
                if args.stream in {"stdout", "both"}
                else store.stderr_path(args.instance_id)
            )
            _tail_file(path, initial="")
        return

    if args.command == "evaluate":
        manifest = control.create_evaluation_instance(
            args.instance_id,
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "deploy":
        manifest = control.create_deployment_instance(
            args.instance_id,
            target=_resolve_deploy_target(args),
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "inference":
        manifest = control.create_inference_instance(
            args.instance_id,
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "compare":
        left = control.get_instance(args.left_instance_id)
        if args.right_instance_id:
            right = control.get_instance(args.right_instance_id)
        else:
            parent_id = left.parent_instance_id
            if not parent_id:
                raise SystemExit("No right instance specified and no parent instance to compare against.")
            right = control.get_instance(parent_id)
        if args.json:
            _render_payload(
                {
                    "left": _manifest_payload(left),
                    "right": _manifest_payload(right),
                },
                as_json=True,
            )
            return
        _render_compare_summary(left, right)
        return

    if args.command == "domain":
        if args.domain_command == "list":
            from ai_factory.domains import list_available_domains

            domains = list_available_domains()
            if args.show_details:
                _render_payload([domain.model_dump() for domain in domains], as_json=args.json)
            else:
                _render_payload([domain.name for domain in domains], as_json=args.json)
            return

        if args.domain_command == "info":
            from ai_factory.domains import get_domain_info

            domain_info = get_domain_info(args.domain_name)
            _render_payload(domain_info, as_json=args.json)
            return

    if args.command == "platform":
        if args.platform_command == "status":
            from ai_factory.platform import get_platform_status

            status = get_platform_status()
            _render_payload(status, as_json=args.json)
            return

        if args.platform_command == "scale":
            from ai_factory.platform import scale_platform

            result = scale_platform(args.target_nodes)
            _render_payload(result, as_json=args.json)
            return

    if args.command == "multi-train":
        from ai_factory.platform import create_multi_domain_training

        manifest = create_multi_domain_training(
            domains=args.domains,
            config_path=args.config,
            start=not args.no_start,
            repo_root=args.repo_root,
            artifacts_dir=args.artifacts_dir,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return


if __name__ == "__main__":
    main()
