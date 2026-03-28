from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ai_factory.core.config.loader import load_orchestration_config, save_cloud_profile
from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.models import (
    ArchitectureSpec,
    EnvironmentSpec,
    EvaluationSuiteSpec,
    InstanceManifest,
    LifecycleProfile,
    PortForward,
)
from ai_factory.core.platform.container import build_platform_container


def _build_manager(args: argparse.Namespace) -> InstanceManager:
    return build_platform_container(
        repo_root=args.repo_root,
        artifacts_dir=args.artifacts_dir,
    ).manager


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


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    response = input(f"{text}{suffix}: ").strip()
    return response or (default or "")


def _interactive_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "interactive", False) and sys.stdin.isatty())


def _new_config_path(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    if _interactive_enabled(args):
        return _prompt("Config path", "configs/finetune.yaml")
    raise SystemExit("ai-factory new requires --config outside interactive mode")


def _deploy_target_from_config(config_path: str) -> str | None:
    try:
        return load_orchestration_config(config_path).subsystem.provider
    except Exception:
        return None


def _resolve_deploy_target(args: argparse.Namespace) -> str:
    if args.target:
        return args.target
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
    remote_repo_root = args.remote_repo_root or (_prompt("Remote repo root", "/tmp/ai-factory") if interactive else None)
    python_bin = args.python_bin or (_prompt("Remote python", "python3") if interactive else "python3")
    port_forwards = []
    for raw in getattr(args, "port_forwards", []) or []:
        local_port, remote_port, bind_host = _parse_port_forward(raw)
        port_forwards.append(
            PortForward(local_port=local_port, remote_port=remote_port, bind_host=bind_host)
        )
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
    origin = args.origin or (_prompt("Training origin (existing_model/from_scratch)", "existing_model") if interactive else None)
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
    parser = argparse.ArgumentParser(prog="ai-factory", description="Unified instance control plane for Atlas Math Lab.")
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
    deploy_parser.add_argument("--target", choices=["huggingface", "ollama", "lmstudio", "api", "custom_api", "openai_compatible_api"])
    deploy_parser.add_argument("--config", default="configs/deploy.yaml")
    deploy_parser.add_argument("--no-start", action="store_true")

    inference_parser = subparsers.add_parser("inference", parents=[common_json])
    inference_parser.add_argument("instance_id")
    inference_parser.add_argument("--config", default="configs/inference.yaml")
    inference_parser.add_argument("--no-start", action="store_true")

    tui_parser = subparsers.add_parser("tui")
    tui_parser.add_argument("--refresh-seconds", type=float, default=2.0)
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

    manager = _build_manager(args)

    if args.command == "new":
        environment = _environment_from_args(args)
        user_level_override = args.user_level or (
            _prompt("Experience level (beginner/hobbyist/dev)", "hobbyist") if _interactive_enabled(args) else None
        )
        name_override = args.name or (_prompt("Instance name", "") if _interactive_enabled(args) else "")
        manifest = manager.create_instance(
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
        manifests = manager.list_instances(
            instance_type=args.instance_type,
            status=args.status,
            parent_instance_id=args.parent_instance_id,
        )
        if args.json:
            _render_payload([_manifest_payload(item) for item in manifests], as_json=True)
            return
        rows = [
            (
                f"{item.id}  {item.type:<9}  {item.status:<9}  {item.environment.kind:<5}  "
                f"level={item.user_level:<9}  mode={item.orchestration_mode:<14}  parent={item.parent_instance_id or '-'}"
            )
            for item in manifests
        ]
        _render_payload(rows, as_json=False)
        return

    if args.command == "status":
        manifest = manager.get_instance(args.instance_id)
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "children":
        manifests = manager.get_children(args.instance_id)
        _render_payload([_manifest_payload(item) for item in manifests], as_json=args.json)
        return

    if args.command == "tasks":
        payload = manager.list_tasks(args.target)
        _render_payload(payload, as_json=args.json)
        return

    if args.command == "events":
        payload = manager.list_orchestration_events(args.target, limit=args.limit)
        _render_payload(payload, as_json=args.json)
        return

    if args.command == "retry":
        manifest = manager.retry_instance(args.target)
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "cancel":
        manifest = manager.cancel_instance(args.target)
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "watch":
        payload = manager.watch_instance(args.target, timeout_s=args.timeout)
        _render_payload(payload, as_json=True if args.json else False)
        return

    if args.command == "recommendations":
        manifest = manager.get_instance(args.instance_id)
        payload = [item.model_dump(mode="json") for item in manifest.recommendations]
        _render_payload(payload, as_json=args.json)
        return

    if args.command == "logs":
        logs = manager.get_logs(args.instance_id)
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
            store = manager.store
            path = (
                store.stdout_path(args.instance_id)
                if args.stream in {"stdout", "both"}
                else store.stderr_path(args.instance_id)
            )
            _tail_file(path, initial="")
        return

    if args.command == "evaluate":
        manifest = manager.create_evaluation_instance(
            args.instance_id,
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "deploy":
        manifest = manager.create_deployment_instance(
            args.instance_id,
            target=_resolve_deploy_target(args),
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return

    if args.command == "inference":
        manifest = manager.create_inference_instance(
            args.instance_id,
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return


if __name__ == "__main__":
    main()
