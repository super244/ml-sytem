from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ai_factory.core.config.loader import save_cloud_profile
from ai_factory.core.instances.manager import InstanceManager
from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest, PortForward
from ai_factory.core.instances.store import FileInstanceStore


def _build_manager() -> InstanceManager:
    return InstanceManager(FileInstanceStore("artifacts"))


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


def _environment_from_args(args: argparse.Namespace) -> EnvironmentSpec | None:
    kind = args.environment
    if kind is None and args.command == "new":
        if getattr(args, "interactive", True) and sys.stdin.isatty():
            kind = _prompt("Environment (local/cloud)", "local")
        else:
            return None
    if not kind:
        return None
    if kind == "local":
        return EnvironmentSpec(kind="local")

    interactive = bool(args.interactive and sys.stdin.isatty())
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_json = argparse.ArgumentParser(add_help=False)
    common_json.add_argument("--json", action="store_true")

    new_parser = subparsers.add_parser("new", parents=[common_json])
    new_parser.add_argument("--config", required=True)
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
    deploy_parser.add_argument("--target", required=True, choices=["huggingface", "ollama", "lmstudio", "custom_api"])
    deploy_parser.add_argument("--config", default="configs/deploy.yaml")
    deploy_parser.add_argument("--no-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = _build_manager()

    if args.command == "new":
        environment = _environment_from_args(args)
        manifest = manager.create_instance(
            args.config,
            start=not args.no_start,
            environment_override=environment,
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
            target=args.target,
            config_path=args.config,
            start=not args.no_start,
        )
        _render_payload(_manifest_payload(manifest), as_json=args.json)
        return


if __name__ == "__main__":
    main()
