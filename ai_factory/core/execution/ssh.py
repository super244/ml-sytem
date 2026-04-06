from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ai_factory.core.execution.base import BaseExecutor, CommandSpec, RunnerPayload, decode_payload, encode_payload
from ai_factory.core.instances.models import EnvironmentSpec, ExecutionHandle, InstanceManifest


def _ssh_target(environment: EnvironmentSpec) -> str:
    if not environment.host or not environment.user:
        raise ValueError("SSH execution requires environment.host and environment.user")
    return f"{environment.user}@{environment.host}"


def _build_remote_command(payload: RunnerPayload) -> str:
    cwd = payload.command.cwd or payload.environment.remote_repo_root or "."
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in {**payload.environment.env, **payload.command.env}.items()
    )
    command = shlex.join(payload.command.argv)
    if env_prefix:
        command = f"{env_prefix} {command}"
    return f"cd {shlex.quote(cwd)} && {command}"


def _port_forward_args(metadata: dict[str, Any]) -> list[str]:
    remote_access = metadata.get("remote_access") or {}
    forwards = remote_access.get("port_forwards") or []
    argv: list[str] = []
    for forward in forwards:
        local_port = forward.get("local_port")
        remote_port = forward.get("remote_port")
        bind_host = forward.get("bind_host", "127.0.0.1")
        if local_port is None or remote_port is None:
            continue
        argv.extend(["-L", f"{bind_host}:{local_port}:127.0.0.1:{remote_port}"])
    if remote_access.get("agent_forwarding"):
        argv.append("-A")
    keepalive = remote_access.get("ssh_keepalive_s")
    if isinstance(keepalive, int) and keepalive > 0:
        argv.extend(["-o", f"ServerAliveInterval={keepalive}"])
    return argv


def _build_ssh_argv(environment: EnvironmentSpec, metadata: dict[str, Any]) -> list[str]:
    ssh_bin = shutil.which("ssh")
    if ssh_bin is None:
        raise FileNotFoundError("ssh executable not found in PATH")
    argv = [ssh_bin, "-p", str(environment.port)]
    if environment.key_path:
        argv.extend(["-i", environment.key_path])
    argv.extend(_port_forward_args(metadata))
    return argv


def _stream_pipe(pipe: Any, path: Path) -> None:
    with path.open("a") as handle:
        for line in iter(pipe.readline, ""):
            handle.write(line)
            handle.flush()
    pipe.close()


def _heartbeat_loop(
    manager: Any, instance_id: str, attempt_id: str, stop_event: threading.Event, interval_s: int
) -> None:
    while not stop_event.wait(interval_s):
        try:
            manager.heartbeat_instance_attempt(instance_id, attempt_id)
        except Exception:
            return


class SshExecutor(BaseExecutor):
    backend_name = "ssh"

    def start(
        self,
        manifest: InstanceManifest,
        command: CommandSpec,
        *,
        artifacts_dir: str | Path,
        stdout_path: str,
        stderr_path: str,
    ) -> ExecutionHandle:
        payload = RunnerPayload(
            artifacts_dir=str(artifacts_dir),
            instance_id=manifest.id,
            environment=manifest.environment,
            manifest_metadata=manifest.metadata,
            command=command,
        )
        process = subprocess.Popen(  # nosec B603 - internal runner bootstrap with explicit argv
            [
                sys.executable,
                "-m",
                "ai_factory.core.execution.ssh",
                "--payload",
                encode_payload(payload),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return ExecutionHandle(
            backend=self.backend_name,
            pid=process.pid,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={"runner_pid": process.pid},
        )

    def read_remote_file(self, manifest: InstanceManifest, path: str) -> str:
        target = _ssh_target(manifest.environment)
        argv = _build_ssh_argv(manifest.environment, manifest.metadata)
        argv.extend([target, f"cat {shlex.quote(path)}"])
        result = subprocess.run(argv, capture_output=True, text=True, check=True)  # nosec B603 - ssh argv with shell=False
        return result.stdout


def _run_payload(payload: RunnerPayload) -> int:
    from ai_factory.core.instances.manager import InstanceManager
    from ai_factory.core.instances.store import FileInstanceStore

    store = FileInstanceStore(payload.artifacts_dir)
    manager = InstanceManager(store)
    manager.mark_running(payload.instance_id)
    manifest = store.load(payload.instance_id)
    attempt_id = (((manifest.execution or ExecutionHandle(backend="ssh")).metadata) or {}).get("attempt_id")

    ssh_command = _build_ssh_argv(payload.environment, payload.manifest_metadata)
    ssh_command.extend([_ssh_target(payload.environment), _build_remote_command(payload)])

    process = subprocess.Popen(  # nosec B603 - ssh command constructed from validated environment and argv
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    manifest = store.load(payload.instance_id)
    if manifest.execution is not None:
        manifest.execution.pid = process.pid
        manifest.execution.metadata["argv"] = ssh_command
        store.save(manifest)

    stop_event = threading.Event()
    heartbeat_thread = None
    if isinstance(attempt_id, str):
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(manager, payload.instance_id, attempt_id, stop_event, manager.platform_settings.heartbeat_interval_s),
            daemon=True,
        )
        heartbeat_thread.start()

    stdout_thread = threading.Thread(target=_stream_pipe, args=(process.stdout, store.stdout_path(payload.instance_id)))
    stderr_thread = threading.Thread(target=_stream_pipe, args=(process.stderr, store.stderr_path(payload.instance_id)))
    stdout_thread.start()
    stderr_thread.start()
    exit_code = process.wait()
    stop_event.set()
    stdout_thread.join()
    stderr_thread.join()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)
    manager.finalize_instance(
        payload.instance_id,
        exit_code,
        runtime_metadata={"ssh_command": ssh_command},
    )
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal ai-factory SSH execution runner.")
    parser.add_argument("--payload", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(_run_payload(decode_payload(args.payload)))


if __name__ == "__main__":
    main()
