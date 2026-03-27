from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from ai_factory.core.execution.base import BaseExecutor, CommandSpec, RunnerPayload, decode_payload, encode_payload
from ai_factory.core.instances.models import ExecutionHandle, InstanceManifest


def _ssh_target(environment) -> str:
    if not environment.host or not environment.user:
        raise ValueError("SSH execution requires environment.host and environment.user")
    return f"{environment.user}@{environment.host}"


def _build_remote_command(payload: RunnerPayload) -> str:
    cwd = payload.command.cwd or payload.environment.remote_repo_root or "."
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in {**payload.environment.env, **payload.command.env}.items()
    )
    command = shlex.join(payload.command.argv)
    if env_prefix:
        command = f"{env_prefix} {command}"
    return f"cd {shlex.quote(cwd)} && {command}"


def _stream_pipe(pipe, path: Path) -> None:
    with path.open("a") as handle:
        for line in iter(pipe.readline, ""):
            handle.write(line)
            handle.flush()
    pipe.close()


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
            command=command,
        )
        process = subprocess.Popen(
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
        argv = ["ssh", "-p", str(manifest.environment.port)]
        if manifest.environment.key_path:
            argv.extend(["-i", manifest.environment.key_path])
        argv.extend([target, f"cat {shlex.quote(path)}"])
        result = subprocess.run(argv, capture_output=True, text=True, check=True)
        return result.stdout


def _run_payload(payload: RunnerPayload) -> int:
    from ai_factory.core.instances.manager import InstanceManager
    from ai_factory.core.instances.store import FileInstanceStore

    store = FileInstanceStore(payload.artifacts_dir)
    manager = InstanceManager(store)
    manager.mark_running(payload.instance_id)

    ssh_command = ["ssh", "-p", str(payload.environment.port)]
    if payload.environment.key_path:
        ssh_command.extend(["-i", payload.environment.key_path])
    ssh_command.extend([_ssh_target(payload.environment), _build_remote_command(payload)])

    process = subprocess.Popen(
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

    stdout_thread = threading.Thread(target=_stream_pipe, args=(process.stdout, store.stdout_path(payload.instance_id)))
    stderr_thread = threading.Thread(target=_stream_pipe, args=(process.stderr, store.stderr_path(payload.instance_id)))
    stdout_thread.start()
    stderr_thread.start()
    exit_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
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
