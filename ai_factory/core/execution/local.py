from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

from ai_factory.core.execution.base import BaseExecutor, CommandSpec, RunnerPayload, decode_payload, encode_payload
from ai_factory.core.instances.models import ExecutionHandle
from ai_factory.core.instances.store import FileInstanceStore


class LocalExecutor(BaseExecutor):
    backend_name = "local"

    def start(
        self,
        manifest,
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
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ai_factory.core.execution.local",
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


def _stream_pipe(pipe, path: Path) -> None:
    with path.open("a") as handle:
        for line in iter(pipe.readline, ""):
            handle.write(line)
            handle.flush()
    pipe.close()


def _heartbeat_loop(manager, instance_id: str, attempt_id: str, stop_event: threading.Event, interval_s: int) -> None:
    while not stop_event.wait(interval_s):
        try:
            manager.heartbeat_instance_attempt(instance_id, attempt_id)
        except Exception:
            return


def _run_payload(payload: RunnerPayload) -> int:
    store = FileInstanceStore(payload.artifacts_dir)
    from ai_factory.core.instances.manager import InstanceManager

    manager = InstanceManager(store)
    manifest = store.load(payload.instance_id)
    manager.mark_running(payload.instance_id)
    manifest = store.load(payload.instance_id)
    attempt_id = (((manifest.execution or ExecutionHandle(backend="local")).metadata) or {}).get("attempt_id")

    try:
        process = subprocess.Popen(
            payload.command.argv,
            cwd=payload.command.cwd,
            env={**os.environ, **payload.environment.env, **payload.command.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        manager.mark_failed(
            payload.instance_id,
            code="command_not_found",
            message=str(exc),
            details={"argv": payload.command.argv},
        )
        return 127

    manifest = store.load(payload.instance_id)
    if manifest.execution is not None:
        manifest.execution.pid = process.pid
        manifest.execution.metadata["argv"] = payload.command.argv
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
        runtime_metadata={"child_pid": process.pid, "argv": payload.command.argv},
    )
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal ai-factory local execution runner.")
    parser.add_argument("--payload", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(_run_payload(decode_payload(args.payload)))


if __name__ == "__main__":
    main()
