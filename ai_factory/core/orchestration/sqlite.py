from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from ai_factory.core.orchestration.models import (
    CircuitState,
    OrchestrationEvent,
    OrchestrationRun,
    OrchestrationTask,
    TaskAttempt,
    TaskDependency,
)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load(value: str | None, default: Any) -> Any:
    if value in {None, ""}:
        return default
    return json.loads(value) if value is not None else {}


class SqliteControlPlane:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS orchestration_runs (
                    id TEXT PRIMARY KEY,
                    legacy_instance_id TEXT UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    root_run_id TEXT,
                    parent_run_id TEXT,
                    idempotency_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_idempotency_key
                ON orchestration_runs(idempotency_key)
                WHERE idempotency_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS orchestration_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
                    legacy_instance_id TEXT,
                    parent_task_id TEXT,
                    task_type TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    resource_class TEXT NOT NULL,
                    retry_policy_json TEXT NOT NULL,
                    current_attempt INTEGER NOT NULL,
                    last_error_code TEXT,
                    last_error_message TEXT,
                    checkpoint_hint TEXT,
                    queued_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON orchestration_tasks(run_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_status_available ON orchestration_tasks(status, available_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency
                ON orchestration_tasks(run_id, legacy_instance_id, task_type);

                CREATE TABLE IF NOT EXISTS orchestration_task_dependencies (
                    task_id TEXT NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
                    depends_on_task_id TEXT NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, depends_on_task_id)
                );

                CREATE TABLE IF NOT EXISTS orchestration_attempts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    lease_owner TEXT,
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    finished_at TEXT,
                    stdout_path TEXT,
                    stderr_path TEXT,
                    exit_code INTEGER,
                    checkpoint_hint TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_task_id ON orchestration_attempts(task_id);

                CREATE TABLE IF NOT EXISTS orchestration_leases (
                    task_id TEXT PRIMARY KEY REFERENCES orchestration_tasks(id) ON DELETE CASCADE,
                    attempt_id TEXT NOT NULL REFERENCES orchestration_attempts(id) ON DELETE CASCADE,
                    lease_owner TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orchestration_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES orchestration_runs(id) ON DELETE CASCADE,
                    task_id TEXT,
                    attempt_id TEXT,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    agent_type TEXT,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_id_created_at
                ON orchestration_events(run_id, created_at);

                CREATE TABLE IF NOT EXISTS orchestration_circuits (
                    agent_type TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    failure_count INTEGER NOT NULL,
                    opened_at TEXT,
                    reopen_after TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def upsert_run(self, run: OrchestrationRun) -> OrchestrationRun:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_runs (
                    id, legacy_instance_id, name, status, root_run_id, parent_run_id, idempotency_key,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    legacy_instance_id=excluded.legacy_instance_id,
                    name=excluded.name,
                    status=excluded.status,
                    root_run_id=excluded.root_run_id,
                    parent_run_id=excluded.parent_run_id,
                    idempotency_key=excluded.idempotency_key,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    run.id,
                    run.legacy_instance_id,
                    run.name,
                    run.status,
                    run.root_run_id,
                    run.parent_run_id,
                    run.idempotency_key,
                    run.created_at,
                    run.updated_at,
                    _dump(run.metadata),
                ),
            )
        return run

    def get_run(self, run_id: str) -> OrchestrationRun | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def get_run_by_legacy_instance(self, legacy_instance_id: str) -> OrchestrationRun | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_runs WHERE legacy_instance_id = ?",
                (legacy_instance_id,),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def list_runs(self) -> list[OrchestrationRun]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM orchestration_runs ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def upsert_task(self, task: OrchestrationTask) -> OrchestrationTask:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_tasks (
                    id, run_id, legacy_instance_id, parent_task_id, task_type, agent_type, status, priority,
                    resource_class, retry_policy_json, current_attempt, last_error_code, last_error_message,
                    checkpoint_hint, queued_at, available_at, started_at, finished_at, created_at, updated_at,
                    input_json, output_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    run_id=excluded.run_id,
                    legacy_instance_id=excluded.legacy_instance_id,
                    parent_task_id=excluded.parent_task_id,
                    task_type=excluded.task_type,
                    agent_type=excluded.agent_type,
                    status=excluded.status,
                    priority=excluded.priority,
                    resource_class=excluded.resource_class,
                    retry_policy_json=excluded.retry_policy_json,
                    current_attempt=excluded.current_attempt,
                    last_error_code=excluded.last_error_code,
                    last_error_message=excluded.last_error_message,
                    checkpoint_hint=excluded.checkpoint_hint,
                    queued_at=excluded.queued_at,
                    available_at=excluded.available_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    updated_at=excluded.updated_at,
                    input_json=excluded.input_json,
                    output_json=excluded.output_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    task.id,
                    task.run_id,
                    task.legacy_instance_id,
                    task.parent_task_id,
                    task.task_type,
                    task.agent_type,
                    task.status,
                    task.priority,
                    task.resource_class,
                    task.retry_policy.model_dump_json(),
                    task.current_attempt,
                    task.last_error_code,
                    task.last_error_message,
                    task.checkpoint_hint,
                    task.queued_at,
                    task.available_at,
                    task.started_at,
                    task.finished_at,
                    task.created_at,
                    task.updated_at,
                    task.input.model_dump_json(),
                    task.output.model_dump_json(),
                    _dump(task.metadata),
                ),
            )
        return task

    def get_task(self, task_id: str) -> OrchestrationTask | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return self._task_from_row(row) if row else None

    def get_task_by_legacy_instance(self, legacy_instance_id: str) -> OrchestrationTask | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_tasks WHERE legacy_instance_id = ? ORDER BY created_at LIMIT 1",
                (legacy_instance_id,),
            ).fetchone()
        return self._task_from_row(row) if row else None

    def list_tasks(self, *, run_id: str | None = None) -> list[OrchestrationTask]:
        with self.connection() as connection:
            if run_id:
                rows = connection.execute(
                    "SELECT * FROM orchestration_tasks WHERE run_id = ? ORDER BY created_at, priority",
                    (run_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM orchestration_tasks ORDER BY created_at, priority"
                ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def create_dependency(self, dependency: TaskDependency) -> TaskDependency:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO orchestration_task_dependencies (task_id, depends_on_task_id, created_at)
                VALUES (?, ?, ?)
                """,
                (dependency.task_id, dependency.depends_on_task_id, dependency.created_at),
            )
        return dependency

    def list_dependencies(self, task_id: str | None = None) -> list[TaskDependency]:
        with self.connection() as connection:
            if task_id:
                rows = connection.execute(
                    "SELECT * FROM orchestration_task_dependencies WHERE task_id = ? ORDER BY created_at",
                    (task_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM orchestration_task_dependencies ORDER BY created_at"
                ).fetchall()
        return [TaskDependency.model_validate(dict(row)) for row in rows]

    def upsert_attempt(self, attempt: TaskAttempt) -> TaskAttempt:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_attempts (
                    id, task_id, sequence, status, lease_owner, started_at, heartbeat_at, finished_at,
                    stdout_path, stderr_path, exit_code, checkpoint_hint, error_code, error_message,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    task_id=excluded.task_id,
                    sequence=excluded.sequence,
                    status=excluded.status,
                    lease_owner=excluded.lease_owner,
                    heartbeat_at=excluded.heartbeat_at,
                    finished_at=excluded.finished_at,
                    stdout_path=excluded.stdout_path,
                    stderr_path=excluded.stderr_path,
                    exit_code=excluded.exit_code,
                    checkpoint_hint=excluded.checkpoint_hint,
                    error_code=excluded.error_code,
                    error_message=excluded.error_message,
                    metadata_json=excluded.metadata_json
                """,
                (
                    attempt.id,
                    attempt.task_id,
                    attempt.sequence,
                    attempt.status,
                    attempt.lease_owner,
                    attempt.started_at,
                    attempt.heartbeat_at,
                    attempt.finished_at,
                    attempt.stdout_path,
                    attempt.stderr_path,
                    attempt.exit_code,
                    attempt.checkpoint_hint,
                    attempt.error_code,
                    attempt.error_message,
                    _dump(attempt.metadata),
                ),
            )
        return attempt

    def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_attempts WHERE id = ?",
                (attempt_id,),
            ).fetchone()
        return self._attempt_from_row(row) if row else None

    def get_active_attempt_for_task(self, task_id: str) -> TaskAttempt | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM orchestration_attempts
                WHERE task_id = ? AND status = 'running'
                ORDER BY sequence DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return self._attempt_from_row(row) if row else None

    def list_attempts(self, task_id: str) -> list[TaskAttempt]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM orchestration_attempts WHERE task_id = ? ORDER BY sequence",
                (task_id,),
            ).fetchall()
        return [self._attempt_from_row(row) for row in rows]

    def write_lease(
        self,
        *,
        task_id: str,
        attempt_id: str,
        lease_owner: str,
        acquired_at: str,
        heartbeat_at: str,
        expires_at: str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_leases (task_id, attempt_id, lease_owner, acquired_at, heartbeat_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    attempt_id=excluded.attempt_id,
                    lease_owner=excluded.lease_owner,
                    acquired_at=excluded.acquired_at,
                    heartbeat_at=excluded.heartbeat_at,
                    expires_at=excluded.expires_at
                """,
                (task_id, attempt_id, lease_owner, acquired_at, heartbeat_at, expires_at),
            )

    def drop_lease(self, task_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM orchestration_leases WHERE task_id = ?", (task_id,))

    def list_stale_leases(self, *, stale_before: str) -> list[sqlite3.Row]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM orchestration_leases WHERE expires_at <= ? ORDER BY expires_at",
                (stale_before,),
            ).fetchall()
        return rows

    def append_event(self, event: OrchestrationEvent) -> OrchestrationEvent:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_events (
                    id, run_id, task_id, attempt_id, event_type, level, agent_type, message, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.run_id,
                    event.task_id,
                    event.attempt_id,
                    event.event_type,
                    event.level,
                    event.agent_type,
                    event.message,
                    event.created_at,
                    _dump(event.payload),
                ),
            )
        return event

    def list_events(self, *, run_id: str, limit: int | None = None) -> list[OrchestrationEvent]:
        with self.connection() as connection:
            if limit:
                rows = connection.execute(
                    """
                    SELECT * FROM orchestration_events
                    WHERE run_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (run_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = connection.execute(
                    "SELECT * FROM orchestration_events WHERE run_id = ? ORDER BY created_at",
                    (run_id,),
                ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def upsert_circuit(self, circuit: CircuitState) -> CircuitState:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO orchestration_circuits (
                    agent_type, status, failure_count, opened_at, reopen_after, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_type) DO UPDATE SET
                    status=excluded.status,
                    failure_count=excluded.failure_count,
                    opened_at=excluded.opened_at,
                    reopen_after=excluded.reopen_after,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (
                    circuit.agent_type,
                    circuit.status,
                    circuit.failure_count,
                    circuit.opened_at,
                    circuit.reopen_after,
                    circuit.last_error,
                    circuit.updated_at,
                ),
            )
        return circuit

    def get_circuit(self, agent_type: str) -> CircuitState | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM orchestration_circuits WHERE agent_type = ?",
                (agent_type,),
            ).fetchone()
        return CircuitState.model_validate(dict(row)) if row else None

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> OrchestrationRun:
        payload = dict(row)
        payload["metadata"] = _load(payload.pop("metadata_json"), {})
        return OrchestrationRun.model_validate(payload)

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> OrchestrationTask:
        payload = dict(row)
        payload["retry_policy"] = _load(payload.pop("retry_policy_json"), {})
        payload["input"] = _load(payload.pop("input_json"), {})
        payload["output"] = _load(payload.pop("output_json"), {})
        payload["metadata"] = _load(payload.pop("metadata_json"), {})
        return OrchestrationTask.model_validate(payload)

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> TaskAttempt:
        payload = dict(row)
        payload["metadata"] = _load(payload.pop("metadata_json"), {})
        return TaskAttempt.model_validate(payload)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> OrchestrationEvent:
        payload = dict(row)
        payload["payload"] = _load(payload.pop("payload_json"), {})
        return OrchestrationEvent.model_validate(payload)
