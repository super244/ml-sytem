from __future__ import annotations

from types import SimpleNamespace

from ai_factory import tui
from ai_factory.core.control.models import InstanceLogsView, InstanceMetricsView, LiveInstanceSnapshot
from ai_factory.core.instances.models import EnvironmentSpec, InstanceManifest, ProgressSnapshot


class _FakeControlService:
    def __init__(self, manifests, logs, metrics, summary):
        self._manifests = manifests
        self._logs = logs
        self._metrics = metrics
        self._summary = summary

    def list_instances(self):
        return self._manifests

    def get_live_instance_snapshot(self, instance_id: str) -> LiveInstanceSnapshot:
        manifest = next(item for item in self._manifests if item.id == instance_id)
        logs = self._logs.get(instance_id, {"stdout": "", "stderr": ""})
        metrics = self._metrics.get(instance_id, {"summary": {}, "points": []})
        return LiveInstanceSnapshot(
            instance=manifest,
            logs=InstanceLogsView(**logs),
            metrics=InstanceMetricsView(**metrics),
            events=[],
            tasks=[],
            available_actions=[],
            orchestration_summary=self._summary,
        )

    def monitoring_summary(self) -> dict[str, object]:
        return self._summary


def test_parse_args_supports_shared_backend_overrides():
    args = tui.parse_args(
        ["--repo-root", "/workspace", "--artifacts-dir", "/tmp/artifacts", "--refresh-seconds", "5"]
    )

    assert args.repo_root == "/workspace"
    assert args.artifacts_dir == "/tmp/artifacts"
    assert args.refresh_seconds == 5.0


def test_controller_refresh_loads_instances_logs_metrics_and_summary(monkeypatch):
    manifest = InstanceManifest(
        id="train-002",
        type="train",
        name="train",
        status="running",
        environment=EnvironmentSpec(kind="cloud", host="10.0.0.2", user="ai"),
        progress=ProgressSnapshot(stage="training", status_message="running", percent=0.5),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    control = _FakeControlService(
        [manifest],
        {"train-002": {"stdout": "step 1\nstep 2\n", "stderr": "warn\n"}},
        {"train-002": {"summary": {"accuracy": 0.92, "avg_latency_s": 1.4}, "points": []}},
        {"runs": 2, "tasks": 4, "task_status_counts": {"running": 1, "retry_waiting": 1}},
    )
    monkeypatch.setattr(tui, "build_platform_container", lambda **_: SimpleNamespace(control_service=control))

    controller = tui.TuiController(repo_root=None, artifacts_dir=None, refresh_seconds=2.0)
    controller.refresh()

    assert len(controller.snapshot.instances) == 1
    assert controller.snapshot.logs["stdout"] == "step 1\nstep 2\n"
    assert controller.snapshot.metrics["summary"]["accuracy"] == 0.92
    assert controller.snapshot.summary["tasks"] == 4


def test_controller_move_and_toggle_stream(monkeypatch):
    manifest = InstanceManifest(
        id="deploy-001",
        type="deploy",
        name="deploy",
        status="pending",
        environment=EnvironmentSpec(kind="local"),
    )
    control = _FakeControlService(
        [manifest],
        {"deploy-001": {"stdout": "", "stderr": "warn"}},
        {"deploy-001": {"summary": {}, "points": []}},
        {"runs": 1, "tasks": 1, "task_status_counts": {}},
    )
    monkeypatch.setattr(tui, "build_platform_container", lambda **_: SimpleNamespace(control_service=control))

    controller = tui.TuiController(repo_root=None, artifacts_dir=None, refresh_seconds=2.0)
    controller.refresh()
    controller.move(1)
    controller.toggle_stream()

    assert controller.selected_index == 0
    assert controller.selected_stream == "stderr"


def test_run_tui_uses_curses_wrapper(monkeypatch):
    manifest = InstanceManifest(
        id="deploy-001",
        type="deploy",
        name="deploy",
        status="pending",
        environment=EnvironmentSpec(kind="local"),
    )
    control = _FakeControlService(
        [manifest],
        {"deploy-001": {"stdout": "", "stderr": ""}},
        {"deploy-001": {"summary": {}, "points": []}},
        {"runs": 1, "tasks": 1, "task_status_counts": {}},
    )
    monkeypatch.setattr(tui, "build_platform_container", lambda **_: SimpleNamespace(control_service=control))
    captured: dict[str, object] = {}

    class _FakeCurses:
        COLOR_YELLOW = 3
        COLOR_GREEN = 2
        COLOR_RED = 1
        COLOR_CYAN = 6
        A_REVERSE = 1
        A_DIM = 2
        A_BOLD = 4
        KEY_UP = 259
        KEY_DOWN = 258

        def wrapper(self, fn, controller):
            captured["wrapped"] = True

            class _Screen:
                def nodelay(self, value):
                    captured["nodelay"] = value

                def keypad(self, value):
                    captured["keypad"] = value

                def getmaxyx(self):
                    return (24, 100)

                def erase(self):
                    captured["erase"] = True

                def addnstr(self, *args, **kwargs):
                    captured["addnstr"] = True

                def refresh(self):
                    captured["refresh"] = True

                def getch(self):
                    return ord("q")

            fn(_Screen(), controller)

        def curs_set(self, value):
            captured["curs_set"] = value

        def has_colors(self):
            return False

    monkeypatch.setattr(tui, "curses", _FakeCurses())

    tui.run_tui(repo_root=None, artifacts_dir=None, refresh_seconds=2.0)

    assert captured["wrapped"] is True
    assert captured["curs_set"] == 0
