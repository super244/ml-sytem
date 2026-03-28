from __future__ import annotations

import argparse
import curses
import locale
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_factory.core.platform.container import build_platform_container


locale.setlocale(locale.LC_ALL, "")


@dataclass
class DashboardSnapshot:
    instances: list[Any]
    summary: dict[str, Any]
    logs: dict[str, str]
    metrics: dict[str, Any]
    selected_stream: str
    loaded_at: float
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-factory tui",
        description="Interactive terminal dashboard for AI-Factory instances.",
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def _safe_addstr(screen, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = screen.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    limit = max(width - x - 1, 0)
    if limit <= 0:
        return
    try:
        screen.addnstr(y, x, text, limit, attr)
    except curses.error:
        return


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "n/a"
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _format_metric(value: Any, *, percent: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if percent:
        return f"{value * 100:.1f}%"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _metric(summary: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in summary:
            return summary[name]
    return None


def _tail_lines(text: str, *, max_lines: int) -> list[str]:
    lines = text.splitlines() or [""]
    return lines[-max_lines:]


class TuiController:
    def __init__(self, *, repo_root: str | None, artifacts_dir: str | None, refresh_seconds: float):
        self.control = build_platform_container(
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        ).control_service
        self.refresh_seconds = max(refresh_seconds, 0.5)
        self.selected_index = 0
        self.selected_stream = "stdout"
        self.snapshot = DashboardSnapshot(
            instances=[],
            summary={},
            logs={"stdout": "", "stderr": ""},
            metrics={"summary": {}, "points": []},
            selected_stream=self.selected_stream,
            loaded_at=0.0,
        )

    def refresh(self) -> None:
        try:
            instances = self.control.list_instances()
            if instances:
                self.selected_index = max(0, min(self.selected_index, len(instances) - 1))
                selected = instances[self.selected_index]
                live = self.control.get_live_instance_snapshot(selected.id)
                logs = live.logs.model_dump(mode="json")
                metrics = live.metrics.model_dump(mode="json")
            else:
                self.selected_index = 0
                logs = {"stdout": "", "stderr": ""}
                metrics = {"summary": {}, "points": []}
            self.snapshot = DashboardSnapshot(
                instances=instances,
                summary=self.control.monitoring_summary(),
                logs=logs,
                metrics=metrics,
                selected_stream=self.selected_stream,
                loaded_at=time.time(),
            )
        except Exception as exc:
            self.snapshot = DashboardSnapshot(
                instances=self.snapshot.instances,
                summary=self.snapshot.summary,
                logs=self.snapshot.logs,
                metrics=self.snapshot.metrics,
                selected_stream=self.selected_stream,
                loaded_at=time.time(),
                error=str(exc),
            )

    def move(self, delta: int) -> None:
        if not self.snapshot.instances:
            return
        self.selected_index = max(0, min(self.selected_index + delta, len(self.snapshot.instances) - 1))

    def toggle_stream(self) -> None:
        self.selected_stream = "stderr" if self.selected_stream == "stdout" else "stdout"


def _status_attr(status: str) -> int:
    if not curses.has_colors():
        return curses.A_BOLD
    mapping = {
        "running": curses.color_pair(2),
        "completed": curses.color_pair(3),
        "failed": curses.color_pair(4),
        "pending": curses.color_pair(5),
    }
    return mapping.get(status, curses.A_NORMAL)


def _render_header(screen, snapshot: DashboardSnapshot) -> int:
    summary = snapshot.summary or {}
    counts = summary.get("task_status_counts") or {}
    header = [
        "ai-factory TUI",
        f"instances={len(snapshot.instances)}",
        f"runs={summary.get('runs', 0)}",
        f"tasks={summary.get('tasks', 0)}",
        f"running={counts.get('running', 0)}",
        f"retries={counts.get('retry_waiting', 0)}",
        f"stream={snapshot.selected_stream}",
        f"loaded={time.strftime('%H:%M:%S', time.localtime(snapshot.loaded_at or time.time()))}",
    ]
    _safe_addstr(screen, 0, 0, " | ".join(header), curses.A_REVERSE)
    _safe_addstr(screen, 1, 0, "q quit  up/down move  tab toggle log stream  r refresh", curses.A_DIM)
    if snapshot.error:
        _safe_addstr(screen, 2, 0, f"Last refresh error: {snapshot.error}", curses.A_BOLD)
        return 4
    return 3


def _render_instances(screen, top: int, width: int, height: int, controller: TuiController) -> None:
    _safe_addstr(screen, top, 0, "Instances", curses.A_BOLD)
    instances = controller.snapshot.instances
    if not instances:
        _safe_addstr(screen, top + 1, 0, "No managed instances yet.", curses.A_DIM)
        return
    available_rows = max(height - top - 1, 1)
    start = max(0, controller.selected_index - available_rows + 1)
    rows = instances[start : start + available_rows]
    for offset, instance in enumerate(rows, start=1):
        row = top + offset
        selected = start + offset - 1 == controller.selected_index
        attr = curses.A_REVERSE if selected else _status_attr(instance.status)
        summary = f"{instance.type:<9} {instance.status:<9} {instance.environment.kind:<5} {instance.name}"
        _safe_addstr(screen, row, 0, summary[: max(width - 1, 1)], attr)


def _render_detail(screen, top: int, left: int, width: int, height: int, controller: TuiController) -> None:
    snapshot = controller.snapshot
    if not snapshot.instances:
        _safe_addstr(screen, top, left, "Details", curses.A_BOLD)
        _safe_addstr(screen, top + 1, left, "Create an instance with `ai-factory new --config ...`.", curses.A_DIM)
        return
    instance = snapshot.instances[controller.selected_index]
    metrics_summary = snapshot.metrics.get("summary") or {}
    logs = snapshot.logs.get(controller.selected_stream, "")

    _safe_addstr(screen, top, left, f"Details: {instance.id}", curses.A_BOLD)
    lines = [
        f"name: {instance.name}",
        f"type/status: {instance.type} / {instance.status}",
        f"lifecycle: {(instance.lifecycle.stage if instance.lifecycle and instance.lifecycle.stage else 'n/a')}",
        f"origin/mode: {((instance.lifecycle.origin or 'n/a') if instance.lifecycle else 'n/a')} / {((instance.lifecycle.learning_mode or 'n/a') if instance.lifecycle else 'n/a')}",
        f"environment: {instance.environment.kind}",
        f"orchestration run: {instance.orchestration_run_id or 'n/a'}",
        f"progress: {(instance.progress.stage if instance.progress else 'n/a')}",
        f"updated: {_format_timestamp(instance.updated_at)}",
        f"accuracy: {_format_metric(_metric(metrics_summary, 'accuracy'), percent=True)}",
        f"parse rate: {_format_metric(_metric(metrics_summary, 'parse_rate'), percent=True)}",
        f"latency: {_format_metric(_metric(metrics_summary, 'avg_latency_s'))} s",
        f"latest step: {_format_metric(_metric(metrics_summary, 'latest_step'))}",
    ]
    if instance.parent_instance_id:
        lines.append(f"parent: {instance.parent_instance_id}")
    if instance.progress and instance.progress.status_message:
        lines.append(f"message: {instance.progress.status_message}")
    if instance.decision is not None:
        lines.append(f"decision: {instance.decision.action} ({instance.decision.rule})")
    if instance.recommendations:
        lines.append(f"recommendations: {len(instance.recommendations)}")
        lines.append(f"next: {instance.recommendations[0].action}")

    log_top = top + len(lines) + 2
    max_detail_lines = max(height - top - 6, 1)
    for offset, line in enumerate(lines[:max_detail_lines], start=1):
        _safe_addstr(screen, top + offset, left, line[: max(width - 1, 1)])

    _safe_addstr(screen, log_top, left, f"{controller.selected_stream} tail", curses.A_BOLD)
    log_rows = max(height - log_top - 1, 1)
    for offset, line in enumerate(_tail_lines(logs, max_lines=log_rows), start=1):
        _safe_addstr(screen, log_top + offset, left, line[: max(width - 1, 1)], curses.A_DIM)


def _curses_main(screen, controller: TuiController) -> None:
    curses.curs_set(0)
    screen.nodelay(True)
    screen.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)

    controller.refresh()
    last_refresh = time.time()
    while True:
        now = time.time()
        if now - last_refresh >= controller.refresh_seconds:
            controller.refresh()
            last_refresh = now

        height, width = screen.getmaxyx()
        screen.erase()
        top = _render_header(screen, controller.snapshot)
        if height < 12 or width < 80:
            _safe_addstr(screen, top, 0, "Resize the terminal to at least 80x12.", curses.A_BOLD)
            screen.refresh()
        else:
            left_width = max(int(width * 0.42), 30)
            _render_instances(screen, top, left_width, height, controller)
            _render_detail(screen, top, left_width + 2, width - left_width - 2, height, controller)
            screen.refresh()

        key = screen.getch()
        if key == -1:
            time.sleep(0.05)
            continue
        if key in {ord("q"), ord("Q")}:
            return
        if key in {curses.KEY_UP, ord("k"), ord("K")}:
            controller.move(-1)
            controller.refresh()
            last_refresh = time.time()
            continue
        if key in {curses.KEY_DOWN, ord("j"), ord("J")}:
            controller.move(1)
            controller.refresh()
            last_refresh = time.time()
            continue
        if key == ord("\t"):
            controller.toggle_stream()
            controller.refresh()
            last_refresh = time.time()
            continue
        if key in {ord("r"), ord("R")}:
            controller.refresh()
            last_refresh = time.time()


def run_tui(
    *,
    repo_root: str | None = None,
    artifacts_dir: str | None = None,
    refresh_seconds: float = 2.0,
) -> None:
    controller = TuiController(
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        refresh_seconds=refresh_seconds,
    )
    curses.wrapper(_curses_main, controller)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_tui(
        repo_root=args.repo_root,
        artifacts_dir=args.artifacts_dir,
        refresh_seconds=args.refresh_seconds,
    )


if __name__ == "__main__":
    main()
