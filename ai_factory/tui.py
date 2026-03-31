from __future__ import annotations

import argparse
import curses
import locale
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_factory.core.platform.container import build_platform_container
from ai_factory.titan import detect_titan_status

locale.setlocale(locale.LC_ALL, "")

_STATUS_ICONS = {"running": "*", "completed": "+", "failed": "!", "pending": "~"}
_ACTION_KEYS = {
    "e": "evaluate",
    "d": "deploy",
    "i": "inference",
}


@dataclass
class DashboardSnapshot:
    instances: list[Any]
    summary: dict[str, Any]
    logs: dict[str, str]
    metrics: dict[str, Any]
    titan: dict[str, Any]
    selected_stream: str
    loaded_at: float
    available_actions: list[str] = field(default_factory=list)
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


def _safe_addstr(screen: Any, y: int, x: int, text: str, attr: int = 0) -> None:
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


def _hline(screen: Any, y: int, x: int, width: int, attr: int = 0) -> None:
    try:
        screen.hline(y, x, curses.ACS_HLINE, width, attr)
    except curses.error:
        pass


def _vline(screen: Any, y: int, x: int, height: int, attr: int = 0) -> None:
    for row in range(height):
        try:
            screen.addch(y + row, x, curses.ACS_VLINE, attr)
        except curses.error:
            pass


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "n/a"
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).strftime("%H:%M:%S")
    except ValueError:
        return value


def _format_timestamp_full(value: str | None) -> str:
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


def _progress_bar(percent: float | None, width: int = 12) -> str:
    if percent is None or not isinstance(percent, (int, float)):
        return "[" + "." * width + "]"
    filled = int(min(max(percent, 0.0), 1.0) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _tail_lines(text: str, *, max_lines: int) -> list[str]:
    lines = text.splitlines() or [""]
    return lines[-max_lines:]


def _wrap_lines(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text]
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        chunks = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped.extend(chunks or [""])
    return wrapped or [""]


class TuiController:
    def __init__(self, *, repo_root: str | None, artifacts_dir: str | None, refresh_seconds: float):
        self.control = build_platform_container(
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        ).control_service
        self.titan = detect_titan_status(repo_root)
        self.refresh_seconds = max(refresh_seconds, 0.5)
        self.selected_index = 0
        self.selected_stream = "stdout"
        self.show_help = False
        self.last_action_msg: str | None = None
        self.last_action_time: float = 0.0
        self.snapshot = DashboardSnapshot(
            instances=[],
            summary={},
            logs={"stdout": "", "stderr": ""},
            metrics={"summary": {}, "points": []},
            titan=self.titan,
            selected_stream=self.selected_stream,
            loaded_at=0.0,
        )

    def refresh(self) -> None:
        try:
            instances = self.control.list_instances()
            available_actions: list[str] = []
            if instances:
                self.selected_index = max(0, min(self.selected_index, len(instances) - 1))
                selected = instances[self.selected_index]
                live = self.control.get_live_instance_snapshot(selected.id)
                logs = live.logs.model_dump(mode="json")
                metrics = live.metrics.model_dump(mode="json")
                available_actions = [
                    action.get("label", action.get("action", "")) for action in (live.available_actions or [])
                ]
            else:
                self.selected_index = 0
                logs = {"stdout": "", "stderr": ""}
                metrics = {"summary": {}, "points": []}
            self.snapshot = DashboardSnapshot(
                instances=instances,
                summary=self.control.monitoring_summary(),
                logs=logs,
                metrics=metrics,
                titan=self.titan,
                selected_stream=self.selected_stream,
                loaded_at=time.time(),
                available_actions=available_actions,
            )
        except Exception as exc:
            self.snapshot = DashboardSnapshot(
                instances=self.snapshot.instances,
                summary=self.snapshot.summary,
                logs=self.snapshot.logs,
                metrics=self.snapshot.metrics,
                titan=self.titan,
                selected_stream=self.selected_stream,
                loaded_at=time.time(),
                available_actions=self.snapshot.available_actions,
                error=str(exc),
            )

    def move(self, delta: int) -> None:
        if not self.snapshot.instances:
            return
        self.selected_index = max(0, min(self.selected_index + delta, len(self.snapshot.instances) - 1))

    def toggle_stream(self) -> None:
        self.selected_stream = "stderr" if self.selected_stream == "stdout" else "stdout"

    def execute_action(self, action_key: str) -> None:
        action_name = _ACTION_KEYS.get(action_key)
        if not action_name or not self.snapshot.instances:
            return
        instance = self.snapshot.instances[self.selected_index]
        try:
            self.control.execute_action(instance.id, action_name)
            self.last_action_msg = f"{action_name} triggered on {instance.name}"
        except Exception as exc:
            self.last_action_msg = f"Error: {exc}"
        self.last_action_time = time.time()

    @property
    def selected_instance(self) -> Any | None:
        if not self.snapshot.instances:
            return None
        return self.snapshot.instances[self.selected_index]


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


def _render_header(screen: Any, snapshot: DashboardSnapshot, width: int) -> int:
    summary = snapshot.summary or {}
    counts = summary.get("task_status_counts") or {}
    left = f" AI-FACTORY  instances:{len(snapshot.instances)}  active:{summary.get('active_runs', 0)}"
    right = (
        f"runs:{summary.get('runs', 0)}  "
        f"ok:{counts.get('completed', 0)}  "
        f"fail:{counts.get('failed', 0)}  "
        f"retry:{counts.get('retry_waiting', 0)}  "
        f"{snapshot.selected_stream}  "
        f"{time.strftime('%H:%M:%S', time.localtime(snapshot.loaded_at or time.time()))}"
    )
    pad = max(width - len(left) - len(right) - 1, 1)
    _safe_addstr(screen, 0, 0, (left + " " * pad + right).ljust(width), curses.A_REVERSE | curses.A_BOLD)

    if snapshot.error:
        error_color = curses.color_pair(4) if curses.has_colors() else curses.A_BOLD
        _safe_addstr(screen, 1, 1, f"ERROR: {snapshot.error}", error_color)
        return 3
    return 1


def _render_statusbar(screen: Any, height: int, width: int, controller: TuiController) -> None:
    now = time.time()
    titan = controller.snapshot.titan or {}
    titan_summary = (
        f" Titan:{titan.get('mode', 'n/a')} | HW:{titan.get('silicon', 'n/a')} | "
        f"BW:{titan.get('bandwidth_gbps') or 'n/a'} GB/s"
    )
    if controller.last_action_msg and (now - controller.last_action_time) < 5.0:
        msg = f" {controller.last_action_msg}{titan_summary}"
    else:
        msg = f" q:quit  j/k:move  tab:stream  r:refresh  e:eval  d:deploy  i:infer  ?:help{titan_summary}"
    _safe_addstr(screen, height - 1, 0, msg.ljust(width), curses.A_REVERSE)


def _render_help_overlay(screen: Any, height: int, width: int) -> None:
    lines = [
        "KEYBOARD SHORTCUTS",
        "",
        "  q / Q          Quit",
        "  j / DOWN       Move down",
        "  k / UP         Move up",
        "  g              Jump to top",
        "  G              Jump to bottom",
        "  TAB            Toggle stdout / stderr",
        "  r / R          Force refresh",
        "  e              Evaluate selected instance",
        "  d              Deploy selected instance",
        "  i              Open inference for selected",
        "  ?              Toggle this help",
        "",
        "Press any key to close.",
    ]
    box_w = min(50, width - 4)
    box_h = min(len(lines) + 2, height - 4)
    start_y = max((height - box_h) // 2, 1)
    start_x = max((width - box_w) // 2, 1)
    for row in range(box_h):
        _safe_addstr(screen, start_y + row, start_x, " " * box_w, curses.A_REVERSE)
    for idx, line in enumerate(lines[: box_h - 2]):
        _safe_addstr(screen, start_y + 1 + idx, start_x + 1, line[: box_w - 2], curses.A_REVERSE)


def _render_instances(screen: Any, top: int, width: int, height: int, controller: TuiController) -> int:
    _safe_addstr(screen, top, 1, "INSTANCES", curses.A_BOLD)
    instances = controller.snapshot.instances
    if not instances:
        _safe_addstr(screen, top + 1, 1, "No instances yet.", curses.A_DIM)
        _safe_addstr(screen, top + 2, 1, "ai-factory new --config ...", curses.A_DIM)
        return top + 3
    col_header = f"{'NAME':<16} {'TYPE':<8} {'STATUS':<8} {'PROGRESS':<14} {'UPDATED':<9}"
    _safe_addstr(screen, top + 1, 1, col_header[: width - 2], curses.A_DIM)
    available_rows = max(height - top - 5, 1)
    start = max(0, controller.selected_index - available_rows + 1)
    rows = instances[start : start + available_rows]
    for offset, instance in enumerate(rows):
        row = top + 2 + offset
        actual_idx = start + offset
        selected = actual_idx == controller.selected_index
        icon = _STATUS_ICONS.get(instance.status, " ")
        progress_str = "n/a"
        if instance.progress and isinstance(instance.progress.percent, (int, float)):
            bar = _progress_bar(instance.progress.percent, width=8)
            progress_str = f"{bar} {instance.progress.percent * 100:.0f}%"
        elif instance.progress:
            progress_str = instance.progress.stage[:14]
        updated = _format_timestamp(instance.updated_at)
        line = f"{icon} {instance.name:<15} {instance.type:<8} {instance.status:<8} {progress_str:<14} {updated}"
        attr = curses.A_REVERSE if selected else _status_attr(instance.status)
        _safe_addstr(screen, row, 1, line[: width - 2], attr)
    bottom = top + 2 + len(rows) + 1
    if len(instances) > available_rows:
        _safe_addstr(screen, bottom - 1, 1, f"  [{start + 1}-{start + len(rows)} of {len(instances)}]", curses.A_DIM)
    return bottom


def _render_detail(screen: Any, top: int, left: int, width: int, height: int, controller: TuiController) -> int:
    snapshot = controller.snapshot
    if not snapshot.instances:
        _safe_addstr(screen, top, left, "DETAIL", curses.A_BOLD)
        _safe_addstr(screen, top + 1, left, "Select an instance.", curses.A_DIM)
        return top + 2
    instance = snapshot.instances[controller.selected_index]
    metrics_summary = snapshot.metrics.get("summary") or {}
    dw = max(width - 1, 1)

    _safe_addstr(screen, top, left, f"DETAIL: {instance.id[:24]}", curses.A_BOLD)
    row = top + 1

    kv_pairs = [
        ("name", instance.name),
        ("type", f"{instance.type} / {instance.status}"),
        ("env", instance.environment.kind),
        ("lifecycle", (instance.lifecycle.stage if instance.lifecycle and instance.lifecycle.stage else "n/a")),
        ("origin", ((instance.lifecycle.origin or "n/a") if instance.lifecycle else "n/a")),
        ("mode", ((instance.lifecycle.learning_mode or "n/a") if instance.lifecycle else "n/a")),
    ]
    if instance.lifecycle and instance.lifecycle.source_model:
        kv_pairs.append(("model", instance.lifecycle.source_model))
    if instance.parent_instance_id:
        kv_pairs.append(("parent", instance.parent_instance_id[:20]))
    kv_pairs.append(("updated", _format_timestamp_full(instance.updated_at)))

    for key, val in kv_pairs:
        if row >= height - 8:
            break
        label = f"  {key}: {val}"
        _safe_addstr(screen, row, left, label[:dw])
        row += 1

    if instance.progress:
        row += 1
        _safe_addstr(screen, row, left, "PROGRESS", curses.A_BOLD)
        row += 1
        bar = _progress_bar(instance.progress.percent, width=20)
        pct = (
            f"{instance.progress.percent * 100:.0f}%" if isinstance(instance.progress.percent, (int, float)) else "n/a"
        )
        _safe_addstr(screen, row, left, f"  {bar} {pct}  {instance.progress.stage}")
        row += 1
        if instance.progress.status_message:
            _safe_addstr(screen, row, left, f"  {instance.progress.status_message}"[:dw], curses.A_DIM)
            row += 1
        if instance.progress.eta_seconds and isinstance(instance.progress.eta_seconds, (int, float)):
            eta_m = int(instance.progress.eta_seconds) // 60
            eta_s = int(instance.progress.eta_seconds) % 60
            _safe_addstr(screen, row, left, f"  ETA: {eta_m}m {eta_s}s", curses.A_DIM)
            row += 1

    metric_keys = [
        ("accuracy", True),
        ("parse_rate", True),
        ("avg_latency_s", False),
        ("loss", False),
        ("perplexity", False),
        ("latest_step", False),
    ]
    visible_metrics = [(k, p) for k, p in metric_keys if _metric(metrics_summary, k) is not None]
    if visible_metrics:
        row += 1
        _safe_addstr(screen, row, left, "METRICS", curses.A_BOLD)
        row += 1
        for mk, is_pct in visible_metrics:
            if row >= height - 4:
                break
            val = _format_metric(_metric(metrics_summary, mk), percent=is_pct)
            _safe_addstr(screen, row, left, f"  {mk}: {val}")
            row += 1

    if instance.decision:
        row += 1
        _safe_addstr(screen, row, left, "DECISION", curses.A_BOLD)
        row += 1
        _safe_addstr(screen, row, left, f"  {instance.decision.action} ({instance.decision.rule})")
        row += 1

    if instance.error:
        row += 1
        err_attr = curses.color_pair(4) if curses.has_colors() else curses.A_BOLD
        _safe_addstr(screen, row, left, "ERROR", curses.A_BOLD | err_attr)
        row += 1
        _safe_addstr(screen, row, left, f"  {instance.error.code}: {instance.error.message}"[:dw], err_attr)
        row += 1

    return row


def _render_logs(screen: Any, top: int, left: int, width: int, height: int, controller: TuiController) -> None:
    snapshot = controller.snapshot
    logs = snapshot.logs.get(controller.selected_stream, "")
    dw = max(width - 1, 1)
    _safe_addstr(screen, top, left, f"LOG ({controller.selected_stream})", curses.A_BOLD)
    log_rows = max(height - top - 2, 1)
    for offset, line in enumerate(_tail_lines(logs, max_lines=log_rows), start=1):
        if top + offset >= height - 1:
            break
        _safe_addstr(screen, top + offset, left, line[:dw], curses.A_DIM)


def _render_recommendations(
    screen: Any,
    top: int,
    left: int,
    width: int,
    height: int,
    controller: TuiController,
) -> None:
    _safe_addstr(screen, top, left, "ACTIONS & NEXT STEPS", curses.A_BOLD)
    row = top + 1
    dw = max(width - 1, 1)

    avail = controller.snapshot.available_actions
    if avail:
        _safe_addstr(screen, row, left, "available:", curses.A_DIM)
        row += 1
        for action in avail[:6]:
            if row >= height - 1:
                break
            _safe_addstr(screen, row, left, f"  {action}"[:dw])
            row += 1
        row += 1

    instance = controller.selected_instance
    if instance and instance.recommendations:
        _safe_addstr(screen, row, left, "recommended:", curses.A_DIM)
        row += 1
        for _idx, rec in enumerate(instance.recommendations[:5]):
            if row >= height - 2:
                break
            priority_str = "*" * min(rec.priority, 3)
            _safe_addstr(screen, row, left, f"  {priority_str} {rec.action}"[:dw])
            row += 1
            reason_lines = _wrap_lines(rec.reason, dw - 4)
            for rline in reason_lines[:2]:
                if row >= height - 2:
                    break
                _safe_addstr(screen, row, left, f"    {rline}"[:dw], curses.A_DIM)
                row += 1
        row += 1

    if instance and instance.task_summary:
        _safe_addstr(screen, row, left, "tasks:", curses.A_DIM)
        row += 1
        for key, value in list(instance.task_summary.items())[:4]:
            if row >= height - 1:
                break
            _safe_addstr(screen, row, left, f"  {key}: {_format_metric(value)}"[:dw])
            row += 1


def _curses_main(screen: Any, controller: TuiController) -> None:
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
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)

    controller.refresh()
    last_refresh = time.time()
    while True:
        now = time.time()
        if now - last_refresh >= controller.refresh_seconds:
            controller.refresh()
            last_refresh = now

        height, width = screen.getmaxyx()
        screen.erase()

        if height < 10 or width < 60:
            _safe_addstr(screen, 0, 0, "Resize terminal to at least 60x10.", curses.A_BOLD)
            screen.refresh()
            key = screen.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key in {ord("q"), ord("Q")}:
                return
            continue

        top = _render_header(screen, controller.snapshot, width)
        _render_statusbar(screen, height, width, controller)

        if controller.show_help:
            _render_help_overlay(screen, height, width)
            screen.refresh()
            key = screen.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            controller.show_help = False
            continue

        usable_height = height - 1
        if width >= 120:
            col1_w = max(int(width * 0.35), 30)
            col2_w = max(int(width * 0.35), 30)
            col3_w = width - col1_w - col2_w - 2

            inst_bottom = _render_instances(screen, top + 1, col1_w - 1, usable_height, controller)
            _vline(screen, top + 1, col1_w, usable_height - top - 2, curses.A_DIM)

            detail_left = col1_w + 1
            detail_bottom = _render_detail(screen, top + 1, detail_left, col2_w - 1, usable_height, controller)
            log_space = usable_height - detail_bottom - 1
            if log_space > 3:
                _render_logs(screen, detail_bottom + 1, detail_left, col2_w - 1, usable_height, controller)

            _vline(screen, top + 1, col1_w + col2_w, usable_height - top - 2, curses.A_DIM)
            rec_left = col1_w + col2_w + 1
            _render_recommendations(screen, top + 1, rec_left, col3_w, usable_height, controller)
        elif width >= 80:
            col1_w = max(int(width * 0.42), 30)
            col2_w = width - col1_w - 1

            inst_bottom = _render_instances(screen, top + 1, col1_w - 1, usable_height, controller)
            _vline(screen, top + 1, col1_w, usable_height - top - 2, curses.A_DIM)

            detail_left = col1_w + 1
            detail_bottom = _render_detail(screen, top + 1, detail_left, col2_w, usable_height, controller)
            log_space = usable_height - detail_bottom - 1
            if log_space > 3:
                _render_logs(screen, detail_bottom + 1, detail_left, col2_w, usable_height, controller)
        else:
            inst_bottom = _render_instances(screen, top + 1, width - 1, usable_height, controller)
            remaining = usable_height - inst_bottom - 1
            if remaining > 4:
                _render_detail(screen, inst_bottom, 1, width - 2, usable_height, controller)

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
        if key in {
            ord("g"),
        }:
            controller.selected_index = 0
            controller.refresh()
            last_refresh = time.time()
            continue
        if key in {
            ord("G"),
        }:
            controller.selected_index = max(len(controller.snapshot.instances) - 1, 0)
            controller.refresh()
            last_refresh = time.time()
            continue
        if key in {ord("r"), ord("R")}:
            controller.refresh()
            last_refresh = time.time()
            continue
        if key == ord("?"):
            controller.show_help = True
            continue
        if chr(key) in _ACTION_KEYS if 0 <= key < 256 else False:
            controller.execute_action(chr(key))
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
