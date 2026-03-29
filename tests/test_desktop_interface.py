from __future__ import annotations

from ai_factory.interfaces.desktop import main as desktop_main
from ai_factory.interfaces.desktop.main import DesktopInterface


def test_desktop_interface_prefers_native_shell_when_requested(tmp_path, monkeypatch):
    native_dir = tmp_path / "desktop" / "macos" / "AIFactoryNative"
    native_dir.mkdir(parents=True)

    interface = DesktopInterface(repo_root=tmp_path)
    monkeypatch.setenv("AI_FACTORY_DESKTOP_NATIVE", "1")
    monkeypatch.setattr(desktop_main.sys, "platform", "darwin", raising=False)

    captured: dict[str, bool] = {}

    monkeypatch.setattr(interface, "run_native", lambda: captured.setdefault("native", True))

    interface.run()

    assert captured["native"] is True
