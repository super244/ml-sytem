"""Unified interface layer for AI-Factory.

This package provides consistent interfaces across all interaction methods:
- CLI: Command-line interface for automation and scripting
- TUI: Terminal user interface for interactive system management
- Web: Browser-based dashboard for monitoring and control
- Desktop: Native desktop application for full visual control

All interfaces connect to the same backend services and provide
consistent functionality and user experience.
"""

from .cli import CLIInterface
from .desktop import DesktopInterface
from .tui import TUIInterface
from .web import WebInterface

__all__ = ["CLIInterface", "TUIInterface", "WebInterface", "DesktopInterface"]
