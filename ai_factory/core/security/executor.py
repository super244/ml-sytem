"""Secure subprocess execution for AI-Factory."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


class SecureExecutor:
    """
    Secure subprocess execution with command whitelisting.

    This class provides a safe way to execute external commands
    with proper validation and timeout handling.
    """

    ALLOWED_COMMANDS = {
        "python",
        "python3",
        "pip",
        "pip3",
        "git",
        "docker",
        "npm",
        "node",
        "make",
        "pytest",
        "ruff",
        "mypy",
        "bandit",
    }

    @classmethod
    def execute_command(
        cls,
        command: str | Sequence[str],
        timeout: float = 30.0,
        capture_output: bool = True,
        check: bool = True,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a command with security constraints.

        Args:
            command: The command to execute.
            timeout: Timeout in seconds.
            capture_output: Whether to capture stdout/stderr.
            check: Whether to raise on non-zero exit code.

        Returns:
            CompletedProcess instance.

        Raises:
            ValueError: If command is not in allowed list.
            TimeoutError: If command execution times out.
            subprocess.CalledProcessError: If command fails and check=True.
        """
        cmd_parts = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
        if not cmd_parts:
            raise ValueError("Empty command")

        base_cmd = Path(cmd_parts[0]).name
        if base_cmd not in cls.ALLOWED_COMMANDS:
            raise ValueError(
                f"Command '{base_cmd}' not in allowed list. Allowed commands: {sorted(cls.ALLOWED_COMMANDS)}"
            )

        resolved = shutil.which(base_cmd)
        if resolved is None:
            raise ValueError(f"Command '{base_cmd}' is not available in PATH")

        safe_cmd = [resolved, *cmd_parts[1:]]

        try:
            result = subprocess.run(  # nosec B603 - command is allowlisted and shell=False
                safe_cmd,
                shell=False,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=check,
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env else None,
            )
            return result

        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Command execution timed out after {timeout}s") from e

        except subprocess.CalledProcessError:
            raise

    @classmethod
    def add_allowed_command(cls, command: str) -> None:
        """
        Add a command to the allowed list.

        Args:
            command: The command name to allow.
        """
        cls.ALLOWED_COMMANDS.add(command)

    @classmethod
    def remove_allowed_command(cls, command: str) -> None:
        """
        Remove a command from the allowed list.

        Args:
            command: The command name to remove.
        """
        cls.ALLOWED_COMMANDS.discard(command)
