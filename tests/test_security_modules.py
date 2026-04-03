from __future__ import annotations

import shutil

import pytest

from ai_factory.core.security import SecureExecutor, SecureHasher, SecureSettings


def test_secure_hasher_round_trip() -> None:
    digest, salt = SecureHasher.hash_data("hello-world")
    assert digest
    assert salt
    assert SecureHasher.verify_data("hello-world", digest, salt) is True
    assert SecureHasher.verify_data("different", digest, salt) is False


def test_secure_settings_from_env_valid() -> None:
    settings = SecureSettings.from_env(
        {
            "AI_FACTORY_DATABASE_URL": "sqlite:///tmp.db",
            "AI_FACTORY_SECRET_KEY": "x" * 32,
            "AI_FACTORY_API_TOKEN": "token",
        }
    )
    assert settings.database_url == "sqlite:///tmp.db"
    assert settings.api_token == "token"


def test_secure_settings_validation_errors() -> None:
    with pytest.raises(ValueError, match="secret_key"):
        SecureSettings(database_url="sqlite:///tmp.db", secret_key="short")

    with pytest.raises(ValueError, match="database_url"):
        SecureSettings(database_url="mysql://localhost/db", secret_key="x" * 32)


def test_secure_executor_allows_whitelisted_commands() -> None:
    python_bin = "python3" if shutil.which("python3") else "python"
    result = SecureExecutor.execute_command([python_bin, "-c", "print('ok')"])
    assert result.stdout.strip() == "ok"


def test_secure_executor_blocks_non_whitelisted_commands() -> None:
    with pytest.raises(ValueError, match="not in allowed list"):
        SecureExecutor.execute_command("echo hello")
