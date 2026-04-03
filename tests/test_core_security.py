from __future__ import annotations

import pytest

from ai_factory.core.security import SecureExecutor, SecureHasher, SecureSettings


def test_secure_hasher_round_trip_verification() -> None:
    digest, salt = SecureHasher.hash_data("payload")
    assert SecureHasher.verify_data("payload", digest, salt) is True
    assert SecureHasher.verify_data("other", digest, salt) is False


def test_secure_hasher_derive_key_is_stable_with_same_salt() -> None:
    salt = b"0123456789abcdef"
    key1, used_salt_1 = SecureHasher.derive_key("secret", salt=salt)
    key2, used_salt_2 = SecureHasher.derive_key("secret", salt=salt)
    assert used_salt_1 == used_salt_2 == salt
    assert key1 == key2


def test_secure_settings_from_env_and_validation() -> None:
    settings = SecureSettings.from_env(
        {
            "AI_FACTORY_DATABASE_URL": "sqlite:///tmp.db",
            "AI_FACTORY_SECRET_KEY": "x" * 32,
            "AI_FACTORY_API_TOKEN": "token",
        }
    )
    assert settings.database_url.startswith("sqlite://")
    assert settings.api_token == "token"

    with pytest.raises(ValueError, match="secret_key must be at least 32"):
        SecureSettings(database_url="sqlite:///tmp.db", secret_key="short")

    with pytest.raises(ValueError, match="database_url must start"):
        SecureSettings(database_url="http://example.com", secret_key="x" * 32)


def test_secure_executor_runs_allowed_command() -> None:
    result = SecureExecutor.execute_command("python -c \"print('ok')\"", timeout=5)
    assert result.returncode == 0
    assert (result.stdout or "").strip() == "ok"


def test_secure_executor_rejects_disallowed_command() -> None:
    with pytest.raises(ValueError, match="not in allowed list"):
        SecureExecutor.execute_command("rm -rf /", timeout=1)
