"""Secure configuration helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, field_validator


class SecureSettings(BaseModel):
    """Security-sensitive runtime settings."""

    database_url: str
    secret_key: str
    api_token: str | None = None

    @field_validator("secret_key")
    @classmethod
    def _validate_secret_key(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError("secret_key must be at least 32 characters")
        return value

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        if not value.startswith(("postgresql://", "sqlite://")):
            raise ValueError("database_url must start with postgresql:// or sqlite://")
        return value

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SecureSettings:
        """Build secure settings from environment variables."""
        source = env if env is not None else os.environ
        return cls(
            database_url=source.get("AI_FACTORY_DATABASE_URL", ""),
            secret_key=source.get("AI_FACTORY_SECRET_KEY", ""),
            api_token=source.get("AI_FACTORY_API_TOKEN"),
        )
