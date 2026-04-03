from __future__ import annotations

import pytest

from ai_factory import cli_scripts


def test_validate_http_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        cli_scripts._validate_http_url("file:///tmp/payload.json")


def test_validate_http_url_requires_host() -> None:
    with pytest.raises(ValueError, match="must include a host"):
        cli_scripts._validate_http_url("http:///missing-host")


def test_request_json_rejects_invalid_scheme_before_network_call() -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        cli_scripts._request_json("GET", "ftp://example.com/data")
