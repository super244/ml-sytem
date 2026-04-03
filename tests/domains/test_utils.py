from __future__ import annotations

import pytest

from ai_factory.domains import utils


def test_list_available_domains_contains_mathematics() -> None:
    domains = utils.list_available_domains()
    assert domains
    assert domains[0].name == "mathematics"


def test_get_domain_info_returns_math_payload() -> None:
    payload = utils.get_domain_info("mathematics")
    assert payload["name"] == "mathematics"
    assert "subdomains" in payload
    assert "benchmarks" in payload


def test_get_domain_info_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unknown domain"):
        utils.get_domain_info("unknown")
