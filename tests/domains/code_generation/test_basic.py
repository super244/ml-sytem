"""Test code generation domain functionality."""

import pytest

from ai_factory.domains.code_generation import CodeGenerationDomain
from ai_factory.domains.factory import DomainFactory, DomainType


def test_code_generation_domain_name():
    """Test domain name property."""
    domain = CodeGenerationDomain()
    assert domain.name == DomainType.CODE_GENERATION.value
    assert domain.name == "code_generation"


def test_code_generation_domain_initialization():
    """Test domain initialization."""
    domain = CodeGenerationDomain()
    assert domain._initialized is False

    config = {"language": "python", "max_tokens": 1000}
    domain.initialize(config)

    assert domain._initialized is True
    assert domain._config == config


def test_code_generation_domain_execute_without_init():
    """Test executing without initialization raises error."""
    domain = CodeGenerationDomain()

    with pytest.raises(RuntimeError, match="must be initialized"):
        domain.execute("generate_snippet")


def test_code_generation_domain_execute_unknown_task():
    """Test executing unknown task raises error."""
    domain = CodeGenerationDomain()
    domain.initialize({})

    with pytest.raises(ValueError, match="Unknown task"):
        domain.execute("unknown_task")


def test_code_generation_domain_generate_snippet():
    """Test generate_snippet task."""
    domain = CodeGenerationDomain()
    domain.initialize({"language": "python"})

    result = domain.execute("generate_snippet", prompt="print hello world")

    assert result["status"] == "success"
    assert "snippet" in result
    assert "print" in result["snippet"].lower()


def test_code_generation_domain_execute_with_kwargs():
    """Test execute passes kwargs correctly."""
    domain = CodeGenerationDomain()
    domain.initialize({})

    result = domain.execute("generate_snippet", language="python", complexity="simple")

    assert result["status"] == "success"
    assert "kwargs" in result
    assert result["kwargs"]["language"] == "python"
    assert result["kwargs"]["complexity"] == "simple"


def test_code_generation_domain_factory_registration():
    """Test domain is properly registered with factory."""
    domain = DomainFactory.create(DomainType.CODE_GENERATION)
    assert isinstance(domain, CodeGenerationDomain)
    assert domain.name == "code_generation"


def test_code_generation_domain_multiple_instances():
    """Test creating multiple domain instances."""
    domain1 = CodeGenerationDomain()
    domain2 = CodeGenerationDomain()

    domain1.initialize({"instance": 1})
    domain2.initialize({"instance": 2})

    assert domain1._config["instance"] == 1
    assert domain2._config["instance"] == 2


def test_code_generation_domain_reinitialize():
    """Test reinitializing domain updates config."""
    domain = CodeGenerationDomain()
    domain.initialize({"version": 1})
    assert domain._config["version"] == 1

    domain.initialize({"version": 2})
    assert domain._config["version"] == 2
