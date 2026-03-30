from enum import Enum

from ai_factory.domains.interfaces import DomainInterface


class DomainType(str, Enum):
    """Enumeration of supported domain types in the AI Factory."""

    CODE_GENERATION = "code_generation"
    MATHEMATICS = "mathematics"
    REASONING = "reasoning"
    CREATIVE = "creative"


class DomainFactory:
    """
    Factory for instantiating domain implementations.
    Provides centralized registration and resolution of domains.
    """

    _registry: dict[str, type[DomainInterface]] = {}

    @classmethod
    def register(cls, domain_type: str | DomainType, domain_class: type[DomainInterface]) -> None:
        """
        Register a new domain class.

        Args:
            domain_type: The string or enum identifier for the domain.
            domain_class: The class implementing DomainInterface.
        """
        key = domain_type.value if isinstance(domain_type, DomainType) else str(domain_type)
        cls._registry[key] = domain_class

    @classmethod
    def create(cls, domain_type: str | DomainType) -> DomainInterface:
        """
        Create and return an instance of the specified domain.

        Args:
            domain_type: The string or enum identifier for the domain.

        Returns:
            An instance of the requested domain class.

        Raises:
            ValueError: If the domain type is not registered.
        """
        key = domain_type.value if isinstance(domain_type, DomainType) else str(domain_type)

        if key not in cls._registry:
            raise ValueError(f"Unknown domain type: '{key}'. Available domains: {list(cls._registry.keys())}")

        domain_class = cls._registry[key]
        return domain_class()
