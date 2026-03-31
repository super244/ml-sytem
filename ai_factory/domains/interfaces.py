from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DomainInterface(Protocol):
    """
    Protocol defining the base interface for AI Factory domains.
    Enterprise code quality requires clear contracts for domain implementations.
    """

    @property
    def name(self) -> str:
        """Get the unique name of the domain."""
        ...

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize domain-specific resources with the given configuration."""
        ...

    def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a domain-specific task.

        Args:
            task: The specific task identifier to execute.
            **kwargs: Additional parameters for the task.

        Returns:
            A dictionary containing the task results.
        """
        ...
