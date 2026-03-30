from typing import Any

from ai_factory.domains.factory import DomainFactory, DomainType


class CodeGenerationDomain:
    """
    Implementation of the Code Generation domain.
    Provides tasks related to programming and code synthesis.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._initialized: bool = False

    @property
    def name(self) -> str:
        """Get the unique name of the domain."""
        return DomainType.CODE_GENERATION.value

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize domain-specific resources with the given configuration."""
        self._config = config
        self._initialized = True

    def execute(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a code generation task.

        Args:
            task: The specific task identifier to execute.
            **kwargs: Additional parameters for the task.

        Returns:
            A dictionary containing the task results.

        Raises:
            RuntimeError: If the domain has not been initialized.
            ValueError: If the task is unknown.
        """
        if not self._initialized:
            raise RuntimeError(f"Domain '{self.name}' must be initialized before execution.")

        if task == "generate_snippet":
            return {"status": "success", "snippet": "print('Hello, World!')", "kwargs": kwargs}

        raise ValueError(f"Unknown task '{task}' in domain '{self.name}'.")


# Automatically register the domain with the factory
DomainFactory.register(DomainType.CODE_GENERATION, CodeGenerationDomain)

__all__ = ["CodeGenerationDomain"]
