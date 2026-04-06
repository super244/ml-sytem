"""AI-Factory core exception hierarchy.

This module defines the complete exception hierarchy for the AI-Factory platform,
providing structured error information for debugging, monitoring, and user feedback.
"""

from __future__ import annotations

from typing import Any


class AIFactoryError(Exception):
    """Base exception for all AI-Factory errors.

    Provides structured error information including error codes, contexts,
    and suggestions for resolution.
    """

    error_code: str = "UNKNOWN_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.error_code
        self.context = context or {}
        self.suggestion = suggestion

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
            "type": self.__class__.__name__,
        }


# =============================================================================
# Resource Not Found Errors
# =============================================================================


class ResourceNotFoundError(AIFactoryError):
    """Base class for resource not found errors."""

    error_code = "RESOURCE_NOT_FOUND"


class JobNotFound(ResourceNotFoundError):
    """Raised when a job/orchestration run cannot be found."""

    error_code = "JOB_NOT_FOUND"

    def __init__(
        self,
        job_id: str,
        *,
        available_jobs: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"job_id": job_id, **(context or {})}
        if available_jobs:
            ctx["available_jobs"] = available_jobs[:10]

        super().__init__(
            f"Job not found: {job_id}",
            error_code=self.error_code,
            context=ctx,
            suggestion="Verify the job ID or list active jobs with 'ai-factory jobs list'",
        )


class DatasetNotFound(ResourceNotFoundError):
    """Raised when a dataset cannot be found."""

    error_code = "DATASET_NOT_FOUND"

    def __init__(
        self,
        dataset_id: str,
        *,
        dataset_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"dataset_id": dataset_id, **(context or {})}
        if dataset_path:
            ctx["dataset_path"] = dataset_path

        super().__init__(
            f"Dataset not found: {dataset_id}",
            error_code=self.error_code,
            context=ctx,
            suggestion="Verify the dataset exists. Run 'ai-factory data list' to see available datasets.",
        )


class ModelNotFound(ResourceNotFoundError):
    """Raised when a model cannot be found."""

    error_code = "MODEL_NOT_FOUND"

    def __init__(
        self,
        model_id: str,
        *,
        model_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"model_id": model_id, **(context or {})}
        if model_path:
            ctx["model_path"] = model_path

        super().__init__(
            f"Model not found: {model_id}",
            error_code=self.error_code,
            context=ctx,
            suggestion="Verify the model is trained and packaged. Check 'ai-factory models list'.",
        )


class SearchNotFound(ResourceNotFoundError):
    """Raised when a search operation returns no results."""

    error_code = "SEARCH_NOT_FOUND"

    def __init__(
        self,
        query: str,
        *,
        search_type: str = "general",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"No results found for search: '{query}'",
            error_code=self.error_code,
            context={"query": query, "search_type": search_type, **(context or {})},
            suggestion="Try broadening your search terms or verify the data exists.",
        )


class ConfigNotFound(ResourceNotFoundError):
    """Raised when a configuration file cannot be found."""

    error_code = "CONFIG_NOT_FOUND"

    def __init__(
        self,
        config_path: str,
        *,
        config_type: str = "general",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Configuration file not found: {config_path}",
            error_code=self.error_code,
            context={"config_path": config_path, "config_type": config_type, **(context or {})},
            suggestion=f"Verify the {config_type} config path exists and is readable.",
        )


# =============================================================================
# Infrastructure Errors
# =============================================================================


class InfrastructureError(AIFactoryError):
    """Base class for infrastructure-related errors."""

    error_code = "INFRASTRUCTURE_ERROR"


class ClusterError(InfrastructureError):
    """Raised when cluster operations fail."""

    error_code = "CLUSTER_ERROR"

    def __init__(
        self,
        message: str,
        *,
        cluster_id: str | None = None,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if cluster_id:
            ctx["cluster_id"] = cluster_id
        if operation:
            ctx["operation"] = operation

        super().__init__(
            message,
            error_code=self.error_code,
            context=ctx,
            suggestion="Check cluster health and connectivity. Verify cluster credentials.",
        )


class GPUError(InfrastructureError):
    """Raised when GPU operations fail."""

    error_code = "GPU_ERROR"

    def __init__(
        self,
        message: str,
        *,
        gpu_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if gpu_id is not None:
            ctx["gpu_id"] = gpu_id

        super().__init__(
            message,
            error_code=self.error_code,
            context=ctx,
            suggestion="Check GPU availability with 'nvidia-smi'. Ensure CUDA/Metal drivers are installed.",
        )


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AIFactoryError):
    """Base class for validation errors."""

    error_code = "VALIDATION_ERROR"


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""

    error_code = "CONFIG_VALIDATION_ERROR"


class SchemaValidationError(ValidationError):
    """Raised when data schema validation fails."""

    error_code = "SCHEMA_VALIDATION_ERROR"


# =============================================================================
# Runtime Errors
# =============================================================================


class RuntimeError(AIFactoryError):
    """Base class for runtime errors."""

    error_code = "RUNTIME_ERROR"


class TimeoutError(RuntimeError):
    """Raised when an operation times out."""

    error_code = "TIMEOUT_ERROR"

    def __init__(
        self,
        operation: str,
        *,
        timeout_seconds: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"operation": operation, **(context or {})}
        if timeout_seconds:
            ctx["timeout_seconds"] = timeout_seconds

        message = f"Operation timed out: {operation}"
        if timeout_seconds:
            message += f" (limit: {timeout_seconds}s)"

        super().__init__(
            message,
            error_code=self.error_code,
            context=ctx,
            suggestion="Increase timeout limit, optimize operation, or check resource availability.",
        )


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open."""

    error_code = "CIRCUIT_BREAKER_OPEN"

    def __init__(
        self,
        service: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Circuit breaker is open for service: {service}",
            error_code=self.error_code,
            context={"service": service, **(context or {})},
            suggestion="Wait for circuit to reset or check service health.",
        )


# =============================================================================
# Security Errors
# =============================================================================


class SecurityError(AIFactoryError):
    """Base class for security-related errors."""

    error_code = "SECURITY_ERROR"


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""

    error_code = "AUTHENTICATION_ERROR"

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=self.error_code,
            context=context,
            suggestion="Verify credentials are correct and not expired.",
        )


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""

    error_code = "AUTHORIZATION_ERROR"

    def __init__(
        self,
        resource: str,
        *,
        required_permission: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = {"resource": resource, **(context or {})}
        if required_permission:
            ctx["required_permission"] = required_permission

        super().__init__(
            f"Insufficient permissions for resource: {resource}",
            error_code=self.error_code,
            context=ctx,
            suggestion="Request appropriate permissions from your administrator.",
        )


# =============================================================================
# API Errors
# =============================================================================


class APIError(AIFactoryError):
    """Base class for API-related errors."""

    error_code = "API_ERROR"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        ctx = context or {}
        if status_code:
            ctx["status_code"] = status_code
        if endpoint:
            ctx["endpoint"] = endpoint

        super().__init__(
            message,
            error_code=error_code or self.error_code,
            context=ctx,
            suggestion=suggestion,
        )


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        *,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if retry_after:
            ctx["retry_after"] = retry_after

        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds."

        super().__init__(
            message,
            error_code=self.error_code,
            context=ctx,
            suggestion="Reduce request frequency or implement exponential backoff.",
        )


# =============================================================================
# Data Processing Errors
# =============================================================================


class DataProcessingError(AIFactoryError):
    """Base class for data processing errors."""

    error_code = "DATA_PROCESSING_ERROR"


class ParseError(DataProcessingError):
    """Raised when data parsing fails."""

    error_code = "PARSE_ERROR"


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(AIFactoryError):
    """Base class for training-related errors."""

    error_code = "TRAINING_ERROR"


class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail."""

    error_code = "CHECKPOINT_ERROR"


class OOMError(TrainingError):
    """Raised when out-of-memory occurs during training."""

    error_code = "OUT_OF_MEMORY"

    def __init__(
        self,
        *,
        requested_memory_gb: float | None = None,
        available_memory_gb: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if requested_memory_gb:
            ctx["requested_memory_gb"] = requested_memory_gb
        if available_memory_gb:
            ctx["available_memory_gb"] = available_memory_gb

        super().__init__(
            "Out of memory during training",
            error_code=self.error_code,
            context=ctx,
            suggestion="Reduce batch size, enable gradient checkpointing, or use quantization.",
        )


# =============================================================================
# Legacy Aliases (for backward compatibility)
# =============================================================================

AIFactoryError = AIFactoryError  # Self-reference for explicit imports
