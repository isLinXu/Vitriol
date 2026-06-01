"""
Custom exceptions for Vitriol.

This module provides a hierarchy of exceptions for better error handling
and user-friendly error messages.
"""


class VitriolError(Exception):
    """Base exception for all Vitriol errors."""

    def __init__(self, message: str, recoverable: bool = False):
        """
        Initialize Vitriol error.

        Args:
            message: Error message
            recoverable: Whether the error is potentially recoverable
        """
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class MissingOptionalDependencyError(VitriolError, ImportError):
    """An optional third-party dependency required for a feature is not installed.

    Subclasses both :class:`VitriolError` and :class:`ImportError` so existing
    ``except ImportError`` handlers keep working while callers get an actionable,
    install-oriented message instead of a bare ``ModuleNotFoundError``.
    """

    def __init__(self, package: str, *, feature: str = None, extra: str = None):
        target = f"feature '{feature}'" if feature else f"'{package}'"
        message = (
            f"The optional dependency '{package}' is required to use {target}, "
            "but it is not installed."
        )
        hints = [f"pip install {package}"]
        if extra:
            hints.append(f"pip install 'vitriol[{extra}]'")
        message += "\n\nInstall it via:\n" + "\n".join(f"  • {h}" for h in hints)
        super().__init__(message, recoverable=True)
        self.package = package
        self.feature = feature
        self.extra = extra


class ConfigError(VitriolError):
    """Base class for configuration-related errors."""
    pass


class ConfigLoadError(ConfigError):
    """Failed to load model configuration."""

    def __init__(self, model_id: str, reason: str = ""):
        message = f"Failed to load configuration for '{model_id}'"
        if reason:
            message += f": {reason}"
        message += "\n\nSuggestions:\n"
        message += "  • Check your network connection\n"
        message += "  • Verify the model ID is correct\n"
        message += "  • Try using a mirror source (e.g., ModelScope for China)"
        super().__init__(message, recoverable=True)


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    def __init__(self, config_attr: str, reason: str):
        message = f"Invalid configuration attribute '{config_attr}': {reason}"
        super().__init__(message, recoverable=False)


class ModelError(VitriolError):
    """Base class for model-related errors."""
    pass


class ModelBuildError(ModelError):
    """Failed to build model from configuration."""

    def __init__(self, model_id: str, reason: str = ""):
        message = f"Failed to build model from '{model_id}'"
        if reason:
            message += f": {reason}"
        message += "\n\nThis might be due to:\n"
        message += "  • Missing model implementation in transformers\n"
        message += "  • Incompatible transformers version\n"
        message += "  • Unsupported architecture type"
        super().__init__(message, recoverable=False)


class WeightGenerationError(VitriolError):
    """Failed to generate weights."""

    def __init__(self, param_name: str, reason: str = ""):
        message = f"Failed to generate weight for '{param_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, recoverable=False)


# Alias for backward compatibility
GenerationError = WeightGenerationError


class ShardSaveError(VitriolError):
    """Failed to save weight shard."""

    def __init__(self, shard_path: str, reason: str = ""):
        message = f"Failed to save shard to '{shard_path}'"
        if reason:
            message += f": {reason}"
        message += "\n\nThis might be due to:\n"
        message += "  • Insufficient disk space\n"
        message += "  • Permission denied\n"
        message += "  • Disk I/O error"
        super().__init__(message, recoverable=True)


class StrategyError(VitriolError):
    """Base class for strategy-related errors."""
    pass


class IncompatibleStrategyError(StrategyError):
    """Strategy incompatible with requested format or model."""

    def __init__(
        self,
        strategy: str,
        format: str = None,
        reason: str = None
    ):
        message = f"Strategy '{strategy}' is incompatible"

        if format:
            message += f" with format '{format}'"

        if reason:
            message += f": {reason}"

        message += "\n\nSuggested alternatives:\n"

        if format == "safetensors" and strategy == "ultra":
            message += "  • Use '--format pytorch' instead\n"
            message += "  • Use '--strategy compact' for Safetensors format"
        else:
            message += "  • Try a different strategy (random, compact, sparse)\n"
            message += "  • Check the strategy documentation"

        super().__init__(message, recoverable=True)


class StrategyNotFoundError(StrategyError, KeyError):
    """Requested strategy does not exist."""

    def __init__(self, strategy: str, available_strategies: list):
        message = f"Strategy '{strategy}' not found.\n"
        message += f"Available strategies: {', '.join(available_strategies)}"
        super().__init__(message, recoverable=True)


class AdapterError(VitriolError):
    """Base class for adapter-related errors."""
    pass


class AdapterNotFoundError(AdapterError):
    """No suitable adapter found for model."""

    def __init__(self, model_id: str):
        message = f"No adapter found for model '{model_id}'.\n"
        message += "The model will use default processing.\n\n"
        message += "If you encounter issues, you can:\n"
        message += "  • Create a custom adapter\n"
        message += "  • Report the model for adapter support"
        super().__init__(message, recoverable=True)


class ModelNotSupportedError(AdapterError):
    """Model architecture is not supported."""

    def __init__(self, model_id: str, reason: str = ""):
        message = f"Model '{model_id}' is not supported"
        if reason:
            message += f": {reason}"
        message += "\n\nSuggestions:\n"
        message += "  • Check if the model ID is correct\n"
        message += "  • Create a custom adapter for this model\n"
        message += "  • Use a different model"
        super().__init__(message, recoverable=True)


class NASError(VitriolError):
    """Base class for NAS-related errors."""
    pass


class DatasetLoadError(NASError):
    """Failed to load dataset for NAS evaluation."""

    def __init__(self, dataset_name: str, reason: str = ""):
        message = f"Failed to load dataset '{dataset_name}'"
        if reason:
            message += f": {reason}"
        message += "\n\nSuggestions:\n"
        message += "  • Check dataset name and configuration\n"
        message += "  • Verify network connection\n"
        message += "  • Try using a local dataset"
        super().__init__(message, recoverable=True)


class CheckpointError(VitriolError):
    """Base class for checkpoint-related errors."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Checkpoint file is corrupted."""

    def __init__(self, checkpoint_path: str, reason: str = ""):
        message = f"Checkpoint corrupted: {checkpoint_path}"
        if reason:
            message += f"\nReason: {reason}"
        message += "\n\nThe checkpoint will be ignored and search will start fresh."
        super().__init__(message, recoverable=True)


class CheckpointSaveError(CheckpointError):
    """Failed to save checkpoint."""

    def __init__(self, checkpoint_path: str, reason: str = ""):
        message = f"Failed to save checkpoint to '{checkpoint_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, recoverable=True)


class ValidationError(VitriolError):
    """Model validation failed."""

    def __init__(self, model_path: str, reason: str = ""):
        message = f"Validation failed for model at '{model_path}'"
        if reason:
            message += f": {reason}"
        message += "\n\nThis might indicate:\n"
        message += "  • Incomplete weight generation\n"
        message += "  • Corrupted weight files\n"
        message += "  • Incompatible transformers version"
        super().__init__(message, recoverable=False)
