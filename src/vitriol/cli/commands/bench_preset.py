"""Preset validation and normalization."""

VALID_PRESETS = (
    "safe",
    "balanced",
    "fast-balanced",
    "ultra-long",
    "aggressive",
)

VALID_MODES = (
    "turboquant",
    "attention-gated",
    "predictive",
    "cross-layer",
    "dict-kv",
    "spectral",
    "sparse",
    "ternary",
    "binary",
    "quantized",
    "lowrank",
)


def validate_preset(preset: str) -> str:
    """Validate and normalize a preset name."""
    if preset not in VALID_PRESETS:
        raise ValueError(f"Invalid preset: {preset}")
    return preset


def validate_mode(mode: str) -> str:
    """Validate and normalize a mode name."""
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}")
    return mode
