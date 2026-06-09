"""
Configuration Management Module.

Provides configuration classes and builders for Vitriol:
- SecurityOptions: Security settings for model loading
- GenerationConfig: Weight generation parameters
- build_generation_config(): Unified config builder from YAML/env/overrides

Configuration sources (in priority order):
1. Explicit overrides (programmatic)
2. Environment variables (VITRIOL_* prefix)
3. YAML config file
4. Default values
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.exceptions import ConfigValidationError
from ..utils.optional import has as _has_optional
from ..utils.size import parse_size_to_bytes

# Single source of truth for the values the validators and the JSON Schema share.
VALID_STRATEGIES = frozenset({
    "random", "sparse", "compact", "ultra", "hybrid_ultra", "ternary",
    "binary", "quantized", "lowrank", "structured_sparse",
    "learned", "hybrid_learned", "quantum",
})
VALID_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SecurityOptions:
    """Security-related options for model loading.

    Controls how Vitriol handles remote code from HuggingFace models.
    Setting trust_remote_code=False is safer but limits model compatibility.
    """
    trust_remote_code: bool = False
    allow_network: bool = True
    local_files_only: bool = False


@dataclass
class GenerationConfig:
    """Configuration for weight generation runs."""
    max_shard_size: str = "5GB"
    dtype: str = "bfloat16"
    strategy: str = "random"
    auto_validate: bool = True

    # Strategy specific params
    n_bits: int = 8
    rank: int = 16
    sparsity: float = 0.5

    # Security
    security: SecurityOptions = None
    # P4: audit/provenance (SSoT output)
    security_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.security is None:
            self.security = SecurityOptions()
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                f"Choose one of: {', '.join(sorted(VALID_STRATEGIES))}"
            )
        if not isinstance(self.sparsity, (int, float)) or isinstance(self.sparsity, bool):
            raise ValueError(f"Sparsity must be a number, got {type(self.sparsity).__name__}")
        if not (0 <= self.sparsity <= 1):
            raise ValueError("Sparsity must be in [0,1]")
        if not isinstance(self.n_bits, int) or isinstance(self.n_bits, bool):
            raise ValueError(f"n_bits must be an integer, got {type(self.n_bits).__name__}")
        if not (1 <= self.n_bits <= 32):
            raise ValueError("n_bits must be in [1,32]")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError(f"rank must be a positive integer, got {self.rank!r}")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype: {self.dtype!r}. "
                f"Choose one of: {', '.join(sorted(VALID_DTYPES))}"
            )
        try:
            parse_size_to_bytes(self.max_shard_size)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid max_shard_size: {self.max_shard_size!r} ({exc})") from exc

    @classmethod
    def from_yaml(cls, path: Path) -> 'GenerationConfig':
        return build_generation_config(config_path=path)

    @classmethod
    def from_env(cls) -> 'GenerationConfig':
        """Load from environment variables."""
        return build_generation_config()


def generation_config_schema() -> Dict[str, Any]:
    """Return a JSON Schema (draft-07) describing the generation config dict.

    The schema is the documented contract for the ``default:`` section of a
    Vitriol YAML file and for programmatic overrides. It is dependency-free; the
    optional ``jsonschema`` package (if installed) is used by
    :func:`validate_generation_dict` for full structural validation.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "VitriolGenerationConfig",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "max_shard_size": {
                "type": "string",
                "description": "Human-readable shard size, e.g. '5GB', '512 MB'.",
                "pattern": r"^\s*[0-9]+(\.[0-9]+)?\s*[A-Za-z]*\s*$",
            },
            "dtype": {"type": "string", "enum": sorted(VALID_DTYPES)},
            "strategy": {"type": "string", "enum": sorted(VALID_STRATEGIES)},
            "auto_validate": {"type": "boolean"},
            "n_bits": {"type": "integer", "minimum": 1, "maximum": 32},
            "rank": {"type": "integer", "minimum": 1},
            "sparsity": {"type": "number", "minimum": 0, "maximum": 1},
            "security": {"type": "object"},
            "security_context": {"type": "object"},
        },
    }


def validate_generation_dict(data: Dict[str, Any]) -> None:
    """Validate a raw generation-config dict against :func:`generation_config_schema`.

    Always rejects unknown keys with an actionable :class:`ConfigValidationError`
    (instead of the cryptic ``TypeError`` from ``GenerationConfig(**data)``). When
    the optional ``jsonschema`` package is installed, full structural validation
    (types, ranges, enums) is performed as well.
    """
    schema = generation_config_schema()
    allowed = set(schema["properties"])
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigValidationError(
            "generation config",
            f"unknown key(s) {unknown}; allowed keys: {sorted(allowed)}",
        )

    if _has_optional("jsonschema"):
        import jsonschema

        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(p) for p in exc.absolute_path) or "<root>"
            raise ConfigValidationError(
                "generation config", f"at '{location}': {exc.message}"
            ) from exc


def _load_generation_dict_from_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    default_section = data.get("default", {})
    return default_section if isinstance(default_section, dict) else {}


def _load_generation_dict_from_env() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if os.getenv("VITRIOL_MAX_SHARD_SIZE") is not None:
        data["max_shard_size"] = os.getenv("VITRIOL_MAX_SHARD_SIZE")
    if os.getenv("VITRIOL_DTYPE") is not None:
        data["dtype"] = os.getenv("VITRIOL_DTYPE")
    return data


def build_generation_config(
    *,
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> GenerationConfig:
    data: Dict[str, Any] = {}
    data.update(_load_generation_dict_from_env())
    if config_path is not None:
        data.update(_load_generation_dict_from_yaml(config_path))

    explicit = dict(overrides or {})
    base_security = SecurityOptions()
    explicit_security: Dict[str, Any] = {}

    yaml_security = data.pop("security", None)
    if isinstance(yaml_security, dict):
        if "trust_remote_code" in yaml_security:
            base_security.trust_remote_code = _coerce_bool(yaml_security["trust_remote_code"])
        if "allow_network" in yaml_security:
            base_security.allow_network = _coerce_bool(yaml_security["allow_network"])
        if "local_files_only" in yaml_security:
            base_security.local_files_only = _coerce_bool(yaml_security["local_files_only"])

    if "trust_remote_code" in explicit and explicit["trust_remote_code"] is not None:
        explicit_security["trust_remote_code"] = _coerce_bool(explicit.pop("trust_remote_code"))
    if "allow_network" in explicit and explicit["allow_network"] is not None:
        explicit_security["allow_network"] = _coerce_bool(explicit.pop("allow_network"))
    if "local_files_only" in explicit and explicit["local_files_only"] is not None:
        explicit_security["local_files_only"] = _coerce_bool(explicit.pop("local_files_only"))

    # P3/P4: resolve final security semantics via a single source of truth, honoring env OFFLINE (non-bypassable),
    # while preserving provenance for auditing.
    from vitriol.security.context import resolve_security_context

    resolved = resolve_security_context(base=base_security, explicit=explicit_security)
    security = resolved.to_security_options()

    data.update({k: v for k, v in explicit.items() if v is not None})
    validate_generation_dict(data)
    config = GenerationConfig(**data)
    config.security = security
    config.security_context = {
        "trust_remote_code": bool(resolved.trust_remote_code),
        "allow_network": bool(resolved.allow_network),
        "local_files_only": bool(resolved.local_files_only),
        "provenance": dict(resolved.provenance),
    }
    return config
