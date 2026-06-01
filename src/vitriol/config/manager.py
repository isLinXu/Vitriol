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
        valid = {'random','sparse','compact','ultra','hybrid_ultra','ternary',
                 'binary','quantized','lowrank','structured_sparse',
                 'learned','hybrid_learned','quantum'}
        if self.strategy not in valid:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if not (0 <= self.sparsity <= 1):
            raise ValueError("Sparsity must be in [0,1]")
        if not (1 <= self.n_bits <= 32):
            raise ValueError("n_bits must be in [1,32]")

    @classmethod
    def from_yaml(cls, path: Path) -> 'GenerationConfig':
        return build_generation_config(config_path=path)

    @classmethod
    def from_env(cls) -> 'GenerationConfig':
        """Load from environment variables."""
        return build_generation_config()


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
    config = GenerationConfig(**data)
    config.security = security
    config.security_context = {
        "trust_remote_code": bool(resolved.trust_remote_code),
        "allow_network": bool(resolved.allow_network),
        "local_files_only": bool(resolved.local_files_only),
        "provenance": dict(resolved.provenance),
    }
    return config
