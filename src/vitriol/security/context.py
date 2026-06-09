from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from vitriol.config.manager import SecurityOptions


def _get_bool(data: Mapping[str, Any], key: str) -> Optional[bool]:
    if key not in data or data[key] is None:
        return None
    v = data[key]
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, SecurityOptions):
        return {
            "trust_remote_code": bool(value.trust_remote_code),
            "allow_network": bool(value.allow_network),
            "local_files_only": bool(value.local_files_only),
        }
    # generic mapping-like
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@dataclass(frozen=True)
class SecurityContext:
    """Security context capturing origin and trust decisions."""
    trust_remote_code: bool
    allow_network: bool
    local_files_only: bool
    provenance: Dict[str, str]

    def to_security_options(self) -> SecurityOptions:
        return SecurityOptions(
            trust_remote_code=bool(self.trust_remote_code),
            allow_network=bool(self.allow_network),
            local_files_only=bool(self.local_files_only),
        )

    def apply_to_environ(self) -> None:
        # Align with hf_loading: once networking is disabled, mark the process OFFLINE (non-bypassable).
        if not self.allow_network:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_security_context(
    *,
    base: Optional[Union[SecurityOptions, Mapping[str, Any]]] = None,
    explicit: Optional[Mapping[str, Any]] = None,
) -> SecurityContext:
    """
    Resolve multi-source security options into a single SecurityContext, and carry provenance (source).

    Current input surfaces:
    - base: defaults / config file / upstream preset
    - explicit: call-site explicit overrides (higher priority than base)
    - env OFFLINE: highest priority (non-bypassable)
    """
    base_d = _as_dict(base)
    explicit_d = _as_dict(explicit)

    # base defaults
    trc = _get_bool(base_d, "trust_remote_code")
    an = _get_bool(base_d, "allow_network")
    lfo = _get_bool(base_d, "local_files_only")
    provenance: Dict[str, str] = {}

    if trc is None:
        trc = False
    provenance["trust_remote_code"] = "base"
    if an is None:
        an = True
    provenance["allow_network"] = "base"
    if lfo is None:
        lfo = False
    provenance["local_files_only"] = "base"

    # explicit overrides
    for key in ("trust_remote_code", "allow_network", "local_files_only"):
        v = _get_bool(explicit_d, key)
        if v is None:
            continue
        if key == "trust_remote_code":
            trc = v
        elif key == "allow_network":
            an = v
        else:
            lfo = v
        provenance[key] = "explicit"

    # invariants
    if not an:
        # allow_network=False => local_files_only=True (if caller didn't set local_files_only explicitly, mark as derived).
        if not lfo:
            lfo = True
            if provenance.get("local_files_only") not in {"explicit", "env_offline"}:
                provenance["local_files_only"] = "inferred_offline"

    # env OFFLINE is the highest priority and cannot be bypassed
    if _env_offline():
        an = False
        lfo = True
        provenance["allow_network"] = "env_offline"
        provenance["local_files_only"] = "env_offline"

    ctx = SecurityContext(
        trust_remote_code=bool(trc),
        allow_network=bool(an),
        local_files_only=bool(lfo),
        provenance=provenance,
    )
    ctx.apply_to_environ()
    return ctx
