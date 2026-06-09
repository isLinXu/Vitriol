"""
vitriol.security

This package provides a single source of truth for "security semantics". It performs consistent
parsing and precedence resolution across CLI / API / config files / environment variables, and
also carries provenance (source) information for auditing.

Integrated with:
- ``vitriol.config.manager`` — resolve_security_context() used in config resolution
- ``vitriol.utils.hf_loading`` — security-aware HuggingFace model loading facade
- ``vitriol.core.generator`` — SecurityOptions consumed throughout the generation pipeline
"""

from __future__ import annotations

from .context import SecurityContext as SecurityContext
from .context import resolve_security_context as resolve_security_context

__all__ = [
    "SecurityContext",
    "resolve_security_context",
]

