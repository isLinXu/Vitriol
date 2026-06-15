"""
Vitriol Model Registry Module
=============================

Provides model storage, versioning, and lineage tracking through
cryptographic fingerprinting.

Features:
    - Model fingerprinting (architecture + weights + behavioral DNA)
    - Version lineage tracking
    - Integrity verification

Example::

    from vitriol.registry import ModelStore, FingerprintRegistry
"""

from .model_store import ModelStore

__all__ = [
    "ModelStore",
]
