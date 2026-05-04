"""
vitriol.utils

Note: some utilities in this package (e.g., the HuggingFace loading facade) intentionally
delay importing heavy dependencies such as transformers/torch, so that vitriol can still be
imported in lightweight environments (static analysis / report generation / unit tests only).
"""

from __future__ import annotations
