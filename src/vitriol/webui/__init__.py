# ⚠️ EXPERIMENTAL: This module is under development and may change without notice.
# Not yet integrated with the core Vitriol pipeline.

"""
Vitriol Web UI Module
====================

Provides a Gradio-based web interface for Vitriol features.
"""

from .app import create_app, launch

__all__ = ["create_app", "launch"]
