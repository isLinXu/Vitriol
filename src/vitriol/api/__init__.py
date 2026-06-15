"""
Vitriol REST API Module
========================

Provides HTTP endpoints for model generation, architecture search,
job status tracking, and batch generation.

Install via: ``pip install "vitriol[api]"``

Example::

    from vitriol.api import app
    # or launch directly:
    # python -m vitriol.api.server
"""

from .server import app

__all__ = ["app"]
