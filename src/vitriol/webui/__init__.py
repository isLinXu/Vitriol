"""
Vitriol Web UI Module
=====================

Provides a Gradio-based web interface for Vitriol features including
model comparison, evolution tree visualization, targeted NAS, and
performance simulation.

Launch via CLI::

    vitriol webui
    vitriol webui --port 8080 --share

Or programmatically::

    from vitriol.webui import create_app, launch
"""

from .app import create_app, launch

__all__ = ["create_app", "launch"]
