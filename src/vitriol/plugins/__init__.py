"""
Vitriol Plugin System
=====================

Provides extensibility through custom strategies, adapters, analyzers,
and CLI commands.

Plugins are auto-discovered from ``~/.vitriol/plugins/`` and
``/usr/share/vitriol/plugins/``.

Example::

    from vitriol.plugins import Plugin, PluginManager
    from vitriol.plugins import get_plugin_manager, init_plugins
"""

from .base import Plugin, PluginManager, get_plugin_manager, init_plugins

__all__ = [
    "Plugin",
    "PluginManager",
    "get_plugin_manager",
    "init_plugins",
]
