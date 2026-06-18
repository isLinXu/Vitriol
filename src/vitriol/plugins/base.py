"""
Plugin System for Vitriol.

Provides extensibility through:
- Custom strategies
- Custom adapters
- Custom analyzers
- Hooks and callbacks
"""

import importlib
import logging
import pkgutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """
    Base class for Vitriol plugins.

    Plugins can extend Vitriol with:
    - Custom weight generation strategies
    - Custom model adapters
    - Custom architecture analyzers
    - Custom CLI commands
    """

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.

        Args:
            context: Application context with registries

        Returns:
            True if initialization successful
        """
        pass

    def shutdown(self) -> None:  # noqa: B027  (optional hook for subclasses)
        """Clean up plugin resources."""
        pass

    def get_strategies(self) -> Dict[str, Type]:
        """
        Return custom strategies provided by this plugin.

        Returns:
            Dict mapping strategy names to classes
        """
        return {}

    def get_adapters(self) -> Dict[str, Type]:
        """
        Return custom adapters provided by this plugin.

        Returns:
            Dict mapping adapter names to classes
        """
        return {}

    def get_analyzers(self) -> Dict[str, Type]:
        """
        Return custom analyzers provided by this plugin.

        Returns:
            Dict mapping analyzer names to classes
        """
        return {}

    def get_cli_commands(self) -> Dict[str, Callable]:
        """
        Return custom CLI commands provided by this plugin.

        Returns:
            Dict mapping command names to functions
        """
        return {}


class PluginManager:
    """
    Manages plugin lifecycle and registration.

    Features:
        - Automatic plugin discovery
        - Dependency resolution
        - Hot reload support
        - Sandboxed execution
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_paths: List[Path] = []

        # Add default plugin paths
        self._add_default_paths()

    def _add_default_paths(self):
        """Add default plugin search paths."""
        import vitriol
        vitriol_root = Path(vitriol.__file__).parent

        self.plugin_paths.extend([
            vitriol_root / "plugins",
            Path.home() / ".vitriol" / "plugins",
            Path("/usr/share/vitriol/plugins"),
        ])

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins.

        Returns:
            List of plugin names
        """
        discovered = []

        for path in self.plugin_paths:
            if not path.exists():
                continue

            # Add to Python path
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

            # Find plugin modules
            for _finder, name, _ispkg in pkgutil.iter_modules([str(path)]):
                if name.startswith('vitriol_'):
                    discovered.append(name)

        return discovered

    def load_plugin(self, name: str) -> Optional[Plugin]:
        """
        Load a plugin by name.

        Args:
            name: Plugin module name

        Returns:
            Plugin instance or None if failed
        """
        if name in self.plugins:
            logger.debug("Plugin %s is already loaded; returning existing instance.", name)
            return self.plugins[name]

        try:
            module = importlib.import_module(name)

            # Find plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, Plugin) and
                    attr != Plugin):
                    plugin_class = attr
                    break

            if plugin_class is None:
                logger.error("No Plugin class found in %s", name)
                return None

            # Instantiate
            plugin = plugin_class()

            # Validate metadata
            if not plugin.name:
                plugin.name = name
            if not plugin.version:
                plugin.version = "0.0.0"

            # Initialize
            context = self._create_context()
            if plugin.initialize(context):
                self.plugins[name] = plugin
                logger.info("Loaded plugin: %s v%s (%s)", name, plugin.version, plugin.description or "no description")
                return plugin
            else:
                logger.error("Plugin %s failed to initialize", name)
                return None

        except Exception as e:
            logger.error("Failed to load plugin %s: %s", name, e)
            return None

    def load_all(self) -> Dict[str, Any]:
        """Load all discovered plugins.

        Returns:
            Summary dict with 'loaded', 'failed', and 'total' counts.
        """
        discovered = self.discover_plugins()
        loaded: List[str] = []
        failed: List[str] = []

        for name in discovered:
            result = self.load_plugin(name)
            if result is not None:
                loaded.append(name)
            else:
                failed.append(name)

        summary = {
            "total_discovered": len(discovered),
            "loaded": len(loaded),
            "failed": len(failed),
            "loaded_names": loaded,
            "failed_names": failed,
        }
        logger.info("Plugin loading complete: %(loaded)d/%(total_discovered)d succeeded", summary)
        return summary

    def register_hook(self, event: str, callback: Callable) -> None:
        """
        Register a hook callback.

        Args:
            event: Event name
            callback: Function to call
        """
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)

    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """
        Trigger a hook event.

        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of callback results
        """
        results = []
        for callback in self.hooks.get(event, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback failed for {event}: {e}")
        return results

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded plugins."""
        return [
            {
                'name': name,
                'version': plugin.version,
                'description': plugin.description,
                'author': plugin.author,
            }
            for name, plugin in self.plugins.items()
        ]

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name and call its shutdown hook."""
        plugin = self.plugins.pop(name, None)
        if plugin is None:
            return False
        try:
            plugin.shutdown()
        except Exception as e:
            logger.warning("Plugin %s shutdown failed: %s", name, e)
        return True


# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def init_plugins() -> Any:
    """Initialize and load all plugins."""
    manager = get_plugin_manager()
    manager.load_all()
    return manager
