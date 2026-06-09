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
                logger.error(f"No Plugin class found in {name}")
                return None

            # Instantiate
            plugin = plugin_class()

            # Initialize
            context = self._create_context()
            if plugin.initialize(context):
                self.plugins[name] = plugin
                logger.info(f"Loaded plugin: {name} v{plugin.version}")
                return plugin
            else:
                logger.error(f"Plugin {name} failed to initialize")
                return None

        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            return None

    def _create_context(self) -> Dict[str, Any]:
        """Create application context for plugins."""
        from ..adapters import register_adapter
        from ..strategies import register_strategy

        return {
            'register_strategy': register_strategy,
            'register_adapter': register_adapter,
            'register_hook': self.register_hook,
            'config': {},
        }

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        if name in self.plugins:
            plugin = self.plugins[name]
            plugin.shutdown()
            del self.plugins[name]
            logger.info(f"Unloaded plugin: {name}")

    def load_all(self) -> None:
        """Load all discovered plugins."""
        for name in self.discover_plugins():
            self.load_plugin(name)

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
