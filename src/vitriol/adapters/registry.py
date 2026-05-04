"""
Model Adapter Registry.

Provides automatic adapter discovery and selection for different model architectures.
Supports LLaMA, Qwen, DeepSeek, Mistral, Gemma, Phi, Cohere, GLM, StableLM, and MiniMax adapters.

Usage:
    from vitriol.adapters.registry import AdapterRegistry

    registry = AdapterRegistry()
    adapter = registry.get_adapter(model_id, config)
"""

import ast
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Type

from .base import DefaultAdapter, ModelAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

logger = logging.getLogger(__name__)

class AdapterRegistry:
    _adapters: List[Type[ModelAdapter]] = []
    _loaded = False

    @staticmethod
    def _iter_builtin_adapter_modules() -> List[tuple[str, Path]]:
        adapters_path = Path(__file__).parent
        return [
            (name, adapters_path / f"{name}.py")
            for _, name, _ in pkgutil.iter_modules([str(adapters_path)])
            if name not in {"base", "registry", "__init__"}
        ]

    @classmethod
    def discover_builtin_adapter_metadata(cls) -> List[Dict[str, Any]]:
        """Discover built-in adapter names without importing heavy ML deps."""
        package_name = __package__
        adapters: List[Dict[str, Any]] = []

        for module_name, module_path in cls._iter_builtin_adapter_modules():
            try:
                tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
            except (OSError, SyntaxError) as exc:
                logger.warning("Failed to inspect adapter module %s: %s", module_name, exc)
                continue

            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if not node.name.endswith("Adapter"):
                    continue
                if node.name in {"ModelAdapter", "DefaultAdapter"}:
                    continue

                adapters.append(
                    {
                        "name": node.name,
                        "module": f"{package_name}.{module_name}",
                    }
                )

        adapters.sort(key=lambda entry: entry["name"])
        adapters.append(
            {
                "name": "DefaultAdapter",
                "module": "vitriol.adapters.base",
                "is_fallback": True,
            }
        )
        return adapters

    @classmethod
    def register(cls, adapter_cls: Type[ModelAdapter]):
        """Register a new adapter class."""
        if adapter_cls not in cls._adapters:
            cls._adapters.insert(0, adapter_cls) # LIFO priority
            logger.debug(f"Registered adapter: {adapter_cls.__name__}")

    @classmethod
    def _load_builtin_adapters(cls):
        """Automatically load built-in adapters from the package."""
        if cls._loaded:
            return

        package_name = __package__

        for name, _module_path in cls._iter_builtin_adapter_modules():
            try:
                importlib.import_module(f"{package_name}.{name}")
                logger.debug("Loaded adapter module: %s", name)
            except Exception as exc:
                logger.error("Failed to load adapter module %s: %s", name, exc)

        cls._loaded = True

    @classmethod
    def get_adapter(cls, model_id: str, config: PretrainedConfig) -> ModelAdapter:
        """Find the first matching adapter for the given model."""
        cls._load_builtin_adapters()

        for adapter_cls in cls._adapters:
            try:
                if adapter_cls.match(model_id, config):
                    logger.info("Using adapter: %s", adapter_cls.__name__)
                    adapter = adapter_cls()
                    adapter.register_classes()
                    return adapter
            except Exception as exc:
                logger.warning("Error checking match for adapter %s: %s", adapter_cls.__name__, exc)

        return DefaultAdapter()
