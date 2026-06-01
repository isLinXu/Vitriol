"""
Weight generation strategies for Vitriol.

This module provides different strategies for generating model weights,
each with different trade-offs between size, training support, and compatibility.
"""

from .base import StrategyCapabilities, WeightGenerationStrategy
from .compact import CompactStrategy
from .hybrid_ultra import HybridUltraStrategy
from .random import RandomStrategy
from .ultra import UltraStrategy

# Import other strategies if they exist
try:
    from .sparse import SparseStrategy
except ImportError:
    SparseStrategy = None

try:
    from .ternary import TernaryStrategy
except ImportError:
    TernaryStrategy = None

try:
    from .binary import BinaryStrategy
except ImportError:
    BinaryStrategy = None

try:
    from .quantized import QuantizedStrategy
except ImportError:
    QuantizedStrategy = None

try:
    from .lowrank import LowRankStrategy
except ImportError:
    LowRankStrategy = None

try:
    from .structured_sparse import StructuredSparseStrategy
except ImportError:
    StructuredSparseStrategy = None

try:
    from .quantum import QuantumStrategy
except ImportError:
    QuantumStrategy = None

# Learning-based strategies (P0 innovation)
try:
    from .learned import HybridLearnedStrategy, LearnedWeightStrategy
except ImportError:
    LearnedWeightStrategy = None
    HybridLearnedStrategy = None


# Strategy registry
STRATEGY_REGISTRY = {
    "random": RandomStrategy,
    "compact": CompactStrategy,
    "ultra": UltraStrategy,
    "hybrid_ultra": HybridUltraStrategy,
}

# Add optional strategies
if SparseStrategy:
    STRATEGY_REGISTRY["sparse"] = SparseStrategy
if TernaryStrategy:
    STRATEGY_REGISTRY["ternary"] = TernaryStrategy
if BinaryStrategy:
    STRATEGY_REGISTRY["binary"] = BinaryStrategy
if QuantizedStrategy:
    STRATEGY_REGISTRY["quantized"] = QuantizedStrategy
if LowRankStrategy:
    STRATEGY_REGISTRY["lowrank"] = LowRankStrategy
if StructuredSparseStrategy:
    STRATEGY_REGISTRY["structured_sparse"] = StructuredSparseStrategy
if QuantumStrategy:
    STRATEGY_REGISTRY["quantum"] = QuantumStrategy
if LearnedWeightStrategy:
    STRATEGY_REGISTRY["learned"] = LearnedWeightStrategy
if HybridLearnedStrategy:
    STRATEGY_REGISTRY["hybrid_learned"] = HybridLearnedStrategy


def get_strategy(name: str, **kwargs) -> WeightGenerationStrategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (e.g., "random", "compact", "ultra")
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        Strategy instance (validated)

    Raises:
        KeyError: If strategy name is not found
        ValueError: If strategy configuration is invalid
    """
    if name not in STRATEGY_REGISTRY:
        from ..utils.exceptions import StrategyNotFoundError
        raise StrategyNotFoundError(name, list(STRATEGY_REGISTRY.keys()))

    strategy = STRATEGY_REGISTRY[name](**kwargs)

    # [Hardening] Validate strategy configuration after construction
    try:
        strategy.validate_config()
    except (ValueError, TypeError) as e:
        from ..utils.exceptions import StrategyError
        raise StrategyError(
            f"Strategy '{name}' has invalid configuration: {e}",
            recoverable=True,
        ) from e

    return strategy


def list_strategies() -> list:
    """
    List all available strategy names.

    Returns:
        List of strategy names
    """
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "WeightGenerationStrategy",
    "StrategyCapabilities",
    "RandomStrategy",
    "CompactStrategy",
    "UltraStrategy",
    "HybridUltraStrategy",
    "LearnedWeightStrategy",
    "HybridLearnedStrategy",
    "QuantumStrategy",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
]
