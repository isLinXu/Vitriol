from typing import Any
from .core import Architecture
from .analyzers import AnalyzerRegistry
import logging

logger = logging.getLogger(__name__)

class ArchitectureAnalyzer:
    """Analyzes model configuration to build Architecture object."""
    
    def analyze(self, config: Any) -> Architecture:
        """Analyze config and return Architecture object."""
        model_type = getattr(config, 'model_type', 'unknown')
        logger.info(f"Analyzing architecture for model_type: {model_type}")
        
        # Get specific analyzer from registry
        analyzer = AnalyzerRegistry.resolve(config)
        logger.info("Using architecture analyzer: %s", analyzer.__class__.__name__)
        return analyzer.analyze(config)
