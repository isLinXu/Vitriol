"""
Vitriol Evolution Module
=======================

Architecture Evolution Tree, Comparison, Simulation, and Recommendation tools.

Features:
- Evolution Tree: Visualize model architecture family trees
- Compare: Generate detailed architecture comparison reports
- Simulator: Estimate performance metrics for architectures
- Recommender: Recommend architectures based on requirements
- Timeline: Visualize architecture innovations over time
"""

__version__ = "0.3.0"

from .compare import ArchComparator, ComparisonReport, ComparisonResult
from .recommender import (
    ArchitectureRecommendation,
    ArchitectureRecommender,
    RecommendationCriteria,
    UseCase,
)
from .simulator import ArchSimulator, SimulationResult, quick_estimate
from .timeline import InnovationTimeline, TimelineEvent
from .tree_builder import ArchInnovation, ArchNode, EvolutionTree
from .tree_visualizer import TreeVisualizer

__all__ = [
    # Core tree structures
    "EvolutionTree",
    "ArchNode",
    "ArchInnovation",
    # Visualization
    "TreeVisualizer",
    # Comparison
    "ArchComparator",
    "ComparisonReport",
    "ComparisonResult",
    # Simulation
    "ArchSimulator",
    "SimulationResult",
    "quick_estimate",
    # Recommendation
    "ArchitectureRecommender",
    "ArchitectureRecommendation",
    "RecommendationCriteria",
    "UseCase",
    # Timeline
    "InnovationTimeline",
    "TimelineEvent",
]
