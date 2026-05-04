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

from .tree_builder import EvolutionTree, ArchNode, ArchInnovation
from .tree_visualizer import TreeVisualizer
from .compare import ArchComparator, ComparisonReport, ComparisonResult
from .simulator import ArchSimulator, SimulationResult, quick_estimate
from .recommender import (
    ArchitectureRecommender,
    ArchitectureRecommendation,
    RecommendationCriteria,
    UseCase,
)
from .timeline import InnovationTimeline, TimelineEvent

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
