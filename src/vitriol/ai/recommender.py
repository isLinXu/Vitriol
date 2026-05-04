"""
AI-Powered Recommendation System for Vitriol.

Provides intelligent recommendations for:
- Strategy selection based on model and hardware
- Parameter optimization
- Hardware configuration
- Performance prediction
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of recommendations."""
    STRATEGY = "strategy"
    HARDWARE = "hardware"
    PARAMETERS = "parameters"
    OPTIMIZATION = "optimization"


@dataclass
class Recommendation:
    """Single recommendation."""
    type: RecommendationType
    item: str
    confidence: float
    reason: str
    expected_benefit: str
    alternatives: List[str]


class StrategyRecommender:
    """
    Recommends optimal generation strategy.
    
    Considers:
    - Model size and architecture
    - Available hardware
    - Desired compression level
    - Use case requirements
    """
    
    STRATEGY_PROFILES = {
        "ultra": {
            "compression": 0.99,
            "speed": "fast",
            "memory": "minimal",
            "training": False,
            "use_cases": ["storage", "transfer", "testing"]
        },
        "quantum": {
            "compression": 0.95,
            "speed": "fast",
            "memory": "low",
            "training": True,
            "use_cases": ["edge", "mobile", "quantization"]
        },
        "compact": {
            "compression": 0.75,
            "speed": "medium",
            "memory": "medium",
            "training": True,
            "use_cases": ["development", "experimentation"]
        },
        "sparse": {
            "compression": 0.50,
            "speed": "medium",
            "memory": "medium",
            "training": True,
            "use_cases": ["pruning", "sparsity_research"]
        },
        "random": {
            "compression": 0.0,
            "speed": "fast",
            "memory": "high",
            "training": True,
            "use_cases": ["baseline", "ablation"]
        }
    }
    
    def recommend(
        self,
        model_size_gb: float,
        available_memory_gb: float,
        use_case: str = "general",
        requires_training: bool = False
    ) -> Recommendation:
        """
        Recommend strategy based on constraints.
        
        Args:
            model_size_gb: Model size in GB
            available_memory_gb: Available memory in GB
            use_case: Intended use case
            requires_training: Whether training is needed
            
        Returns:
            Recommendation
        """
        scores = {}
        
        for strategy, profile in self.STRATEGY_PROFILES.items():
            score = 0.0
            
            # Memory constraint
            if profile["memory"] == "minimal":
                score += 1.0
            elif profile["memory"] == "low":
                score += 0.8
            elif profile["memory"] == "medium":
                score += 0.5
            else:
                score += 0.2
            
            # Training requirement
            if requires_training and profile["training"]:
                score += 1.0
            elif requires_training and not profile["training"]:
                score -= 10.0  # Heavy penalty
            
            # Use case match
            if use_case in profile["use_cases"]:
                score += 0.5
            
            # Compression preference for large models
            if model_size_gb > 10:
                score += profile["compression"] * 0.5
            
            scores[strategy] = score
        
        # Select best
        best_strategy = max(scores, key=scores.get)
        best_score = scores[best_strategy]
        
        # Get alternatives
        alternatives = sorted(
            [s for s in scores if s != best_strategy],
            key=lambda s: scores[s],
            reverse=True
        )[:2]
        
        profile = self.STRATEGY_PROFILES[best_strategy]
        
        return Recommendation(
            type=RecommendationType.STRATEGY,
            item=best_strategy,
            confidence=min(best_score / 3.0, 1.0),
            reason=f"Optimal for {use_case} with {profile['memory']} memory usage",
            expected_benefit=f"{profile['compression']*100:.0f}% size reduction",
            alternatives=alternatives
        )


class HardwareRecommender:
    """
    Recommends optimal hardware configuration.
    """
    
    def recommend(
        self,
        model_params: int,
        batch_size: int = 1,
        latency_requirement_ms: Optional[float] = None
    ) -> List[Recommendation]:
        """
        Recommend hardware configuration.
        
        Args:
            model_params: Number of model parameters
            batch_size: Target batch size
            latency_requirement_ms: Latency requirement
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Memory estimation
        model_size_gb = model_params * 2 / (1024**3)  # bfloat16
        
        # GPU recommendation
        if model_size_gb > 40:
            recommendations.append(Recommendation(
                type=RecommendationType.HARDWARE,
                item="A100_80GB",
                confidence=0.95,
                reason="Model requires >40GB memory",
                expected_benefit="Fit entire model on single GPU",
                alternatives=["A100_40GB_x2", "CPU_offloading"]
            ))
        elif model_size_gb > 20:
            recommendations.append(Recommendation(
                type=RecommendationType.HARDWARE,
                item="A100_40GB",
                confidence=0.90,
                reason="Model requires >20GB memory",
                expected_benefit="Good balance of memory and cost",
                alternatives=["A10_x2", "V100_32GB"]
            ))
        else:
            recommendations.append(Recommendation(
                type=RecommendationType.HARDWARE,
                item="T4",
                confidence=0.85,
                reason="Cost-effective for smaller models",
                expected_benefit="Low cost inference",
                alternatives=[["A10", "CPU"]]
            ))
        
        # CPU recommendation
        if latency_requirement_ms and latency_requirement_ms < 100:
            recommendations.append(Recommendation(
                type=RecommendationType.HARDWARE,
                item="high_cpu_count",
                confidence=0.80,
                reason="Low latency requirement",
                expected_benefit="Parallel processing for speed",
                alternatives=["GPU_acceleration"]
            ))
        
        return recommendations


class ParameterRecommender:
    """
    Recommends optimal generation parameters.
    """
    
    def recommend_shard_size(
        self,
        model_size_gb: float,
        disk_speed_mbps: float,
        network_speed_mbps: Optional[float] = None
    ) -> Recommendation:
        """
        Recommend shard size.
        
        Args:
            model_size_gb: Model size in GB
            disk_speed_mbps: Disk write speed
            network_speed_mbps: Network speed (if distributed)
            
        Returns:
            Recommendation
        """
        # Base recommendation on disk speed
        if disk_speed_mbps > 1000:  # NVMe
            shard_size = "5GB"
            confidence = 0.95
            reason = "Fast NVMe storage can handle large shards"
        elif disk_speed_mbps > 500:  # SSD
            shard_size = "2GB"
            confidence = 0.90
            reason = "SSD storage optimal for medium shards"
        else:  # HDD
            shard_size = "500MB"
            confidence = 0.85
            reason = "Slow HDD requires smaller shards"
        
        # Adjust for network if distributed
        if network_speed_mbps:
            if network_speed_mbps < 100:  # Slow network
                shard_size = "1GB"
                reason += ", adjusted for slow network"
        
        return Recommendation(
            type=RecommendationType.PARAMETERS,
            item=shard_size,
            confidence=confidence,
            reason=reason,
            expected_benefit="Optimal I/O performance",
            alternatives=["2GB", "1GB"] if shard_size == "5GB" else ["5GB", "1GB"]
        )
    
    def recommend_parallel_workers(
        self,
        cpu_count: int,
        io_bound: bool = True
    ) -> Recommendation:
        """
        Recommend number of parallel workers.
        
        Args:
            cpu_count: Number of CPUs
            io_bound: Whether task is I/O bound
            
        Returns:
            Recommendation
        """
        if io_bound:
            # I/O bound can use more workers
            workers = min(cpu_count * 2, 16)
            confidence = 0.90
            reason = "I/O bound tasks benefit from more workers"
        else:
            # CPU bound limited by cores
            workers = max(1, cpu_count - 1)
            confidence = 0.95
            reason = "CPU bound tasks limited by core count"
        
        return Recommendation(
            type=RecommendationType.PARAMETERS,
            item=str(workers),
            confidence=confidence,
            reason=reason,
            expected_benefit=f"Optimal parallelism with {workers} workers",
            alternatives=[str(workers//2), str(workers*2)]
        )


class VitriolRecommender:
    """
    Main recommendation engine.
    
    Aggregates recommendations from all sub-recommenders.
    """
    
    def __init__(self):
        self.strategy = StrategyRecommender()
        self.hardware = HardwareRecommender()
        self.parameters = ParameterRecommender()
    
    def recommend_all(
        self,
        model_id: str,
        model_params: int,
        available_memory_gb: float,
        use_case: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get all recommendations.
        
        Args:
            model_id: Model identifier
            model_params: Number of parameters
            available_memory_gb: Available memory
            use_case: Intended use case
            **kwargs: Additional context
            
        Returns:
            Complete recommendation set
        """
        model_size_gb = model_params * 2 / (1024**3)
        
        recommendations = {
            "model_id": model_id,
            "model_size_gb": round(model_size_gb, 2),
            "timestamp": time.time(),
            "recommendations": []
        }
        
        # Strategy recommendation
        strategy_rec = self.strategy.recommend(
            model_size_gb=model_size_gb,
            available_memory_gb=available_memory_gb,
            use_case=use_case,
            requires_training=kwargs.get("requires_training", False)
        )
        recommendations["recommendations"].append({
            "category": "strategy",
            "primary": strategy_rec.item,
            "confidence": round(strategy_rec.confidence, 2),
            "reason": strategy_rec.reason,
            "expected_benefit": strategy_rec.expected_benefit,
            "alternatives": strategy_rec.alternatives
        })
        
        # Hardware recommendations
        hardware_recs = self.hardware.recommend(
            model_params=model_params,
            batch_size=kwargs.get("batch_size", 1)
        )
        recommendations["recommendations"].extend([
            {
                "category": "hardware",
                "primary": rec.item,
                "confidence": round(rec.confidence, 2),
                "reason": rec.reason,
                "expected_benefit": rec.expected_benefit
            }
            for rec in hardware_recs
        ])
        
        # Parameter recommendations
        shard_rec = self.parameters.recommend_shard_size(
            model_size_gb=model_size_gb,
            disk_speed_mbps=kwargs.get("disk_speed_mbps", 500)
        )
        recommendations["recommendations"].append({
            "category": "parameters",
            "parameter": "shard_size",
            "value": shard_rec.item,
            "confidence": round(shard_rec.confidence, 2),
            "reason": shard_rec.reason
        })
        
        workers_rec = self.parameters.recommend_parallel_workers(
            cpu_count=kwargs.get("cpu_count", 4)
        )
        recommendations["recommendations"].append({
            "category": "parameters",
            "parameter": "parallel_workers",
            "value": workers_rec.item,
            "confidence": round(workers_rec.confidence, 2),
            "reason": workers_rec.reason
        })
        
        return recommendations
    
    def explain_recommendation(
        self,
        recommendation_type: str,
        choice: str
    ) -> str:
        """
        Explain why a recommendation was made.
        
        Args:
            recommendation_type: Type of recommendation
            choice: The chosen option
            
        Returns:
            Human-readable explanation
        """
        explanations = {
            ("strategy", "ultra"): 
                "Ultra strategy selected for maximum compression. "
                "Uses stride=0 tensors to achieve 99%+ size reduction. "
                "Best for storage and transfer, not for training.",
            ("strategy", "quantum"):
                "Quantum strategy selected for extreme quantization. "
                "Uses 1-bit weights with learned scaling. "
                "Good for edge deployment with minimal accuracy loss.",
            ("strategy", "compact"):
                "Compact strategy selected for balanced compression. "
                "Uses small random values for reasonable initialization. "
                "Good for development and experimentation.",
        }
        
        return explanations.get(
            (recommendation_type, choice),
            f"{choice} is recommended based on your requirements."
        )


# Global instance
_recommender: Optional[VitriolRecommender] = None


def get_recommender() -> VitriolRecommender:
    """Get global recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = VitriolRecommender()
    return _recommender
