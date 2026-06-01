"""
Targeted NAS - Constraint Optimization and Multi-Objective Search
===============================================================

Enhanced NAS with:
- Constraint-based optimization (max VRAM, max params)
- Multi-objective optimization (Pareto front)
- Directed mutation for specific targets
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .search_space import ArchitectureGene, LLMSearchSpace

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums and Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MINIMIZE_PARAMS = "minimize_params"
    MINIMIZE_VRAM = "minimize_vram"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_SCORE = "maximize_score"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"


class ConstraintType(Enum):
    """Types of constraints."""
    MAX_PARAMS = "max_params"
    MAX_VRAM = "max_vram"
    MAX_LAYERS = "max_layers"
    MIN_LAYERS = "min_layers"
    MAX_HIDDEN_SIZE = "max_hidden_size"
    ATTENTION_TYPE = "attention_type"
    FFN_TYPE = "ffn_type"


@dataclass
class Constraint:
    """A constraint for NAS search."""
    constraint_type: ConstraintType
    value: Any
    compare: str = "<="  # <=, >=, ==, !=, in

    def is_satisfied(self, gene: ArchitectureGene, metrics: Dict[str, float]) -> bool:
        """Check if constraint is satisfied by a gene."""
        if self.constraint_type == ConstraintType.MAX_PARAMS:
            return metrics.get("params", float("inf")) <= self.value
        elif self.constraint_type == ConstraintType.MAX_VRAM:
            return metrics.get("vram_gb", float("inf")) <= self.value
        elif self.constraint_type == ConstraintType.MAX_LAYERS:
            return gene.n_layers <= self.value
        elif self.constraint_type == ConstraintType.MIN_LAYERS:
            return gene.n_layers >= self.value
        elif self.constraint_type == ConstraintType.MAX_HIDDEN_SIZE:
            return gene.hidden_size <= self.value
        elif self.constraint_type == ConstraintType.ATTENTION_TYPE:
            if self.compare == "in":
                return gene.attention_type in self.value
            return gene.attention_type == self.value
        elif self.constraint_type == ConstraintType.FFN_TYPE:
            if self.compare == "in":
                return gene.ffn_type in self.value
            return gene.ffn_type == self.value
        return True


@dataclass
class OptimizationTarget:
    """An objective to optimize toward."""
    objective_type: ObjectiveType
    target_value: Optional[float] = None
    weight: float = 1.0

    def score(self, metrics: Dict[str, float]) -> float:
        """Calculate score for this objective based on metrics."""
        if self.objective_type == ObjectiveType.MINIMIZE_PARAMS:
            return -metrics.get("params", 0) * self.weight
        elif self.objective_type == ObjectiveType.MINIMIZE_VRAM:
            return -metrics.get("vram_gb", 0) * self.weight
        elif self.objective_type == ObjectiveType.MINIMIZE_LATENCY:
            return -metrics.get("latency_ms", 0) * self.weight
        elif self.objective_type == ObjectiveType.MAXIMIZE_SCORE:
            return metrics.get("nas_score", 0) * self.weight
        elif self.objective_type == ObjectiveType.MAXIMIZE_EFFICIENCY:
            params = metrics.get("params", 1)
            vram = metrics.get("vram_gb", 1)
            if vram > 0:
                return (params / vram) * self.weight
            return 0
        return 0


@dataclass
class ParetoSolution:
    """A solution on the Pareto front."""
    gene: ArchitectureGene
    objectives: Dict[ObjectiveType, float]
    metrics: Dict[str, float]
    dominates_count: int = 0
    dominated_by: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(str(self.gene.to_dict()))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """Calculate architecture metrics for constraint checking."""

    BYTES_PER_PARAM_BF16 = 2
    BYTES_PER_PARAM_FP32 = 4
    KV_CACHE_FACTOR = 0.1

    @classmethod
    def calculate_params(cls, gene: ArchitectureGene) -> int:
        """Estimate total parameter count."""
        V = gene.vocab_size
        H = gene.hidden_size
        N = gene.n_layers
        I_size = gene.intermediate_size

        emb_params = V * H
        qkv_params = 3 * H * H
        o_params = H * H
        attn_params = qkv_params + o_params

        if gene.ffn_type in ["SwiGLU", "GeGLU"]:
            ffn_params = 3 * H * I_size
        else:
            ffn_params = 2 * H * I_size

        ln_params = 2 * H
        layer_params = attn_params + ffn_params + ln_params
        transformer_params = layer_params * N
        head_params = H * V
        final_norm = H

        total = emb_params + transformer_params + head_params + final_norm
        return int(total)

    @classmethod
    def calculate_vram(cls, gene: ArchitectureGene, dtype: str = "bfloat16") -> float:
        """Estimate VRAM usage in GB."""
        params = cls.calculate_params(gene)
        bytes_per_param = cls.BYTES_PER_PARAM_BF16 if dtype == "bfloat16" else cls.BYTES_PER_PARAM_FP32

        model_vram = params * bytes_per_param / (1024 ** 3)
        kv_vram = model_vram * cls.KV_CACHE_FACTOR
        activation_vram = model_vram * 0.5 if dtype == "bfloat16" else model_vram

        return model_vram + kv_vram + activation_vram

    @classmethod
    def calculate_flops(cls, gene: ArchitectureGene, seq_len: int = 512) -> float:
        """Estimate FLOPs per token."""
        H = gene.hidden_size
        N = gene.n_layers
        I_size = gene.intermediate_size

        attn_flops = 6 * H * H * seq_len
        ffn_flops = 4 * H * I_size * seq_len
        layer_flops = attn_flops + ffn_flops
        total_flops = layer_flops * N

        return float(total_flops)

    @classmethod
    def calculate_all(cls, gene: ArchitectureGene) -> Dict[str, float]:
        """Calculate all metrics."""
        params = cls.calculate_params(gene)
        vram_gb = cls.calculate_vram(gene)
        flops = cls.calculate_flops(gene)

        return {
            "params": params,
            "params_millions": params / 1e6,
            "vram_gb": vram_gb,
            "flops": flops,
            "flops_per_param": flops / params if params > 0 else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constraint Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintOptimizer:
    """
    Optimizer that respects user-defined constraints.
    """

    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        objectives: Optional[List[OptimizationTarget]] = None,
    ):
        self.constraints = constraints or []
        self.objectives = objectives or []

    def add_constraint(self, constraint: Constraint) -> ConstraintOptimizer:
        self.constraints.append(constraint)
        return self

    def add_objective(self, objective: OptimizationTarget) -> ConstraintOptimizer:
        self.objectives.append(objective)
        return self

    def check_constraints(self, gene: ArchitectureGene) -> Tuple[bool, List[str]]:
        """Check if gene satisfies all constraints."""
        metrics = MetricsCalculator.calculate_all(gene)
        violated = []

        for constraint in self.constraints:
            if not constraint.is_satisfied(gene, metrics):
                violated.append(
                    f"{constraint.constraint_type.value} ({constraint.compare} {constraint.value})"
                )

        return len(violated) == 0, violated

    def evaluate_objectives(self, gene: ArchitectureGene, base_score: float = 0) -> float:
        """Evaluate combined objective score."""
        metrics = MetricsCalculator.calculate_all(gene)
        metrics["nas_score"] = base_score

        total_score = 0
        for objective in self.objectives:
            total_score += objective.score(metrics)

        return total_score

    def optimize(
        self,
        search_space: LLMSearchSpace,
        base_evaluator,
        n_iterations: int = 50,
        verbose: bool = True,
    ) -> Tuple[ArchitectureGene, float, Dict[str, float]]:
        """Run constraint-optimized random search."""
        best_gene = None
        best_score = -float("inf")
        best_metrics = {}

        for i in range(n_iterations):
            gene = search_space.sample()
            satisfied, violated = self.check_constraints(gene)

            if not satisfied:
                continue

            base_score = random.random()
            score = self.evaluate_objectives(gene, base_score)

            if score > best_score:
                best_score = score
                best_gene = gene
                best_metrics = MetricsCalculator.calculate_all(gene)
                if verbose:
                    logger.info(
                        f"Iter {i}: New best score={score:.4f}, "
                        f"params={best_metrics['params_millions']:.1f}M, "
                        f"vram={best_metrics['vram_gb']:.2f}GB"
                    )

        return best_gene, best_score, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Objective Optimizer (Pareto)
# ─────────────────────────────────────────────────────────────────────────────

class MultiObjectiveOptimizer:
    """Multi-objective optimizer that finds Pareto front."""

    def __init__(self, objectives: Optional[List[ObjectiveType]] = None):
        self.objectives = objectives or [
            ObjectiveType.MINIMIZE_PARAMS,
            ObjectiveType.MAXIMIZE_SCORE,
        ]

    def add_objective(self, obj_type: ObjectiveType) -> MultiObjectiveOptimizer:
        self.objectives.append(obj_type)
        return self

    def _compute_objective_values(
        self,
        gene: ArchitectureGene,
        base_score: float = 0,
    ) -> Dict[ObjectiveType, float]:
        """Compute values for all objectives."""
        metrics = MetricsCalculator.calculate_all(gene)
        metrics["nas_score"] = base_score

        values = {}
        for obj_type in self.objectives:
            target = OptimizationTarget(obj_type, weight=1.0)
            values[obj_type] = -target.score(metrics)

        return values

    def _dominates(
        self,
        sol1: ParetoSolution,
        sol2: ParetoSolution,
    ) -> bool:
        """Check if sol1 dominates sol2 (in minimization context)."""
        better_in_any = False

        for obj_type in self.objectives:
            v1 = sol1.objectives.get(obj_type, 0)
            v2 = sol2.objectives.get(obj_type, 0)

            if v1 > v2:
                return False
            if v1 < v2:
                better_in_any = True

        return better_in_any

    def _update_pareto_front(
        self,
        new_sol: ParetoSolution,
        front: List[ParetoSolution],
    ) -> Tuple[List[ParetoSolution], bool]:
        """Update Pareto front with new solution."""
        for sol in front:
            if self._dominates(sol, new_sol):
                return front, True

        new_front = []
        for sol in front:
            if not self._dominates(new_sol, sol):
                new_front.append(sol)

        new_front.append(new_sol)
        return new_front, False

    def optimize(
        self,
        search_space: LLMSearchSpace,
        base_evaluator,
        n_iterations: int = 100,
        verbose: bool = True,
    ) -> List[ParetoSolution]:
        """Find Pareto front using NSGA-II inspired approach."""
        pareto_front: List[ParetoSolution] = []

        for i in range(n_iterations):
            gene = search_space.sample()
            obj_values = self._compute_objective_values(gene, random.random())
            metrics = MetricsCalculator.calculate_all(gene)

            sol = ParetoSolution(
                gene=gene,
                objectives=obj_values,
                metrics=metrics,
            )

            old_size = len(pareto_front)
            pareto_front, was_dominated = self._update_pareto_front(sol, pareto_front)

            if was_dominated:
                if verbose and i % 20 == 0:
                    logger.debug(f"Iter {i}: Solution dominated, front size={len(pareto_front)}")
            elif len(pareto_front) > old_size and verbose:
                logger.info(f"Iter {i}: New Pareto solution! Front size={len(pareto_front)}")

        pareto_front.sort(key=lambda s: list(s.objectives.values())[0])

        if verbose:
            logger.info(f"Pareto front contains {len(pareto_front)} solutions")

        return pareto_front


# ─────────────────────────────────────────────────────────────────────────────
# Directed Mutator
# ─────────────────────────────────────────────────────────────────────────────

class DirectedMutator:
    """Mutation strategies that bias toward specific targets."""

    def __init__(self):
        self.target_objective: Optional[ObjectiveType] = None
        self.target_value: Optional[float] = None

    def set_target(
        self,
        objective: ObjectiveType,
        target_value: Optional[float] = None,
    ) -> DirectedMutator:
        self.target_objective = objective
        self.target_value = target_value
        return self

    def mutate_toward(
        self,
        gene: ArchitectureGene,
        search_space: LLMSearchSpace,
    ) -> ArchitectureGene:
        """Mutate gene in direction of target objective."""
        if self.target_objective is None:
            return search_space.mutate(gene)

        if self.target_objective == ObjectiveType.MINIMIZE_PARAMS:
            bias = -1
        elif self.target_objective == ObjectiveType.MINIMIZE_VRAM:
            bias = -1
        elif self.target_objective == ObjectiveType.MAXIMIZE_EFFICIENCY:
            bias = 1
        else:
            bias = 0

        new_gene_dict = gene.to_dict()

        if bias < 0:
            if random.random() < 0.5 and gene.n_layers > 4:
                new_gene_dict["n_layers"] = max(4, gene.n_layers - 2)
            if random.random() < 0.3 and gene.hidden_size > 256:
                options = [h for h in search_space.hidden_size_choices if h < gene.hidden_size]
                if options:
                    new_gene_dict["hidden_size"] = max(256, min(options))
        elif bias > 0:
            if gene.attention_type == "MHA" and random.random() < 0.5:
                new_gene_dict["attention_type"] = "GQA"

        return search_space.mutate(
            ArchitectureGene.from_dict(new_gene_dict),
            mutation_rate=0.3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_constraint_optimizer(
    max_vram_gb: Optional[float] = None,
    max_params_m: Optional[float] = None,
    attention_type: Optional[str] = None,
) -> ConstraintOptimizer:
    """Factory to create common constraint optimizers."""
    optimizer = ConstraintOptimizer()

    if max_vram_gb is not None:
        optimizer.add_constraint(Constraint(ConstraintType.MAX_VRAM, max_vram_gb))

    if max_params_m is not None:
        optimizer.add_constraint(Constraint(ConstraintType.MAX_PARAMS, max_params_m * 1e6))

    if attention_type is not None:
        optimizer.add_constraint(Constraint(ConstraintType.ATTENTION_TYPE, attention_type))

    return optimizer


def quick_search(
    target_vram_gb: float,
    n_iterations: int = 50,
) -> Tuple[ArchitectureGene, Dict[str, float]]:
    """Quick constrained search for target VRAM."""
    optimizer = create_constraint_optimizer(max_vram_gb=target_vram_gb)
    optimizer.add_objective(OptimizationTarget(ObjectiveType.MAXIMIZE_EFFICIENCY))

    search_space = LLMSearchSpace()
    gene, score, metrics = optimizer.optimize(search_space, None, n_iterations=n_iterations, verbose=True)

    return gene, metrics


__all__ = [
    "ObjectiveType",
    "ConstraintType",
    "Constraint",
    "OptimizationTarget",
    "ParetoSolution",
    "ConstraintOptimizer",
    "MultiObjectiveOptimizer",
    "DirectedMutator",
    "MetricsCalculator",
    "create_constraint_optimizer",
    "quick_search",
]
