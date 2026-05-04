"""Tests for vitriol.nas.targeted_nas module."""
from unittest.mock import MagicMock

from vitriol.nas.targeted_nas import (
    ObjectiveType,
    ConstraintType,
    Constraint,
    OptimizationTarget,
    ParetoSolution,
    MetricsCalculator,
    ConstraintOptimizer,
    MultiObjectiveOptimizer,
    DirectedMutator,
    create_constraint_optimizer,
    quick_search,
)
from vitriol.nas.search_space import ArchitectureGene


# ─────────────────────────────────────────────────────────────
# Constraint
# ─────────────────────────────────────────────────────────────

class TestConstraint:
    def test_max_params_satisfied(self):
        gene = MagicMock()
        c = Constraint(ConstraintType.MAX_PARAMS, 100)
        assert c.is_satisfied(gene, {"params": 50}) is True
        assert c.is_satisfied(gene, {"params": 150}) is False

    def test_max_vram_satisfied(self):
        gene = MagicMock()
        c = Constraint(ConstraintType.MAX_VRAM, 10)
        assert c.is_satisfied(gene, {"vram_gb": 5}) is True
        assert c.is_satisfied(gene, {"vram_gb": 15}) is False

    def test_max_layers(self):
        gene = MagicMock(n_layers=8)
        c = Constraint(ConstraintType.MAX_LAYERS, 10)
        assert c.is_satisfied(gene, {}) is True
        gene.n_layers = 12
        assert c.is_satisfied(gene, {}) is False

    def test_min_layers(self):
        gene = MagicMock(n_layers=8)
        c = Constraint(ConstraintType.MIN_LAYERS, 5)
        assert c.is_satisfied(gene, {}) is True
        gene.n_layers = 3
        assert c.is_satisfied(gene, {}) is False

    def test_max_hidden_size(self):
        gene = MagicMock(hidden_size=512)
        c = Constraint(ConstraintType.MAX_HIDDEN_SIZE, 1024)
        assert c.is_satisfied(gene, {}) is True
        gene.hidden_size = 2048
        assert c.is_satisfied(gene, {}) is False

    def test_attention_type(self):
        gene = MagicMock(attention_type="GQA")
        c = Constraint(ConstraintType.ATTENTION_TYPE, "GQA")
        assert c.is_satisfied(gene, {}) is True
        gene.attention_type = "MHA"
        assert c.is_satisfied(gene, {}) is False

    def test_attention_type_in(self):
        gene = MagicMock(attention_type="GQA")
        c = Constraint(ConstraintType.ATTENTION_TYPE, ["GQA", "MQA"], compare="in")
        assert c.is_satisfied(gene, {}) is True
        gene.attention_type = "MHA"
        assert c.is_satisfied(gene, {}) is False

    def test_ffn_type(self):
        gene = MagicMock(ffn_type="SwiGLU")
        c = Constraint(ConstraintType.FFN_TYPE, "SwiGLU")
        assert c.is_satisfied(gene, {}) is True
        gene.ffn_type = "GeLU"
        assert c.is_satisfied(gene, {}) is False


# ─────────────────────────────────────────────────────────────
# OptimizationTarget
# ─────────────────────────────────────────────────────────────

class TestOptimizationTarget:
    def test_minimize_params(self):
        t = OptimizationTarget(ObjectiveType.MINIMIZE_PARAMS, weight=1.0)
        assert t.score({"params": 100}) == -100

    def test_minimize_vram(self):
        t = OptimizationTarget(ObjectiveType.MINIMIZE_VRAM, weight=2.0)
        assert t.score({"vram_gb": 5}) == -10.0

    def test_maximize_score(self):
        t = OptimizationTarget(ObjectiveType.MAXIMIZE_SCORE, weight=1.0)
        assert t.score({"nas_score": 0.8}) == 0.8

    def test_maximize_efficiency(self):
        t = OptimizationTarget(ObjectiveType.MAXIMIZE_EFFICIENCY, weight=1.0)
        assert t.score({"params": 100, "vram_gb": 10}) == 10.0
        assert t.score({"params": 100, "vram_gb": 0}) == 0


# ─────────────────────────────────────────────────────────────
# MetricsCalculator
# ─────────────────────────────────────────────────────────────

class TestMetricsCalculator:
    def _make_gene(self, ffn_type="GeLU"):
        return ArchitectureGene(
            vocab_size=1000,
            hidden_size=64,
            n_layers=2,
            n_heads=4,
            attention_type="MHA",
            ffn_type=ffn_type,
            activation="gelu",
            norm_type="LayerNorm",
        )

    def test_calculate_params(self):
        gene = self._make_gene()
        params = MetricsCalculator.calculate_params(gene)
        assert params > 0
        assert isinstance(params, int)

    def test_calculate_params_swiglu(self):
        gene_swiglu = self._make_gene(ffn_type="SwiGLU")
        params_swiglu = MetricsCalculator.calculate_params(gene_swiglu)
        gene_geglu = self._make_gene(ffn_type="GeGLU")
        params_geglu = MetricsCalculator.calculate_params(gene_geglu)
        assert params_geglu == params_swiglu

    def test_calculate_vram(self):
        gene = self._make_gene()
        vram = MetricsCalculator.calculate_vram(gene)
        assert vram > 0

    def test_calculate_vram_fp32(self):
        gene = self._make_gene()
        vram_bf16 = MetricsCalculator.calculate_vram(gene, dtype="bfloat16")
        vram_fp32 = MetricsCalculator.calculate_vram(gene, dtype="float32")
        assert vram_fp32 > vram_bf16

    def test_calculate_flops(self):
        gene = self._make_gene()
        flops = MetricsCalculator.calculate_flops(gene, seq_len=512)
        assert flops > 0

    def test_calculate_all(self):
        gene = self._make_gene()
        metrics = MetricsCalculator.calculate_all(gene)
        assert "params" in metrics
        assert "params_millions" in metrics
        assert "vram_gb" in metrics
        assert "flops" in metrics
        assert "flops_per_param" in metrics


# ─────────────────────────────────────────────────────────────
# ConstraintOptimizer
# ─────────────────────────────────────────────────────────────

class TestConstraintOptimizer:
    def test_add_constraint(self):
        opt = ConstraintOptimizer()
        opt.add_constraint(Constraint(ConstraintType.MAX_PARAMS, 1e9))
        assert len(opt.constraints) == 1

    def test_add_objective(self):
        opt = ConstraintOptimizer()
        opt.add_objective(OptimizationTarget(ObjectiveType.MINIMIZE_PARAMS))
        assert len(opt.objectives) == 1

    def test_check_constraints_all_satisfied(self):
        gene = MagicMock(n_layers=4, hidden_size=128, attention_type="GQA", ffn_type="SwiGLU")
        opt = ConstraintOptimizer([
            Constraint(ConstraintType.MAX_LAYERS, 10),
            Constraint(ConstraintType.MAX_HIDDEN_SIZE, 256),
        ])
        ok, violated = opt.check_constraints(gene)
        assert ok is True
        assert violated == []

    def test_check_constraints_violated(self):
        gene = MagicMock(n_layers=20, hidden_size=128, attention_type="GQA", ffn_type="SwiGLU")
        opt = ConstraintOptimizer([
            Constraint(ConstraintType.MAX_LAYERS, 10),
        ])
        ok, violated = opt.check_constraints(gene)
        assert ok is False
        assert len(violated) == 1

    def test_evaluate_objectives(self):
        gene = MagicMock(n_layers=4, hidden_size=128, attention_type="GQA", ffn_type="SwiGLU")
        opt = ConstraintOptimizer(objectives=[
            OptimizationTarget(ObjectiveType.MINIMIZE_PARAMS, weight=1.0),
        ])
        score = opt.evaluate_objectives(gene, base_score=0.5)
        assert score < 0  # minimize params means negative score

    def test_optimize(self):
        from vitriol.nas.search_space import LLMSearchSpace
        opt = ConstraintOptimizer(
            constraints=[Constraint(ConstraintType.MAX_LAYERS, 100)],
            objectives=[OptimizationTarget(ObjectiveType.MAXIMIZE_SCORE)],
        )
        search_space = LLMSearchSpace()
        gene, score, metrics = opt.optimize(search_space, None, n_iterations=5, verbose=False)
        # May find a solution or not
        if gene is not None:
            assert isinstance(metrics, dict)


# ─────────────────────────────────────────────────────────────
# MultiObjectiveOptimizer
# ─────────────────────────────────────────────────────────────

class TestMultiObjectiveOptimizer:
    def test_dominates_better_in_all(self):
        opt = MultiObjectiveOptimizer()
        sol1 = ParetoSolution(
            gene=MagicMock(),
            objectives={ObjectiveType.MINIMIZE_PARAMS: 10},
            metrics={},
        )
        sol2 = ParetoSolution(
            gene=MagicMock(),
            objectives={ObjectiveType.MINIMIZE_PARAMS: 20},
            metrics={},
        )
        assert opt._dominates(sol1, sol2) is True
        assert opt._dominates(sol2, sol1) is False

    def test_dominates_worse_in_one(self):
        opt = MultiObjectiveOptimizer(objectives=[
            ObjectiveType.MINIMIZE_PARAMS,
            ObjectiveType.MINIMIZE_VRAM,
        ])
        sol1 = ParetoSolution(
            gene=MagicMock(),
            objectives={ObjectiveType.MINIMIZE_PARAMS: 10, ObjectiveType.MINIMIZE_VRAM: 5},
            metrics={},
        )
        sol2 = ParetoSolution(
            gene=MagicMock(),
            objectives={ObjectiveType.MINIMIZE_PARAMS: 20, ObjectiveType.MINIMIZE_VRAM: 3},
            metrics={},
        )
        # sol1 is better in params but worse in vram -> does not dominate
        assert opt._dominates(sol1, sol2) is False

    def test_update_pareto_front_adds_new(self):
        opt = MultiObjectiveOptimizer()
        sol = ParetoSolution(
            gene=MagicMock(),
            objectives={ObjectiveType.MINIMIZE_PARAMS: 10},
            metrics={},
        )
        front, was_dominated = opt._update_pareto_front(sol, [])
        assert len(front) == 1
        assert was_dominated is False

    def test_optimize(self):
        from vitriol.nas.search_space import LLMSearchSpace
        opt = MultiObjectiveOptimizer()
        search_space = LLMSearchSpace()
        front = opt.optimize(search_space, None, n_iterations=5, verbose=False)
        assert isinstance(front, list)


# ─────────────────────────────────────────────────────────────
# DirectedMutator
# ─────────────────────────────────────────────────────────────

class TestDirectedMutator:
    def test_set_target(self):
        m = DirectedMutator()
        m.set_target(ObjectiveType.MINIMIZE_PARAMS, target_value=1e9)
        assert m.target_objective == ObjectiveType.MINIMIZE_PARAMS
        assert m.target_value == 1e9

    def test_mutate_toward_no_target(self):
        from vitriol.nas.search_space import LLMSearchSpace
        m = DirectedMutator()
        search_space = LLMSearchSpace()
        gene = search_space.sample()
        result = m.mutate_toward(gene, search_space)
        assert result is not None

    def test_mutate_toward_minimize_params(self):
        from vitriol.nas.search_space import LLMSearchSpace
        m = DirectedMutator()
        m.set_target(ObjectiveType.MINIMIZE_PARAMS)
        search_space = LLMSearchSpace()
        gene = search_space.sample()
        result = m.mutate_toward(gene, search_space)
        assert result is not None


# ─────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────

class TestFactoryFunctions:
    def test_create_constraint_optimizer(self):
        opt = create_constraint_optimizer(max_vram_gb=24, max_params_m=70, attention_type="GQA")
        assert isinstance(opt, ConstraintOptimizer)
        assert len(opt.constraints) == 3

    def test_create_constraint_optimizer_minimal(self):
        opt = create_constraint_optimizer()
        assert isinstance(opt, ConstraintOptimizer)
        assert len(opt.constraints) == 0

    def test_quick_search(self):
        gene, metrics = quick_search(target_vram_gb=100, n_iterations=3)
        if gene is not None:
            assert isinstance(metrics, dict)
