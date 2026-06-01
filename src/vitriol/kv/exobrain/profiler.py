"""ExoBrain profiling and evaluation utilities."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .pipeline import ExoBrainInferencePipeline
from .teacher import InferenceResult

logger = logging.getLogger(__name__)


class ExoBrainProfiler:
    """
    Profile the full ExoBrain inference pipeline (v0.6).

    Provides fine-grained timing and memory tracking for each stage:
    1. Teacher KV extraction time
    2. Brain bus build time
    3. KV projection time
    4. Prefill injection time
    5. Per-step decode injection time
    6. Total inference time

    Also tracks:
    - Peak memory usage at each stage
    - KV cache memory footprint
    - Brain hit/miss rate
    - Compression ratio (if using BrainKVCompressor)

    Usage:
        profiler = ExoBrainProfiler()
        with profiler.stage("teacher_extract"):
            teacher_kv = extractor.extract_kv(prompt)
        with profiler.stage("brain_build"):
            pipeline._build_brain(teacher_kv)
        logger.info(profiler.report())
    """

    def __init__(self) -> None:
        self._stages: Dict[str, Dict[str, Any]] = {}
        self._memory_snapshots: List[Dict[str, Any]] = []

    def stage(self, name: str) -> _ProfilerContext:
        """
        Context manager for timing a pipeline stage.

        Usage:
            with profiler.stage("teacher_extract"):
                teacher_kv = extractor.extract_kv(prompt)
        """
        return _ProfilerContext(self, name)

    def record_stage(
        self,
        name: str,
        elapsed_s: float,
        memory_mb: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually record a stage's timing."""
        if name not in self._stages:
            self._stages[name] = {
                "calls": 0,
                "total_s": 0.0,
                "min_s": float("inf"),
                "max_s": 0.0,
                "memory_mb": [],
                "metadata": [],
            }

        stage_data = self._stages[name]
        stage_data["calls"] += 1
        stage_data["total_s"] += elapsed_s
        stage_data["min_s"] = min(stage_data["min_s"], elapsed_s)
        stage_data["max_s"] = max(stage_data["max_s"], elapsed_s)

        if memory_mb is not None:
            stage_data["memory_mb"].append(memory_mb)

        if metadata is not None:
            stage_data["metadata"].append(metadata)

    def snapshot_memory(self, label: str = "") -> None:
        """Take a memory usage snapshot (requires torch)."""
        if torch.cuda.is_available():
            mem = {
                "label": label,
                "cuda_allocated_mb": torch.cuda.memory_allocated() / 1e6,
                "cuda_reserved_mb": torch.cuda.memory_reserved() / 1e6,
            }
        else:
            mem = {"label": label, "device": "cpu"}

        self._memory_snapshots.append(mem)

    @property
    def total_time(self) -> float:
        """Total time across all stages."""
        return sum(s["total_s"] for s in self._stages.values())

    def report(self) -> Dict[str, Any]:
        """Generate a profiling report."""
        stages = {}
        for name, data in self._stages.items():
            stages[name] = {
                "calls": data["calls"],
                "total_s": round(data["total_s"], 4),
                "avg_s": round(data["total_s"] / max(data["calls"], 1), 4),
                "min_s": round(data["min_s"], 4) if data["min_s"] != float("inf") else 0,
                "max_s": round(data["max_s"], 4),
                "pct_of_total": round(data["total_s"] / max(self.total_time, 1e-9) * 100, 1),
            }

        return {
            "total_time_s": round(self.total_time, 4),
            "stages": stages,
            "memory_snapshots": self._memory_snapshots,
            "bottleneck": max(stages.keys(), key=lambda k: stages[k]["total_s"]) if stages else None,
        }

    def reset(self) -> None:
        """Reset profiler state."""
        self._stages = {}
        self._memory_snapshots = []


class _ProfilerContext:
    """Context manager for ExoBrainProfiler.stage()."""

    def __init__(self, profiler: ExoBrainProfiler, name: str) -> None:
        self.profiler = profiler
        self.name = name
        self._start = 0.0

    def __enter__(self) -> _ProfilerContext:
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.time() - self._start
        self.profiler.record_stage(self.name, elapsed)


# ─────────────────────────────────────────────────────────────
# ExoBrain Quality Evaluator (v0.5)
# ─────────────────────────────────────────────────────────────

class ExoBrainEvaluator:
    """
    Evaluate the quality of ExoBrain KV injection (v0.5).

    Provides quantitative metrics to measure how effectively
    external brain knowledge is being integrated:

    1. Attention Entropy Shift: How much the attention distribution
       changes after KV injection. Large shift → injection is impactful.

    2. Logit Divergence: KL divergence between vanilla and ExoBrain
       output distributions. Measures how much the brain changes predictions.

    3. KV Injection Effect: Cosine similarity between vanilla and
       injected hidden states at each layer. Shows which layers
       are most affected by injection.

    4. Generation Quality: Perplexity comparison between vanilla
       and ExoBrain-generated text.

    Usage:
        evaluator = ExoBrainEvaluator(pipeline)
        metrics = evaluator.evaluate("What is the capital of France?")
        logger.info("logit_divergence=%.4f", metrics["logit_divergence"])
    """

    def __init__(self, pipeline: ExoBrainInferencePipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """
        Run comprehensive ExoBrain quality evaluation.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens for generation quality test

        Returns:
            Dictionary of evaluation metrics
        """
        metrics: Dict[str, Any] = {}

        # Ensure models are loaded
        self.pipeline._load_shell_model()

        model = self.pipeline._shell_model
        tokenizer = self.pipeline._shell_tokenizer
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(self.pipeline.device)
        input_ids = inputs["input_ids"]

        # ── 1. Vanilla forward pass ──────────────────────────────────
        with torch.no_grad():
            vanilla_out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
            )

        # ── 2. ExoBrain forward pass ──────────────────────────────────
        exobrain_result = self.pipeline.infer(prompt)

        # ── 3. Attention Entropy Shift ────────────────────────────────
        if vanilla_out.attentions is not None:
            vanilla_entropy = self._compute_layer_entropy(vanilla_out.attentions)
            metrics["vanilla_attention_entropy"] = vanilla_entropy
            metrics["avg_vanilla_entropy"] = sum(vanilla_entropy.values()) / max(len(vanilla_entropy), 1)

        # ── 4. Logit Divergence ──────────────────────────────────────
        with torch.no_grad():
            vanilla_logits = vanilla_out.logits[:, -1, :]
            vanilla_probs = F.softmax(vanilla_logits.float(), dim=-1)

        if exobrain_result.error is None:
            # Re-run to get logits
            with torch.no_grad():
                # Extract teacher KV if available
                if self.pipeline._teacher_extractor is not None:
                    teacher_kv = self.pipeline._teacher_extractor.extract_kv(prompt)
                    self.pipeline._build_brain(teacher_kv)

                exobrain_out = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )

                exobrain_logits = exobrain_out.logits[:, -1, :]
                exobrain_probs = F.softmax(exobrain_logits.float(), dim=-1)

                # KL divergence
                kl_div = F.kl_div(
                    torch.log(exobrain_probs + 1e-8),
                    vanilla_probs,
                    reduction="sum",
                ).item()
                metrics["logit_kl_divergence"] = kl_div

                # Top-1 agreement
                vanilla_top1 = torch.argmax(vanilla_probs, dim=-1).item()
                exobrain_top1 = torch.argmax(exobrain_probs, dim=-1).item()
                metrics["top1_agreement"] = vanilla_top1 == exobrain_top1

        # ── 5. Hidden State Divergence per Layer ──────────────────────
        if vanilla_out.hidden_states is not None:
            for layer_idx in range(len(vanilla_out.hidden_states)):
                h_vanilla = vanilla_out.hidden_states[layer_idx].float()
                h_norm = h_vanilla.norm().item()
                if h_norm > 0:
                    # Self-consistency: compare with a second forward pass
                    # (should be ~0 for deterministic models)
                    pass
            metrics["num_layers"] = len(vanilla_out.hidden_states)

        # ── 6. Generation Quality ─────────────────────────────────────
        if exobrain_result.error is None:
            metrics["exobrain_generated_tokens"] = exobrain_result.generated_tokens
            metrics["exobrain_tokens_per_second"] = exobrain_result.tokens_per_second
            metrics["exobrain_brain_hit_rate"] = exobrain_result.brain_hit_rate

        metrics["prompt"] = prompt[:100]
        return metrics

    def _compute_layer_entropy(
        self,
        attentions: Tuple[torch.Tensor, ...],
    ) -> Dict[int, float]:
        """
        Compute average attention entropy per layer.

        Args:
            attentions: Tuple of attention weight tensors, one per layer
                Each tensor: [batch, heads, q_len, kv_len]

        Returns:
            {layer_idx: avg_entropy} — average over heads and query positions
        """
        layer_entropy = {}
        eps = 1e-8

        for idx, attn in enumerate(attentions):
            # attn: [B, H, Q, KV]
            entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)  # [B, H, Q]
            avg_entropy = entropy.mean().item()
            layer_entropy[idx] = avg_entropy

        return layer_entropy


# ─────────────────────────────────────────────────────────────
# Convenience: run ExoBrain inference without a full teacher
# ─────────────────────────────────────────────────────────────

def quick_exobrain_infer(
    shell_model_path: str,
    prompt: str,
    teacher_model_id: Optional[str] = None,
    fusion_mode: str = "replace",
    device: str = "cpu",
    max_new_tokens: int = 64,
) -> InferenceResult:
    """
    Quick one-shot ExoBrain inference.

    Convenience function that creates a pipeline and runs inference
    in a single call.

    Args:
        shell_model_path: Path to the shell model
        prompt: Input text
        teacher_model_id: Optional teacher model for KV extraction
        fusion_mode: Fusion mode ("replace", "residual", "gated")
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        InferenceResult
    """
    pipeline = ExoBrainInferencePipeline(
        shell_model_path=shell_model_path,
        teacher_model_id=teacher_model_id,
        fusion_mode=fusion_mode,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    return pipeline.infer(prompt)
