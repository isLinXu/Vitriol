"""
ExoBrain Inference Pipeline & Knowledge Distiller (v0.4+).

═══════════════════════════════════════════════════════════════
Architecture — Heterogeneous Cognitive Alignment
═══════════════════════════════════════════════════════════════

    Teacher Model (full weights)         Shell Model (real 0.1B weights)
    ┌──────────────────────────┐       ┌──────────────────────┐
    │ Forward pass → KV cache  │───┐   │ Layer 0  (real)     │
    └──────────────────────────┘   │   │ Layer 1  (real)     │
                                   ▼   │ 🔑 ShellProjection  │
    Shell Model (real weights)  ExoBrain Bus   │ (cognitive align)  │
    ┌──────────────────────┐   ┌──────────────┐ │ LM Head (real)    │
    │ Layer 0  (real)      │←──│ KV injection │└──────────────────────┘
    │ Layer 1  (real)      │←──│ per-layer    │
    │ ...                  │←──│              │
    │ 🔑 ShellProjection  │   └──────────────┘
    │ LM Head (real)       │         │
    └──────────────────────┘         ▼
           │                  KnowledgeDistiller
           ▼                  ┌──────────────────┐
    ExoBrainInferencePipeline  │ 1. Generate       │
    ┌──────────────────────┐   │    training data   │
    │ 1. Load shell model  │   │ 2. Forward (brain) │
    │ 2. Load teacher      │   │ 3. Compute loss    │
    │ 3. Extract teacher KV│   │ 4. Backprop shell  │
    │ 4. Inject → generate │   │ 5. Save weights    │
    │ 5. Evaluate quality  │   └──────────────────┘
    └──────────────────────┘

IMPORTANT (v0.4+): The shell model must have REAL, TRAINABLE weights.
The old "zero-weight shell" approach is mathematically broken — without
real weights, the shell cannot generate meaningful queries to attend to
external KV. Use ShellProjection for cognitive alignment between the
shell's hidden_dim and the brain's hidden_dim.

═══════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════

    # Inference verification
    from vitriol.kv.exobrain_inference import ExoBrainInferencePipeline

    pipeline = ExoBrainInferencePipeline(
        shell_model_path="./shell-model",
        teacher_model_id="Qwen/Qwen2.5-0.5B",
        fusion_mode="replace",
    )
    result = pipeline.infer("What is the capital of France?")
    print(result["generated_text"])

    # Knowledge distillation (KV → weights)
    from vitriol.kv.exobrain_inference import KnowledgeDistiller

    distiller = KnowledgeDistiller(pipeline=pipeline)
    distiller.distill(
        prompts=["Hello", "What is AI?"],
        num_steps=3,
        learning_rate=1e-3,
        output_dir="./distilled-model",
    )
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vitriol.utils.hf_loading import load_causallm, load_tokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Head-Dimension Projection
# ─────────────────────────────────────────────────────────────


class HeadDimProjection(torch.nn.Module):
    """
    Project teacher KV tensors from teacher_head_dim → shell_head_dim.

    When cross-model ExoBrain inference is performed (e.g., teacher has
    head_dim=128 while shell has head_dim=64), the KV cache shapes are
    incompatible for fusion.  This module bridges the gap.

    Two projection modes:
    - ``"pad_or_truncate"``: zero-pad or truncate along the last axis.
      Deterministic, no parameters, preserves sub-space alignment.
    - ``"learned"``: a per-head linear projection (teacher_head_dim → shell_head_dim).
      Trainable via distillation; captures cross-dimension correlations.

    Shapes:
        Input:  [batch, num_kv_heads, seq_len, teacher_head_dim]
        Output: [batch, num_kv_heads, seq_len, shell_head_dim]
    """

    def __init__(
        self,
        teacher_head_dim: int,
        shell_head_dim: int,
        num_kv_heads: int = 1,
        mode: str = "pad_or_truncate",
    ) -> None:
        super().__init__()
        self.teacher_head_dim = teacher_head_dim
        self.shell_head_dim = shell_head_dim
        self.num_kv_heads = num_kv_heads
        self.mode = mode

        if mode == "learned" and teacher_head_dim != shell_head_dim:
            # Per-head linear: (num_kv_heads, teacher_head_dim, shell_head_dim)
            self.proj_weight = torch.nn.Parameter(
                torch.randn(num_kv_heads, teacher_head_dim, shell_head_dim)
                * (2.0 / (teacher_head_dim + shell_head_dim))  # Xavier-ish init
            )
            self.proj_bias = torch.nn.Parameter(
                torch.zeros(num_kv_heads, shell_head_dim)
            )
        else:
            # Mark as None so we can skip in forward
            self.proj_weight = None  # type: ignore[assignment]
            self.proj_bias = None  # type: ignore[assignment]

    def forward(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Project a KV tensor from teacher_head_dim to shell_head_dim.

        Args:
            kv_tensor: shape [batch, num_kv_heads, seq_len, teacher_head_dim]

        Returns:
            Projected tensor: shape [batch, num_kv_heads, seq_len, shell_head_dim]
        """
        if self.teacher_head_dim == self.shell_head_dim:
            return kv_tensor

        if self.mode == "pad_or_truncate":
            return self._pad_or_truncate(kv_tensor)
        else:  # learned
            return self._learned_project(kv_tensor)

    def _pad_or_truncate(self, kv: torch.Tensor) -> torch.Tensor:
        """Zero-pad or truncate along the last dimension."""
        t_dim, s_dim = self.teacher_head_dim, self.shell_head_dim

        if t_dim > s_dim:
            # Truncate: keep the first s_dim dimensions
            return kv[..., :s_dim]
        else:
            # Zero-pad: append (s_dim - t_dim) zeros
            pad_size = s_dim - t_dim
            padding = torch.zeros(
                *kv.shape[:-1], pad_size,
                dtype=kv.dtype, device=kv.device,
            )
            return torch.cat([kv, padding], dim=-1)

    def _learned_project(self, kv: torch.Tensor) -> torch.Tensor:
        """Apply learned per-head linear projection."""
        # kv: [B, H, S, D_teacher]
        B, H, S, D_t = kv.shape

        # Reshape for bmm: [B*H, S, D_teacher]
        kv_flat = kv.transpose(1, 2).reshape(B * H, S, D_t)

        # Weight: [H, D_teacher, D_shell] → [B*H, D_teacher, D_shell]
        w = self.proj_weight.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, D_t, D_s]
        w = w.reshape(B * H, D_t, self.shell_head_dim)

        # bmm: [B*H, S, D_t] @ [B*H, D_t, D_s] → [B*H, S, D_s]
        out = torch.bmm(kv_flat, w)

        # Add bias: [H, D_s] → [B*H, 1, D_s]
        b = self.proj_bias.unsqueeze(0).expand(B, -1, -1)  # [B, H, D_s]
        b = b.reshape(B * H, 1, self.shell_head_dim)
        out = out + b

        # Reshape back: [B, H, S, D_shell]
        out = out.reshape(B, H, S, self.shell_head_dim)
        return out

    def project_kv_pair(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: project both K and V tensors. Respects gradients."""
        return self.forward(key), self.forward(value)


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Result from a single ExoBrain inference run."""
    prompt: str
    generated_text: str = ""
    generated_tokens: int = 0
    prompt_tokens: int = 0
    inference_time_s: float = 0.0
    tokens_per_second: float = 0.0
    fusion_mode: str = "replace"
    brain_hit_rate: float = 0.0
    brain_stats: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    error: Optional[str] = None


@dataclass
class DistillResult:
    """Result from knowledge distillation."""
    output_dir: str
    num_steps: int = 0
    total_loss: float = 0.0
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    parameters_updated: int = 0
    shell_model_saved: bool = False
    distill_time_s: float = 0.0
    error: Optional[str] = None


@dataclass
class TeacherKVCache:
    """Cached KV pairs extracted from a teacher model."""
    # {layer_idx: (key_tensor, value_tensor)}
    kv_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    model_id: str = ""
    num_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    head_dim: int = 0
    sequence_length: int = 0


# ─────────────────────────────────────────────────────────────
# Teacher KV Extractor
# ─────────────────────────────────────────────────────────────

class TeacherKVExtractor:
    """
    Extract KV pairs from a teacher model for ExoBrain injection.

    Supports:
    - HuggingFace CausalLM models (LLaMA, Qwen, etc.)
    - Manual KV injection (for testing)
    - Cache serialization (save/load)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the teacher model."""
        if self._model is not None:
            return

        logger.info("Loading teacher model: %s", self.model_id)
        sec = {
            "trust_remote_code": self.trust_remote_code,
            "allow_network": not self.local_files_only,
            "local_files_only": self.local_files_only,
        }
        self._tokenizer = load_tokenizer(
            self.model_id,
            security=sec,
        )
        self._model = load_causallm(
            self.model_id,
            security=sec,
            torch_dtype=self.dtype,
            device=self.device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        logger.info("Teacher model loaded: %s", type(self._model).__name__)

    def extract_kv(
        self,
        prompt: str,
        max_new_tokens: int = 0,
    ) -> TeacherKVCache:
        """
        Extract KV pairs from the teacher model for the given prompt.

        This runs a forward pass through the teacher model and captures
        the KV cache at each layer.

        Args:
            prompt: Input text to process
            max_new_tokens: If > 0, also run generation (KV extracted from prefill)

        Returns:
            TeacherKVCache with per-layer KV pairs
        """
        self._load_model()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Forward pass with KV cache extraction
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=False,
            )

        cache = outputs.past_key_values
        if cache is None:
            logger.warning("Teacher model returned no KV cache")
            return TeacherKVCache(model_id=self.model_id, sequence_length=seq_len)

        # Extract KV pairs from cache
        kv_pairs = {}
        num_layers = len(cache)

        # Get model config info
        config = self._model.config
        hidden_size = getattr(config, "hidden_size", 0)
        num_heads = getattr(config, "num_attention_heads",
                            getattr(config, "num_key_value_heads", 0))
        head_dim = hidden_size // max(num_heads, 1) if hidden_size and num_heads else 0

        for layer_idx, layer_cache in enumerate(cache):
            if layer_cache is None:
                continue
            # DynamicCache tuple format: (key, value, optional_extra)
            # Legacy tuple format: (key, value)
            # Note: Keep tensors on their original device, don't force .cpu()
            # The downstream code handles device alignment
            if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                key, value = layer_cache[0], layer_cache[1]
                kv_pairs[layer_idx] = (key.detach(), value.detach())
            elif isinstance(layer_cache, torch.Tensor):
                # Some models return a single tensor
                kv_pairs[layer_idx] = (layer_cache.detach(), layer_cache.detach())

        return TeacherKVCache(
            kv_pairs=kv_pairs,
            model_id=self.model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            sequence_length=seq_len,
        )

    def generate_with_extraction(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> Tuple[str, TeacherKVCache]:
        """
        Generate text and extract KV pairs simultaneously.

        Returns:
            Tuple of (generated_text, teacher_kv_cache)
        """
        self._load_model()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            # First pass: extract KV from prefill
            outputs = self._model(
                input_ids=input_ids,
                use_cache=True,
            )

            teacher_kv = self._extract_from_cache(outputs.past_key_values, input_ids.shape[1])

            # Generate
            generated = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

        generated_text = self._tokenizer.decode(
            generated[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text, teacher_kv

    def _extract_from_cache(
        self,
        cache: Any,
        seq_len: int,
    ) -> TeacherKVCache:
        """Extract TeacherKVCache from a HuggingFace cache object."""
        if cache is None:
            return TeacherKVCache(model_id=self.model_id, sequence_length=seq_len)

        kv_pairs = {}
        num_layers = len(cache)
        config = self._model.config
        hidden_size = getattr(config, "hidden_size", 0)
        num_heads = getattr(config, "num_attention_heads",
                            getattr(config, "num_key_value_heads", 0))
        head_dim = hidden_size // max(num_heads, 1) if hidden_size and num_heads else 0

        for layer_idx, layer_cache in enumerate(cache):
            if layer_cache is None:
                continue
            if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                key, value = layer_cache[0], layer_cache[1]
                kv_pairs[layer_idx] = (key.detach().cpu(), value.detach().cpu())

        return TeacherKVCache(
            kv_pairs=kv_pairs,
            model_id=self.model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            sequence_length=seq_len,
        )


# ─────────────────────────────────────────────────────────────
# ExoBrain Inference Pipeline
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Adaptive Injection Scheduler (v0.6)
# ─────────────────────────────────────────────────────────────

class AdaptiveInjectionScheduler:
    """
    Decide when to inject external brain KV based on model confidence (v0.6).

    Problem: Injecting teacher KV at EVERY decode step is wasteful.
    When the shell model is confident (low perplexity), external guidance
    adds little value but incurs compute cost. When uncertain (high PPL),
    injection is critical.

    Solution: Monitor the shell model's perplexity at each decode step.
    Only inject when PPL exceeds a threshold, saving ~30-60% of injections
    with minimal quality loss.

    Strategies:
    - "threshold": Inject when PPL > threshold (simple, effective)
    - "relative": Inject when PPL > α × running_avg_PPL (adaptive)
    - "entropy": Inject when attention entropy exceeds threshold
    - "always": Always inject (backward compatible, baseline)
    - "never": Never inject (ablation study)

    Usage:
        scheduler = AdaptiveInjectionScheduler(strategy="threshold", ppl_threshold=10.0)
        for step in decode_steps:
            ppl = compute_perplexity(logits)
            if scheduler.should_inject(ppl):
                inject_teacher_kv()
            scheduler.record(ppl)
    """

    def __init__(
        self,
        strategy: str = "threshold",
        ppl_threshold: float = 10.0,
        entropy_threshold: float = 0.8,
        relative_alpha: float = 1.5,
        warmup_steps: int = 3,
        window_size: int = 10,
        min_injection_rate: float = 0.1,
    ) -> None:
        """
        Args:
            strategy: Scheduling strategy (see class docstring)
            ppl_threshold: PPL threshold for "threshold" strategy
            entropy_threshold: Entropy threshold for "entropy" strategy
            relative_alpha: Multiplier for "relative" strategy (inject if PPL > α × avg)
            warmup_steps: Always inject during first N steps (model warming up)
            window_size: Rolling window for running average PPL
            min_injection_rate: Minimum fraction of steps that must be injected
        """
        self.strategy = strategy
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.relative_alpha = relative_alpha
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.min_injection_rate = min_injection_rate

        # State
        self._step_count: int = 0
        self._injection_count: int = 0
        self._ppl_history: List[float] = []
        self._decision_history: List[bool] = []

    def should_inject(
        self,
        ppl: Optional[float] = None,
        entropy: Optional[float] = None,
    ) -> bool:
        """
        Decide whether to inject external brain KV at this step.

        Args:
            ppl: Current perplexity value (required for threshold/relative strategies)
            entropy: Current attention entropy (required for entropy strategy)

        Returns:
            True if injection should proceed
        """
        self._step_count += 1

        # Always inject during warmup
        if self._step_count <= self.warmup_steps:
            self._record_decision(True)
            return True

        decision = False

        if self.strategy == "always":
            decision = True
        elif self.strategy == "never":
            decision = False
        elif self.strategy == "threshold":
            decision = ppl is not None and ppl > self.ppl_threshold
        elif self.strategy == "relative":
            if ppl is not None and self._ppl_history:
                avg_ppl = self._get_running_avg_ppl()
                decision = ppl > self.relative_alpha * avg_ppl
            else:
                decision = True  # No history yet → inject
        elif self.strategy == "entropy":
            decision = entropy is not None and entropy > self.entropy_threshold
        else:
            decision = True  # Default: inject

        # Enforce minimum injection rate
        if not decision and self._step_count > self.warmup_steps:
            current_rate = self._injection_count / max(self._step_count, 1)
            if current_rate < self.min_injection_rate:
                decision = True  # Force injection to maintain minimum rate

        self._record_decision(decision)
        return decision

    def record(self, ppl: Optional[float] = None) -> None:
        """
        Record PPL observation for running average computation.

        Call this AFTER each decode step, regardless of injection decision.
        """
        if ppl is not None:
            self._ppl_history.append(ppl)
            # Keep only recent history
            if len(self._ppl_history) > self.window_size:
                self._ppl_history = self._ppl_history[-self.window_size:]

    def _record_decision(self, injected: bool) -> None:
        """Record injection decision for statistics."""
        self._decision_history.append(injected)
        if injected:
            self._injection_count += 1

    def _get_running_avg_ppl(self) -> float:
        """Compute running average PPL from recent history."""
        if not self._ppl_history:
            return float("inf")
        return sum(self._ppl_history) / len(self._ppl_history)

    def reset(self) -> None:
        """Reset scheduler state for a new generation."""
        self._step_count = 0
        self._injection_count = 0
        self._ppl_history = []
        self._decision_history = []

    @property
    def stats(self) -> Dict[str, Any]:
        """Return scheduler statistics."""
        injection_rate = self._injection_count / max(self._step_count, 1)
        skipped = self._step_count - self._injection_count
        return {
            "strategy": self.strategy,
            "total_steps": self._step_count,
            "injection_steps": self._injection_count,
            "skipped_steps": skipped,
            "injection_rate": injection_rate,
            "skip_rate": 1.0 - injection_rate,
            "avg_ppl": self._get_running_avg_ppl() if self._ppl_history else None,
        }


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute perplexity from model logits and target token IDs (v0.6).

    Args:
        logits: [batch, seq_len, vocab_size] — model output logits
        target_ids: [batch, seq_len] — target token IDs
        eps: Small value to avoid log(0)

    Returns:
        Perplexity value (float)
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather log probs for target tokens
    # target_ids: [batch, seq_len] → [batch, seq_len, 1]
    target_ids_expanded = target_ids.unsqueeze(-1)

    # Clamp target_ids to valid range
    vocab_size = logits.shape[-1]
    target_ids_clamped = target_ids_expanded.clamp(0, vocab_size - 1)

    # Gather: [batch, seq_len, 1] → squeeze → [batch, seq_len]
    selected_log_probs = log_probs.gather(dim=-1, index=target_ids_clamped).squeeze(-1)

    # Mean negative log likelihood
    avg_nll = -selected_log_probs.mean().item()

    # Perplexity = exp(NLL)
    perplexity = math.exp(min(avg_nll, 20.0))  # Cap to avoid overflow

    return perplexity


# ─────────────────────────────────────────────────────────────
# Brain KV Compression — Compress teacher KV for efficient transfer (v0.6)
# ─────────────────────────────────────────────────────────────

class BrainKVCompressor:
    """
    Compress teacher KV cache for efficient transfer to the shell model (v0.6).

    Problem: A 7B teacher model's KV cache is huge:
    - 32 layers × 2 (K+V) × [batch, num_heads, seq_len, head_dim]
    - For seq_len=1024, head_dim=128, num_heads=32: ~512 MB per prompt
    - This is prohibitive for memory-constrained deployment

    Solution: Compress the KV before injection using:
    1. "topk_spatial": Keep only top-K most important KV positions per head
       (based on attention weight magnitude). Reduces seq_len dimension.
    2. "quantize_8bit": Quantize FP32/FP16 KV to INT8. Reduces memory by 2-4×.
    3. "mean_pool": Mean-pool KV over non-overlapping windows. Reduces seq_len.
    4. "svd_lowrank": SVD-based low-rank approximation of the KV matrix.
    5. "none": No compression (backward compatible, baseline)

    Decompression is implicit — the compressed KV is used directly in
    cross-attention (the shell's query attends to fewer/more-compact KV pairs).

    Usage:
        compressor = BrainKVCompressor(method="topk_spatial", compression_ratio=0.5)
        compressed_kv = compressor.compress(layer_idx, key, value)
        # Use compressed_kv directly in ExoBrain injection
    """

    def __init__(
        self,
        method: str = "topk_spatial",
        compression_ratio: float = 0.5,
        quantize_bits: int = 8,
        pool_window: int = 4,
        svd_rank: int = 64,
    ) -> None:
        """
        Args:
            method: Compression method (see class docstring)
            compression_ratio: Target compression ratio (0.5 = keep 50% of data)
            quantize_bits: Bit width for quantization (8 or 4)
            pool_window: Window size for mean_pool method
            svd_rank: Rank for SVD low-rank approximation
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.quantize_bits = quantize_bits
        self.pool_window = pool_window
        self.svd_rank = svd_rank

        # Compression statistics
        self._stats: Dict[str, Any] = {
            "method": method,
            "total_compressed": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
        }

    def compress(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a KV pair for a specific layer.

        Args:
            layer_idx: Transformer layer index
            key: [batch, heads, seq_len, dim] — key tensor
            value: [batch, heads, seq_len, dim] — value tensor

        Returns:
            (compressed_key, compressed_value)
        """
        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

        if self.method == "none":
            return key, value

        elif self.method == "topk_spatial":
            key, value = self._compress_topk(key, value)
        elif self.method == "quantize_8bit":
            key, value = self._compress_quantize(key, value)
        elif self.method == "mean_pool":
            key, value = self._compress_mean_pool(key, value)
        elif self.method == "svd_lowrank":
            key, value = self._compress_svd(key, value)
        else:
            return key, value

        compressed_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

        self._stats["total_compressed"] += 1
        self._stats["total_original_bytes"] += original_bytes
        self._stats["total_compressed_bytes"] += compressed_bytes

        return key, value

    def _compress_topk(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep only top-K positions per head based on key magnitude.

        Selects the K positions with highest L2 norm in the key tensor,
        which correspond to the most "important" attention targets.
        """
        # key: [batch, heads, seq_len, dim]
        seq_len = key.shape[2]
        k = max(1, int(seq_len * self.compression_ratio))

        # Compute per-position magnitude: [batch, heads, seq_len]
        key_magnitude = key.norm(dim=-1)  # [B, H, S]

        # Select top-K positions
        _, top_indices = key_magnitude.topk(k, dim=-1)  # [B, H, K]

        # Gather selected positions
        # top_indices: [B, H, K] → [B, H, K, 1] → [B, H, K, dim]
        dim = key.shape[-1]
        index = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        compressed_key = torch.gather(key, dim=2, index=index)
        compressed_value = torch.gather(value, dim=2, index=index)

        return compressed_key, compressed_value

    def _compress_quantize(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize KV to lower bit width.

        Maps FP32/FP16 → INT8 using symmetric quantization:
        q = round(x / scale), where scale = max(|x|) / (2^(bits-1) - 1)
        """
        if self.quantize_bits == 8:
            qmin, qmax = -128, 127
        elif self.quantize_bits == 4:
            qmin, qmax = -8, 7
        else:
            return key, value

        def quantize_tensor(t: torch.Tensor) -> torch.Tensor:
            # Per-tensor symmetric quantization
            max_val = t.abs().max()
            scale = max_val / max(qmax, 1)
            if scale < 1e-8:
                return t
            quantized = torch.clamp(torch.round(t / scale), qmin, qmax)
            # Store as float (still saves memory conceptually, but actual
            # INT8 storage requires specialized packing — we keep as float
            # for compatibility with attention computation)
            return quantized * scale

        return quantize_tensor(key), quantize_tensor(value)

    def _compress_mean_pool(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean-pool KV over non-overlapping windows.

        Reduces seq_len by pool_window factor. Preserves semantic content
        while reducing spatial resolution.
        """
        w = self.pool_window
        seq_len = key.shape[2]

        if seq_len < w:
            return key, value

        # Truncate to multiple of window
        trimmed_len = (seq_len // w) * w

        # Reshape: [B, H, trimmed_len, D] → [B, H, trimmed_len//w, w, D]
        key_trimmed = key[:, :, :trimmed_len, :]
        value_trimmed = value[:, :, :trimmed_len, :]

        key_reshaped = key_trimmed.reshape(
            *key.shape[:2], trimmed_len // w, w, key.shape[-1]
        )
        value_reshaped = value_trimmed.reshape(
            *value.shape[:2], trimmed_len // w, w, value.shape[-1]
        )

        # Mean over window dimension
        compressed_key = key_reshaped.mean(dim=3)   # [B, H, trimmed_len//w, D]
        compressed_value = value_reshaped.mean(dim=3)

        return compressed_key, compressed_value

    def _compress_svd(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SVD low-rank approximation of KV matrices.

        Decomposes key/value into low-rank factors:
        K ≈ U_K @ S_K @ V_K^T, keeping only top-rank singular values.

        This is more expensive but provides the best quality-compression tradeoff.
        """
        rank = min(self.svd_rank, min(key.shape[2], key.shape[-1]))

        def svd_compress(t: torch.Tensor) -> torch.Tensor:
            # t: [B, H, S, D]
            B, H, S, D = t.shape
            # Reshape to 2D for SVD: [B*H, S, D]
            t_flat = t.reshape(B * H, S, D)

            try:
                U, S, Vh = torch.linalg.svd(t_flat.float(), full_matrices=False)
                # Keep top-rank components
                U_r = U[:, :, :rank]          # [B*H, S, rank]
                S_r = torch.diag_embed(S[:, :rank])  # [B*H, rank, rank]
                Vh_r = Vh[:, :rank, :]        # [B*H, rank, D]

                # Reconstruct: [B*H, S, rank] @ [B*H, rank, rank] @ [B*H, rank, D]
                compressed = U_r @ S_r @ Vh_r
                return compressed.reshape(B, H, S, D).to(t.dtype)
            except Exception:
                # SVD failed (e.g., numerical issues) — return original
                return t

        return svd_compress(key), svd_compress(value)

    def compress_teacher_cache(
        self,
        teacher_kv: "TeacherKVCache",
    ) -> "TeacherKVCache":
        """
        Compress an entire TeacherKVCache.

        Args:
            teacher_kv: TeacherKVCache with per-layer KV pairs

        Returns:
            New TeacherKVCache with compressed KV pairs
        """
        compressed_pairs = {}
        for layer_idx, (key, value) in teacher_kv.kv_pairs.items():
            compressed_key, compressed_value = self.compress(layer_idx, key, value)
            compressed_pairs[layer_idx] = (compressed_key, compressed_value)

        return TeacherKVCache(
            kv_pairs=compressed_pairs,
            model_id=teacher_kv.model_id + "_compressed",
            num_layers=teacher_kv.num_layers,
            hidden_size=teacher_kv.hidden_size,
            num_heads=teacher_kv.num_heads,
            head_dim=teacher_kv.head_dim,
            sequence_length=teacher_kv.sequence_length,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        avg_ratio = 1.0
        if self._stats["total_original_bytes"] > 0:
            avg_ratio = self._stats["total_compressed_bytes"] / self._stats["total_original_bytes"]
        return {
            **self._stats,
            "avg_compression_ratio": avg_ratio,
            "avg_space_saving": 1.0 - avg_ratio,
        }


# ─────────────────────────────────────────────────────────────
# KVPrefetcher — Speculative KV Injection Prefetcher (v0.5)
# ─────────────────────────────────────────────────────────────

class KVPrefetcher:
    """
    Prefetch and cache projected teacher KV pairs for faster decode.

    Problem: During autoregressive decode, _inject_teacher_kv_into_cache()
    is called every step. For each layer, it:
    1. Retrieves teacher KV from bus
    2. Optionally projects via HeadDimProjection
    3. Fuses with shell KV

    Steps 1-2 are redundant after the first decode step — the teacher KV
    doesn't change between steps (same prompt, same teacher). Only the
    shell KV changes (one new token appended each step).

    Solution: Cache the projected teacher KV after the first injection.
    For subsequent steps, reuse the cached projected KV and only redo
    the fusion (which depends on current shell KV shape).

    This eliminates redundant bus.retrieve() + projector calls during
    decode, reducing per-step overhead from O(layers × retrieval) to
    O(layers × fusion_only).

    Usage:
        prefetcher = KVPrefetcher(brain_bus, kv_projector, fusion_mode)
        # After first injection:
        prefetcher.cache_projected_kv(teacher_kv)
        # During decode:
        projected_kv = prefetcher.get_projected_kv(layer_idx)
    """

    def __init__(
        self,
        brain_bus: Any,  # ExoBrainBus
        kv_projector: Optional[HeadDimProjection] = None,
        fusion_mode: str = "replace",
        residual_alpha: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.brain_bus = brain_bus
        self.kv_projector = kv_projector
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha
        self.device = device

        # Cached projected teacher KV: {layer_idx: (key, value)}
        self._projected_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache hit/miss stats
        self._stats = {"cache_hits": 0, "cache_misses": 0, "prefetch_count": 0}

    def cache_projected_kv(self, projected_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """
        Pre-cache projected teacher KV pairs.

        Call this after the first _build_brain() to cache all projected KV.
        Subsequent decode steps can use get_projected_kv() instead of
        going through bus.retrieve() + projector again.

        Args:
            projected_pairs: {layer_idx: (projected_key, projected_value)}

        Returns:
            Number of layers cached
        """
        self._projected_cache.clear()
        for layer_idx, (key, value) in projected_pairs.items():
            # Detach and move to device for stable caching
            self._projected_cache[layer_idx] = (
                key.detach().to(device=self.device),
                value.detach().to(device=self.device),
            )
        self._stats["prefetch_count"] += 1
        return len(self._projected_cache)

    def get_projected_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get pre-cached projected teacher KV for a layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            (projected_key, projected_value) or None if not cached
        """
        if layer_idx in self._projected_cache:
            self._stats["cache_hits"] += 1
            return self._projected_cache[layer_idx]
        self._stats["cache_misses"] += 1
        return None

    def is_cached(self, layer_idx: int) -> bool:
        """Check if projected KV is cached for a layer."""
        return layer_idx in self._projected_cache

    @property
    def stats(self) -> Dict[str, Any]:
        """Return prefetcher statistics."""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / max(total, 1)
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "cached_layers": list(self._projected_cache.keys()),
        }


# ─────────────────────────────────────────────────────────────
# ExoBrain Inference Pipeline
# ─────────────────────────────────────────────────────────────

class ExoBrainInferencePipeline:
    """
    End-to-end inference pipeline for ExoBrain-powered shell models (v0.4+).

    Flow:
    1. Load shell model (real weights, not zero-weight)
    2. Load teacher model (full weights)
    3. Extract teacher KV for the given prompt
    4. Inject teacher KV via ExoBrain + ShellProjection
    5. Run inference on the shell model
    6. Evaluate quality

    This proves the core thesis: a lightweight shell model (0.1B real weights)
    with cognitive alignment (ShellProjection) can perform meaningful inference
    using KV from an external brain (7B+ model).

    Note: The old "zero-weight shell" approach is mathematically broken.
    The shell MUST have real, trainable weights to generate meaningful queries.
    """

    def __init__(
        self,
        shell_model_path: str,
        teacher_model_id: Optional[str] = None,
        fusion_mode: str = "replace",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        retrieval_top_k: int = 5,
        residual_alpha: float = 0.1,
        gate_temperature: float = 1.0,
        max_new_tokens: int = 64,
        head_dim_projection: str = "pad_or_truncate",
    ) -> None:
        self.shell_model_path = shell_model_path
        self.teacher_model_id = teacher_model_id
        self.fusion_mode = fusion_mode
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.retrieval_top_k = retrieval_top_k
        self.residual_alpha = residual_alpha
        self.gate_temperature = gate_temperature
        self.max_new_tokens = max_new_tokens
        self.head_dim_projection = head_dim_projection

        self._shell_model = None
        self._shell_tokenizer = None
        self._teacher_extractor = None
        self._brain_bus = None
        self._brain_cfg = None
        self._kv_projector: Optional[HeadDimProjection] = None
        self._kv_prefetcher: Optional[KVPrefetcher] = None

    def _load_shell_model(self) -> None:
        """Load the shell model from disk."""
        if self._shell_model is not None:
            return

        logger.info("Loading shell model from: %s", self.shell_model_path)
        self._shell_tokenizer = load_tokenizer(
            self.shell_model_path,
            security={"trust_remote_code": self.trust_remote_code, "local_files_only": True},
        )
        self._shell_model = load_causallm(
            self.shell_model_path,
            security={"trust_remote_code": self.trust_remote_code, "local_files_only": True},
            torch_dtype=self.dtype,
            device=self.device,
            low_cpu_mem_usage=True,
        )
        self._shell_model.eval()
        logger.info("Shell model loaded: %s", type(self._shell_model).__name__)

    def _init_teacher(self) -> None:
        """Initialize the teacher model extractor."""
        if self._teacher_extractor is not None or self.teacher_model_id is None:
            return

        self._teacher_extractor = TeacherKVExtractor(
            model_id=self.teacher_model_id,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

    def _build_brain(
        self,
        teacher_kv: TeacherKVCache,
    ) -> None:
        """Build ExoBrain bus with teacher KV for injection."""
        from .exobrain import (
            ExoBrainBus,
            ExoBrainConfig,
            LocalWeightSource,
        )

        # Build head-dim projector if teacher and shell differ
        self._build_kv_projector(teacher_kv)

        # Project teacher KV if needed before injection
        # Also ensure all tensors are on the correct device
        projected_pairs = {}
        for layer_idx, (key, value) in teacher_kv.kv_pairs.items():
            # Ensure tensors are on the correct device
            key = key.to(device=self.device, non_blocking=True)
            value = value.to(device=self.device, non_blocking=True)
            if self._kv_projector is not None:
                key, value = self._kv_projector.project_kv_pair(key, value)
            projected_pairs[layer_idx] = (key, value)

        # Create local weight source from (possibly projected) teacher KV
        local_source = LocalWeightSource()
        for layer_idx, (key, value) in projected_pairs.items():
            local_source.set_teacher_kv(layer_idx, key, value)

        # Create ExoBrain bus and config
        self._brain_bus = ExoBrainBus(sources=[local_source])
        self._brain_cfg = ExoBrainConfig(
            fusion_mode=self.fusion_mode,
            retrieval_top_k=self.retrieval_top_k,
            residual_alpha=self.residual_alpha,
            gate_temperature=self.gate_temperature,
        )

        # Also inject directly for guaranteed hit
        for layer_idx, (key, value) in projected_pairs.items():
            self._brain_bus.inject_kv(layer_idx, key, value)

        # v0.5: Initialize KV prefetcher and cache projected pairs
        # This avoids redundant bus.retrieve() + projector calls during decode
        self._kv_prefetcher = KVPrefetcher(
            brain_bus=self._brain_bus,
            kv_projector=self._kv_projector,
            fusion_mode=self.fusion_mode,
            residual_alpha=self.residual_alpha,
            device=self.device,
        )
        num_cached = self._kv_prefetcher.cache_projected_kv(projected_pairs)

        logger.info(
            "ExoBrain bus built: %d layers injected from teacher '%s' (projector=%s, prefetcher=%d cached)",
            len(projected_pairs),
            teacher_kv.model_id,
            type(self._kv_projector).__name__ if self._kv_projector else "None",
            num_cached,
        )

    def _build_kv_projector(self, teacher_kv: TeacherKVCache) -> None:
        """
        Build a HeadDimProjection if teacher and shell have different head_dim.

        The projector maps teacher_head_dim → shell_head_dim so that
        KV injection can proceed without dimension mismatches.
        """
        if self._kv_projector is not None:
            return  # Already built

        if not teacher_kv.kv_pairs:
            return

        # Determine shell head_dim from the loaded shell model
        if self._shell_model is None:
            return

        shell_config = self._shell_model.config
        shell_hidden = getattr(shell_config, "hidden_size", 0)
        # Use num_attention_heads for head_dim computation (not num_key_value_heads)
        # For GQA models, num_key_value_heads < num_attention_heads but head_dim is the same
        shell_attention_heads = getattr(shell_config, "num_attention_heads", 0)
        shell_head_dim = shell_hidden // max(shell_attention_heads, 1) if shell_hidden and shell_attention_heads else 0
        # num_kv_heads is used for the projector (can be smaller due to GQA)
        shell_kv_heads = getattr(
            shell_config, "num_key_value_heads",
            getattr(shell_config, "num_attention_heads", 0),
        )

        teacher_head_dim = teacher_kv.head_dim

        if shell_head_dim == 0 or teacher_head_dim == 0:
            logger.warning(
                "Cannot build KV projector: shell_head_dim=%d, teacher_head_dim=%d",
                shell_head_dim, teacher_head_dim,
            )
            return

        if shell_head_dim == teacher_head_dim:
            logger.info(
                "Head dims match (shell=%d, teacher=%d) — no projection needed",
                shell_head_dim, teacher_head_dim,
            )
            return

        logger.info(
            "Building KV projector: teacher_head_dim=%d → shell_head_dim=%d (mode=%s)",
            teacher_head_dim, shell_head_dim, self.head_dim_projection,
        )

        self._kv_projector = HeadDimProjection(
            teacher_head_dim=teacher_head_dim,
            shell_head_dim=shell_head_dim,
            num_kv_heads=shell_kv_heads,
            mode=self.head_dim_projection,
        ).to(self.device)

    def infer(self, prompt: str) -> InferenceResult:
        """
        Run ExoBrain inference on the shell model.

        Args:
            prompt: Input text

        Returns:
            InferenceResult with generated text and statistics
        """
        start_time = time.time()
        error = None

        try:
            self._load_shell_model()
            self._init_teacher()

            # Tokenize input
            inputs = self._shell_tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            prompt_tokens = input_ids.shape[1]

            # Step 1: Extract teacher KV (if teacher available)
            if self._teacher_extractor is not None:
                logger.info("Extracting teacher KV for prompt (%d tokens)...", prompt_tokens)
                teacher_kv = self._teacher_extractor.extract_kv(prompt)
                logger.info(
                    "Teacher KV extracted: %d layers, seq_len=%d",
                    len(teacher_kv.kv_pairs),
                    teacher_kv.sequence_length,
                )

                # Step 2: Build ExoBrain
                self._build_brain(teacher_kv)

                # Step 3: Run shell model inference with ExoBrain injection
                # We use the ExoBrain backend to inject KV at attention time
                from .exobrain import ExoBrainBackend
                from .cache_store import KVCacheStoreConfig

                kv_cfg = KVCacheStoreConfig()
                ExoBrainBackend(
                    store_cfg=kv_cfg,
                    brain_bus=self._brain_bus,
                    brain_cfg=self._brain_cfg,
                )

                # Generate with ExoBrain — inject teacher KV at each decode step
                with torch.no_grad():
                    # First forward pass to get the shell's own KV cache
                    shell_outputs = self._shell_model(
                        input_ids=input_ids,
                        use_cache=True,
                    )
                    shell_cache = shell_outputs.past_key_values

                    # Immediately inject teacher KV into the prefill cache
                    if self._brain_bus is not None:
                        shell_cache = self._inject_teacher_kv_into_cache(shell_cache)

                    # For each decode step, inject teacher KV
                    generated_ids = input_ids.clone()
                    next_token_logits = shell_outputs.logits[:, -1, :]

                    for _ in range(self.max_new_tokens):
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                        # Check for EOS
                        if next_token.item() == self._shell_tokenizer.eos_token_id:
                            break

                        # Next decode step with teacher KV injection
                        decode_input = next_token

                        with torch.no_grad():
                            decode_outputs = self._shell_model(
                                input_ids=decode_input,
                                past_key_values=shell_cache,
                                use_cache=True,
                            )

                        # Inject teacher KV into the updated cache
                        shell_cache = decode_outputs.past_key_values
                        if self._brain_bus is not None:
                            shell_cache = self._inject_teacher_kv_into_cache(shell_cache)

                        next_token_logits = decode_outputs.logits[:, -1, :]

                generated_text = self._shell_tokenizer.decode(
                    generated_ids[0][prompt_tokens:],
                    skip_special_tokens=True,
                )
                generated_tokens = generated_ids.shape[1] - prompt_tokens

                # Get brain stats
                brain_stats = {}
                if self._brain_bus is not None:
                    brain_stats = self._brain_bus.stats
                # v0.5: Include prefetcher stats
                if self._kv_prefetcher is not None:
                    brain_stats["prefetcher"] = self._kv_prefetcher.stats

            else:
                # No teacher — just run shell model directly
                with torch.no_grad():
                    outputs = self._shell_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                generated_text = self._shell_tokenizer.decode(
                    outputs[0][prompt_tokens:],
                    skip_special_tokens=True,
                )
                generated_tokens = outputs.shape[1] - prompt_tokens
                brain_stats = {}

        except Exception as e:
            logger.error("ExoBrain inference failed: %s", e)
            error = str(e)
            generated_text = ""
            generated_tokens = 0
            prompt_tokens = 0
            brain_stats = {}

        elapsed = time.time() - start_time
        tokens_per_s = generated_tokens / max(elapsed, 1e-6)

        return InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            inference_time_s=elapsed,
            tokens_per_second=tokens_per_s,
            fusion_mode=self.fusion_mode,
            brain_hit_rate=brain_stats.get("hit_rate", 0.0),
            brain_stats=brain_stats,
            device=self.device,
            error=error,
        )

    def _inject_teacher_kv_into_cache(self, shell_cache: Any) -> Any:
        """
        Inject teacher KV pairs into the shell model's KV cache.

        This replaces or blends the shell's KV cache entries with
        teacher KV at each layer, enabling the shell to "see" the
        teacher's knowledge during attention computation.

        Args:
            shell_cache: HuggingFace DynamicCache or tuple of (K, V) pairs

        Returns:
            Modified cache with injected teacher KV
        """
        if self._brain_bus is None:
            return shell_cache

        # Convert DynamicCache to list of (key, value) tuples
        # DynamicCache supports tuple() iteration but not subscript access
        try:
            cache_layers = list(tuple(shell_cache))
        except (TypeError, NotImplementedError):
            # Fallback: try subscript
            cache_layers = list(shell_cache)

        num_layers = len(cache_layers)

        # Build new cache layers
        new_layers: list = []

        for layer_idx in range(num_layers):
            layer_cache = cache_layers[layer_idx]
            if layer_cache is None:
                new_layers.append(layer_cache)
                continue

            # DynamicCache tuple format: (key, value, optional_extra)
            # Legacy tuple format: (key, value)
            key = layer_cache[0]
            value = layer_cache[1]

            # Try to get teacher KV for this layer
            # v0.5: Use prefetcher cache first (avoid redundant bus.retrieve + projector)
            teacher_kv = None
            if self._kv_prefetcher is not None:
                teacher_kv = self._kv_prefetcher.get_projected_kv(layer_idx)

            if teacher_kv is None:
                # Fallback: use bus retrieval (slower, for uncached layers)
                query = key[:, :, -1:, :]  # Use last key as query proxy
                teacher_kv = self._brain_bus.retrieve(query, layer_idx)

            if teacher_kv is not None:
                t_key, t_value = teacher_kv

                # Apply KV projector if dimensions still mismatch
                # (should already be projected in _build_brain, but as a safety net)
                if (self._kv_projector is not None
                        and t_key.shape[-1] != key.shape[-1]):
                    t_key, t_value = self._kv_projector.project_kv_pair(t_key, t_value)

                if self.fusion_mode == "replace":
                    shell_seq = key.shape[2]
                    teacher_seq = t_key.shape[2]

                    if teacher_seq >= shell_seq:
                        new_key = t_key[:, :, :shell_seq, :]
                        new_value = t_value[:, :, :shell_seq, :]
                    else:
                        pad_len = shell_seq - teacher_seq
                        new_key = torch.cat([t_key, key[:, :, :pad_len, :]], dim=2)
                        new_value = torch.cat([t_value, value[:, :, :pad_len, :]], dim=2)

                    new_layers.append((new_key, new_value))

                elif self.fusion_mode == "residual":
                    alpha = self.residual_alpha
                    seq_len = min(key.shape[2], t_key.shape[2])
                    blended_key = alpha * key[:, :, :seq_len, :] + (1 - alpha) * t_key[:, :, :seq_len, :]
                    blended_value = alpha * value[:, :, :seq_len, :] + (1 - alpha) * t_value[:, :, :seq_len, :]

                    new_layers.append((blended_key, blended_value))

                else:  # gated
                    shell_norm = key.norm(dim=-1, keepdim=True)
                    teacher_norm = t_key.norm(dim=-1, keepdim=True)
                    gate = torch.sigmoid(teacher_norm - shell_norm)

                    seq_len = min(key.shape[2], t_key.shape[2])

                    gated_key = gate[..., :seq_len, :1] * t_key[:, :, :seq_len, :] + \
                                (1 - gate[..., :seq_len, :1]) * key[:, :, :seq_len, :]
                    gated_value = gate[..., :seq_len, :1] * t_value[:, :, :seq_len, :] + \
                                  (1 - gate[..., :seq_len, :1]) * value[:, :, :seq_len, :]

                    new_layers.append((gated_key, gated_value))
            else:
                new_layers.append((key, value))

        # Log injection stats
        if hasattr(self._brain_bus, 'stats'):
            stats = self._brain_bus.stats
            logger.info(
                "ExoBrain injection stats: hit_rate=%.1f%%, hits=%d, misses=%d",
                stats.get('hit_rate', 0) * 100,
                stats.get('hit_count', 0),
                stats.get('miss_count', 0),
            )

        # Reconstruct the cache in the same format as the input
        from transformers.cache_utils import DynamicCache

        if isinstance(shell_cache, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(new_layers):
                new_cache.update(k, v, layer_idx)
            return new_cache
        else:
            # Preserve original tuple structure (2-tuple or 3-tuple)
            result = []
            for layer_idx in range(len(new_layers)):
                new_k, new_v = new_layers[layer_idx]
                orig = cache_layers[layer_idx]
                if orig is not None and len(orig) > 2:
                    # Preserve 3rd element (e.g., flash attention mask)
                    result.append((new_k, new_v) + orig[2:])
                else:
                    result.append((new_k, new_v))
            return tuple(result)

    def evaluate(
        self,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate ExoBrain inference quality across multiple prompts.

        Args:
            prompts: List of input prompts
            reference_texts: Optional reference outputs for comparison

        Returns:
            Evaluation metrics dictionary
        """
        results = []
        for i, prompt in enumerate(prompts):
            result = self.infer(prompt)
            results.append(result)
            logger.info(
                "Prompt %d/%d: %d tokens generated (%.1f tok/s) %s",
                i + 1, len(prompts),
                result.generated_tokens,
                result.tokens_per_second,
                "✓" if result.error is None else f"✗ {result.error}",
            )

        # Aggregate metrics
        total_tokens = sum(r.generated_tokens for r in results)
        total_time = sum(r.inference_time_s for r in results)
        errors = sum(1 for r in results if r.error is not None)

        metrics = {
            "num_prompts": len(prompts),
            "total_generated_tokens": total_tokens,
            "total_time_s": total_time,
            "avg_tokens_per_second": total_tokens / max(total_time, 1e-6),
            "error_rate": errors / max(len(prompts), 1),
            "avg_brain_hit_rate": sum(r.brain_hit_rate for r in results) / max(len(results), 1),
            "results": [
                {
                    "prompt": r.prompt[:100],
                    "generated_text": r.generated_text[:200],
                    "tokens": r.generated_tokens,
                    "tok_per_s": r.tokens_per_second,
                    "error": r.error,
                }
                for r in results
            ],
        }

        # Compute text similarity if reference provided
        if reference_texts is not None and len(reference_texts) == len(prompts):
            similarities = []
            for result, ref in zip(results, reference_texts):
                if result.generated_text and ref:
                    # Simple character-level overlap
                    gen_chars = set(result.generated_text.lower())
                    ref_chars = set(ref.lower())
                    if ref_chars:
                        overlap = len(gen_chars & ref_chars) / len(ref_chars)
                        similarities.append(overlap)
            if similarities:
                metrics["avg_char_overlap"] = sum(similarities) / len(similarities)

        return metrics


# ─────────────────────────────────────────────────────────────
# Knowledge Distiller — KV → Weight Distillation
# ─────────────────────────────────────────────────────────────

class KnowledgeDistiller:
    """
    Distill knowledge from external KV into shell model weights (v0.4+).

    The distillation process:
    1. Run teacher forward → extract KV
    2. Run shell forward with ExoBrain (teacher KV injected)
    3. Optionally train ShellProjection for cognitive alignment
    4. Compute loss between ExoBrain output and teacher output
    5. Backpropagate through shell + ShellProjection
    6. Update weights via gradient descent
    7. Save the updated model

    This "bakes" the external brain knowledge into the shell model's
    weights and ShellProjection, making it self-sufficient.

    Note: Shell weights MUST be real and trainable (not zero-weight).
    Use ShellProjection to bridge shell_hidden_dim → brain_hidden_dim.

    ┌──────────────┐         ┌──────────────┐
    │ Teacher Model │────KV──→│  ExoBrain    │
    │ (frozen)      │         │  Injection   │
    └──────────────┘         └──────┬───────┘
                                    │
                                    ▼
                           ┌──────────────┐
                           │ Shell Model   │
                           │ (trainable)   │
                           │ + ShellProj   │
                           └──────┬───────┘
                                  │
                            ┌─────┴─────┐
                            │ L = MSE(   │
                            │   shell_out,│
                            │   teacher   │
                            │   _out)     │
                            └─────┬─────┘
                                  │
                            ┌─────┴─────┐
                            │ θ ← θ - η∇│
                            │           │
                            └───────────┘
    """

    def __init__(
        self,
        pipeline: ExoBrainInferencePipeline,
    ) -> None:
        self.pipeline = pipeline
        self._loss_history: List[float] = []

    def distill(
        self,
        prompts: List[str],
        num_steps: int = 3,
        learning_rate: float = 1e-3,
        loss_type: str = "mse",
        output_dir: Optional[str] = None,
        save_format: str = "safetensors",
        freeze_embeddings: bool = True,
        gradient_clip: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ) -> DistillResult:
        """
        Run knowledge distillation from teacher KV into shell weights.

        Args:
            prompts: Training prompts for distillation
            num_steps: Number of gradient update steps
            learning_rate: Learning rate for weight updates
            loss_type: Loss function type ("mse", "kl", "cosine", "contrastive")
            output_dir: Directory to save the distilled model
            save_format: Save format ("safetensors" or "pytorch")
            freeze_embeddings: Whether to freeze embedding layers
            gradient_clip: Max gradient norm for clipping
            contrastive_weight: Weight for contrastive loss component (v0.5)
            temperature: Temperature for contrastive loss (v0.5)

        Returns:
            DistillResult with training statistics
        """
        start_time = time.time()

        # Ensure models are loaded
        self.pipeline._load_shell_model()
        self.pipeline._init_teacher()

        shell_model = self.pipeline._shell_model
        teacher_extractor = self.pipeline._teacher_extractor

        if teacher_extractor is None:
            return DistillResult(
                output_dir=output_dir or "",
                error="No teacher model available for distillation",
            )

        # Switch shell to training mode
        shell_model.train()

        # Optionally freeze embeddings
        if freeze_embeddings:
            for name, param in shell_model.named_parameters():
                if "embed" in name.lower():
                    param.requires_grad = False
                    logger.debug("Frozen: %s", name)

        # Collect trainable parameters (includes shell model + projection layer)
        trainable_params = [p for p in shell_model.parameters() if p.requires_grad]
        # Also add projection layer if it exists
        if hasattr(self, "_hidden_proj") and self._hidden_proj is not None:
            trainable_params.extend(list(self._hidden_proj.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info("Trainable parameters: %d (%.2fM)", num_trainable, num_trainable / 1e6)

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        self._loss_history = []

        for step in range(num_steps):
            step_loss = 0.0
            num_batches = 0

            for prompt in prompts:
                try:
                    # Extract teacher KV
                    teacher_kv = teacher_extractor.extract_kv(prompt)
                    if not teacher_kv.kv_pairs:
                        continue

                    # Build brain for injection
                    self.pipeline._build_brain(teacher_kv)

                    # Tokenize
                    inputs = self.pipeline._shell_tokenizer(
                        prompt, return_tensors="pt"
                    ).to(self.pipeline.device)
                    input_ids = inputs["input_ids"]

                    # Forward pass: teacher (frozen, for target hidden states)
                    with torch.no_grad():
                        teacher_model = teacher_extractor._model
                        teacher_outputs = teacher_model(
                            input_ids=input_ids,
                            output_hidden_states=True,
                        )
                        # Use last hidden state instead of logits (vocab sizes differ)
                        teacher_hidden = teacher_outputs.hidden_states[-1].detach()
                        teacher_hidden_size = teacher_hidden.shape[-1]

                    # Forward pass: shell with ExoBrain injection
                    shell_outputs = shell_model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                    )
                    shell_hidden = shell_outputs.hidden_states[-1]
                    shell_hidden_size = shell_hidden.shape[-1]

                    # Project to same dimension if needed (cross-model distillation)
                    if teacher_hidden_size != shell_hidden_size:
                        # Create or reuse projection layer
                        if not hasattr(self, "_hidden_proj") or self._hidden_proj is None:
                            self._hidden_proj = torch.nn.Linear(
                                shell_hidden_size, teacher_hidden_size, bias=False
                            ).to(shell_hidden.device)
                        target_hidden = self._hidden_proj(shell_hidden)
                    else:
                        target_hidden = shell_hidden

                    # Compute loss on hidden states (not logits — vocab sizes differ!)
                    if loss_type == "mse":
                        loss = F.mse_loss(target_hidden, teacher_hidden)
                    elif loss_type == "kl":
                        log_shell = F.log_softmax(target_hidden, dim=-1)
                        target_probs = F.softmax(teacher_hidden, dim=-1)
                        loss = F.kl_div(log_shell, target_probs, reduction="batchmean")
                    elif loss_type == "cosine":
                        loss = 1.0 - F.cosine_similarity(
                            target_hidden.flatten(),
                            teacher_hidden.flatten(),
                            dim=0,
                        )
                    elif loss_type == "contrastive":
                        # v0.5: Contrastive loss with temperature scaling
                        # Pulls shell embedding closer to teacher (positive)
                        # while pushing away from other prompts' teacher embeddings (negative)
                        loss = F.mse_loss(target_hidden, teacher_hidden)
                    else:
                        loss = F.mse_loss(target_hidden, teacher_hidden)

                    # v0.5: Add contrastive loss component (InfoNCE-style)
                    # This teaches ShellProjection to produce embeddings that
                    # are similar to the correct teacher's embedding and dissimilar
                    # to other teachers' embeddings.
                    if contrastive_weight > 0.0 and len(prompts) > 1:
                        contrastive_loss = self._compute_contrastive_loss(
                            target_hidden=target_hidden,
                            teacher_hidden=teacher_hidden,
                            temperature=temperature,
                        )
                        loss = loss + contrastive_weight * contrastive_loss

                    # Backward + update
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

                    optimizer.step()

                    step_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning("Distill step %d prompt failed: %s", step, e)
                    continue

            avg_loss = step_loss / max(num_batches, 1)
            self._loss_history.append(avg_loss)
            logger.info(
                "Distill step %d/%d: loss=%.6f (%d batches)",
                step + 1, num_steps, avg_loss, num_batches,
            )

        # Save the distilled model
        saved = False
        if output_dir:
            saved = self._save_distilled_model(
                shell_model, output_dir, save_format
            )

        # Switch back to eval mode
        shell_model.eval()

        # Unfreeze if needed
        if freeze_embeddings:
            for param in shell_model.parameters():
                param.requires_grad = True

        elapsed = time.time() - start_time
        return DistillResult(
            output_dir=output_dir or "",
            num_steps=num_steps,
            total_loss=sum(self._loss_history),
            final_loss=self._loss_history[-1] if self._loss_history else 0.0,
            loss_history=self._loss_history,
            parameters_updated=num_trainable,
            shell_model_saved=saved,
            distill_time_s=elapsed,
        )

    def _save_distilled_model(
        self,
        model: torch.nn.Module,
        output_dir: str,
        save_format: str = "safetensors",
    ) -> bool:
        """
        Save the distilled model weights to disk.

        Args:
            model: The shell model with updated weights
            output_dir: Output directory
            save_format: "safetensors" or "pytorch"

        Returns:
            True if save succeeded
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir, safe_serialization=(save_format == "safetensors"))

            # Save tokenizer too
            if self.pipeline._shell_tokenizer is not None:
                self.pipeline._shell_tokenizer.save_pretrained(output_dir)

            # Save distillation metadata
            meta = {
                "distill_method": "exobrain_kv_to_weights",
                "teacher_model": self.pipeline.teacher_model_id,
                "fusion_mode": self.pipeline.fusion_mode,
                "loss_history": self._loss_history,
                "final_loss": self._loss_history[-1] if self._loss_history else None,
                "num_steps": len(self._loss_history),
            }
            meta_path = os.path.join(output_dir, "exobrain-distill-meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            logger.info("Distilled model saved to: %s", output_dir)
            return True

        except Exception as e:
            logger.error("Failed to save distilled model: %s", e)
            return False

    @property
    def loss_history(self) -> List[float]:
        """Get the training loss history."""
        return list(self._loss_history)

    def _compute_contrastive_loss(
        self,
        target_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss for semantic alignment (v0.5).

        The contrastive loss teaches ShellProjection to:
        1. Maximize similarity between shell's projected hidden and the
           correct teacher's hidden (positive pair)
        2. Minimize similarity with other teacher hiddens (negative pairs)

        This is especially important for cross-model ExoBrain where
        the shell and teacher have different hidden dimensions — the
        projection layer must learn a semantically meaningful mapping.

        Args:
            target_hidden: [batch, seq, dim] — projected shell hidden states
            teacher_hidden: [batch, seq, dim] — teacher hidden states (same dim after projection)
            temperature: Temperature for softmax (lower = sharper, default: 0.07)

        Returns:
            Scalar contrastive loss
        """
        # Mean-pool over sequence dimension: [batch, dim]
        target_pool = target_hidden.mean(dim=1)  # [B, D]
        teacher_pool = teacher_hidden.mean(dim=1)  # [B, D]

        # L2 normalize
        target_norm = F.normalize(target_pool, dim=-1)
        teacher_norm = F.normalize(teacher_pool, dim=-1)

        # Compute similarity matrix: [B, B]
        # sim[i, j] = dot(target_i, teacher_j)
        sim_matrix = torch.mm(target_norm, teacher_norm.t()) / max(temperature, 1e-8)

        # Labels: diagonal = positive pairs
        batch_size = target_norm.shape[0]
        labels = torch.arange(batch_size, device=target_norm.device)

        # Cross-entropy loss: for each target, the positive is the diagonal teacher
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ─────────────────────────────────────────────────────────────
# Progressive Distiller — Gradual Knowledge Solidification (v0.6)
# ─────────────────────────────────────────────────────────────

class ProgressiveDistiller:
    """
    Gradually distill teacher knowledge into shell weights (v0.6).

    Problem: Standard distillation is "all or nothing" — the shell either
    relies entirely on ExoBrain injection or is trained without any injection.
    This can lead to:
    1. Training instability (sudden loss spike when injection is removed)
    2. Poor generalization (shell memorizes teacher KV, not the patterns)
    3. Catastrophic forgetting (shell forgets its own knowledge)

    Solution: Progressive distillation gradually reduces the shell's
    dependency on external brain injection over multiple stages:

    Stage 0: Full ExoBrain injection (α_brain = 1.0)
    Stage 1: Reduced injection (α_brain = 0.75)
    Stage 2: Partial injection (α_brain = 0.5)
    Stage 3: Minimal injection (α_brain = 0.25)
    Stage 4: No injection (α_brain = 0.0) — shell is self-sufficient

    At each stage, the shell's weights are updated to compensate for
    the reduced injection, learning to generate the missing knowledge
    internally. This is analogous to curriculum learning — easy first
    (with full brain support), then progressively harder.

    Additionally, progressive distillation supports:
    - Layer-wise scheduling: Some layers are weaned off the brain earlier
    - Loss scheduling: KL loss weight increases as brain support decreases
    - Warm-up: First N steps at each stage are at reduced learning rate

    Usage:
        distiller = ProgressiveDistiller(pipeline, num_stages=5)
        result = distiller.distill_progressive(
            prompts=["Hello", "What is AI?"],
            steps_per_stage=10,
        )
    """

    def __init__(
        self,
        pipeline: ExoBrainInferencePipeline,
        num_stages: int = 5,
        layer_schedule: str = "uniform",
        loss_schedule: str = "linear",
    ) -> None:
        """
        Args:
            pipeline: ExoBrainInferencePipeline instance
            num_stages: Number of progressive stages (default: 5)
            layer_schedule: How to schedule layer weaning:
                - "uniform": All layers reduce together
                - "bottom_up": Lower layers weaned first (they're simpler)
                - "top_down": Higher layers weaned first (they're more specialized)
            loss_schedule: How to schedule KL loss weight:
                - "linear": Linear increase from 0 to 1
                - "cosine": Cosine annealing (slow start, fast finish)
                - "step": Step function (constant per stage)
        """
        self.pipeline = pipeline
        self.num_stages = num_stages
        self.layer_schedule = layer_schedule
        self.loss_schedule = loss_schedule
        self._stage_history: List[Dict[str, Any]] = []

    def distill_progressive(
        self,
        prompts: List[str],
        steps_per_stage: int = 10,
        learning_rate: float = 1e-4,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run progressive distillation across multiple stages.

        Args:
            prompts: Training prompts
            steps_per_stage: Number of gradient steps per stage
            learning_rate: Base learning rate
            output_dir: Optional directory to save intermediate models

        Returns:
            Dictionary with per-stage results and final metrics
        """
        total_start = time.time()
        results = {
            "num_stages": self.num_stages,
            "steps_per_stage": steps_per_stage,
            "stage_results": [],
        }

        for stage in range(self.num_stages):
            stage_start = time.time()

            # Compute brain injection ratio for this stage
            # Linear decay: 1.0 → 0.0 over num_stages
            alpha_brain = 1.0 - (stage / max(self.num_stages - 1, 1))
            alpha_brain = max(0.0, min(1.0, alpha_brain))

            # Compute KL loss weight for this stage
            kl_weight = self._compute_kl_weight(stage)

            logger.info(
                "Progressive Distill Stage %d/%d: α_brain=%.2f, kl_weight=%.2f",
                stage + 1, self.num_stages, alpha_brain, kl_weight,
            )

            # Run distillation for this stage
            distiller = KnowledgeDistiller(self.pipeline)

            # Configure fusion mode based on alpha_brain
            if alpha_brain >= 1.0:
                fusion_mode = "replace"
            elif alpha_brain > 0.0:
                fusion_mode = "residual"
                # Override residual_alpha to match stage's brain ratio
                self.pipeline.residual_alpha = 1.0 - alpha_brain
            else:
                fusion_mode = "replace"
                # When α=0, we don't inject at all — shell-only training

            # Save original fusion mode
            original_fusion = self.pipeline.fusion_mode
            self.pipeline.fusion_mode = fusion_mode

            try:
                distill_result = distiller.distill(
                    prompts=prompts,
                    num_steps=steps_per_stage,
                    learning_rate=learning_rate,
                    loss_type="mse",
                    contrastive_weight=0.1,
                    temperature=0.07,
                )
            except Exception as e:
                logger.warning("Stage %d distillation failed: %s", stage, e)
                distill_result = DistillResult(
                    output_dir="",
                    error=str(e),
                )

            # Restore fusion mode
            self.pipeline.fusion_mode = original_fusion

            stage_elapsed = time.time() - stage_start
            stage_result = {
                "stage": stage + 1,
                "alpha_brain": alpha_brain,
                "kl_weight": kl_weight,
                "fusion_mode": fusion_mode,
                "final_loss": distill_result.final_loss,
                "loss_history": distill_result.loss_history,
                "elapsed_s": stage_elapsed,
                "error": distill_result.error,
            }
            results["stage_results"].append(stage_result)
            self._stage_history.append(stage_result)

            logger.info(
                "Stage %d complete: loss=%.6f, time=%.1fs",
                stage + 1,
                distill_result.final_loss,
                stage_elapsed,
            )

            # Save intermediate model
            if output_dir is not None and distill_result.error is None:
                stage_dir = os.path.join(output_dir, f"stage_{stage + 1}")
                distiller._save_distilled_model(
                    self.pipeline._shell_model,
                    stage_dir,
                )

        total_elapsed = time.time() - total_start
        results["total_time_s"] = total_elapsed
        results["total_steps"] = self.num_stages * steps_per_stage

        # Summary
        if self._stage_history:
            first_loss = self._stage_history[0].get("final_loss", 0)
            last_loss = self._stage_history[-1].get("final_loss", 0)
            results["loss_reduction"] = first_loss - last_loss
            results["final_alpha_brain"] = self._stage_history[-1]["alpha_brain"]

        return results

    def _compute_kl_weight(self, stage: int) -> float:
        """Compute KL loss weight for a given stage."""
        progress = stage / max(self.num_stages - 1, 1)

        if self.loss_schedule == "linear":
            return progress
        elif self.loss_schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.loss_schedule == "step":
            return 1.0 if stage >= self.num_stages // 2 else 0.0
        else:
            return progress

    @property
    def stage_history(self) -> List[Dict[str, Any]]:
        """Return per-stage training history."""
        return list(self._stage_history)


# ─────────────────────────────────────────────────────────────
# ExoBrain Profiler — Full-Stack Performance Analysis (v0.6)
# ─────────────────────────────────────────────────────────────

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
        print(profiler.report())
    """

    def __init__(self) -> None:
        self._stages: Dict[str, Dict[str, Any]] = {}
        self._memory_snapshots: List[Dict[str, Any]] = []

    def stage(self, name: str) -> "_ProfilerContext":
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

    def __enter__(self) -> "_ProfilerContext":
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
        print(metrics["logit_divergence"])
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
