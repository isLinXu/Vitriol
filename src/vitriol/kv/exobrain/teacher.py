"""Teacher KV extraction, projection and inference/distill result types."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from vitriol.utils.hf_loading import load_causallm, load_tokenizer

logger = logging.getLogger(__name__)


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
        trust_remote_code: bool = False,
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
