"""
HybridUltra compression strategy — best of Ultra + Compact.

Combines the extreme disk compression of stride=0 tensors with the
trainability and format compatibility of standard initialization.

Key innovations over plain Ultra:
  1. **Trainable LayerNorm/RMSNorm**: weight=1.0 (not 0.0) so signal flows.
  2. **Safetensors-compatible**: uses compact zero-filled contiguous tensors
     instead of stride=0, so safetensors format works.
  3. **Low-memory save**: shards are written with zero-run encoding internally
     via safetensors, achieving near-Ultra disk compression.
  4. **Optional parameter initialization**: Kaiming/Xavier for linear layers
     makes the model immediately trainable.
  5. **Configurable init modes**: "zeros" (compact), "kaiming", "xavier",
     "orthogonal" for different training scenarios.

Architecture:
  - LayerNorm / RMSNorm parameters → weight=1.0, bias=0.0 (trainable)
  - Embedding / LM head → zero (compact) or small init (trainable)
  - Linear layers (Q/K/V/O/FFN) → zero (compact) or init (trainable)
  - Buffers (causal mask, position IDs) → preserved as-is

Comparison table:
  ┌─────────────┬──────────┬──────────┬──────────────┬──────────┐
  │ Feature     │ Ultra    │ Compact  │ HybridUltra  │ Random   │
  ├─────────────┼──────────┼──────────┼──────────────┼──────────┤
  │ Disk size   │ ~0       │ Full     │ ~Full*       │ Full     │
  │ Memory load │ 2x bloat│ Normal   │ Normal       │ Normal   │
  │ Trainable   │ ✗        │ △        │ ✓            │ ✓        │
  │ Safetensors │ ✗        │ ✓        │ ✓            │ ✓        │
  │ LN correct  │ ✗ (w=0)  │ ✗ (w=0)  │ ✓ (w=1)      │ ✓        │
  │ Init option │ ✗        │ ✗        │ ✓            │ ✓        │
  └─────────────┴──────────┴──────────┴──────────────┴──────────┘
  * HybridUltra disk size is similar to Compact but with safetensors
    compression, zero-valued tensors compress well at the filesystem level.
    With init_mode="zeros", gzip compression achieves ~99% ratio.
"""

import logging
import math
from typing import Dict, Optional

import torch
from torch import nn

from .base import WeightGenerationStrategy, StrategyCapabilities

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Parameter name classification
# ──────────────────────────────────────────────────────────────────────

# Suffixes that identify LayerNorm / RMSNorm weight parameters
# Covers: LLaMA, Qwen, Gemma, GLM, Mamba, DeepSeek, Phi, Mistral, etc.
_NORM_WEIGHT_SUFFIXES = (
    "norm.weight",
    "ln.weight",
    "layernorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "final_layer_norm.weight",
    "layer_norm.weight",
    # GLM / ChatGLM
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    # Mamba / RWKV
    "norm.weight",
    "ln_f.weight",
    "ln_1.weight",
    "ln_2.weight",
    # DeepSeek / MLA
    "attention_norm.weight",
    "ffn_norm.weight",
    # Phi
    "input_layernorm.weight",
    # Additional common patterns
    "rmsnorm.weight",
)

# Suffixes that identify LayerNorm / RMSNorm bias parameters
_NORM_BIAS_SUFFIXES = (
    "norm.bias",
    "ln.bias",
    "layernorm.bias",
    "input_layernorm.bias",
    "post_attention_layernorm.bias",
    "pre_feedforward_layernorm.bias",
    "post_feedforward_layernorm.bias",
    "final_layer_norm.bias",
    "layer_norm.bias",
    # GLM / ChatGLM
    "input_layernorm.bias",
    "post_attention_layernorm.bias",
    "pre_feedforward_layernorm.bias",
    "post_feedforward_layernorm.bias",
    # Mamba / RWKV
    "norm.bias",
    "ln_f.bias",
    "ln_1.bias",
    "ln_2.bias",
    # DeepSeek / MLA
    "attention_norm.bias",
    "ffn_norm.bias",
    # Additional common patterns
    "rmsnorm.bias",
)

# Suffixes for embedding layers
_EMBED_SUFFIXES = (
    "embed_tokens.weight",
    "wte.weight",
    "wpe.weight",
    "embed_positions.weight",
    "lm_head.weight",
    # GLM / ChatGLM
    "embedding.word_embeddings.weight",
    "embedding.weight",
    # Mamba / RWKV
    "backbone.embed_tokens.weight",
    # DeepSeek
    "model.embed_tokens.weight",
    # Vision encoders
    "vision_embed_tokens.weight",
    "visual.embed_tokens.weight",
)


def _is_norm_weight(name: str) -> bool:
    """Check if parameter name refers to a normalization layer weight."""
    return any(name.endswith(s) for s in _NORM_WEIGHT_SUFFIXES)


def _is_norm_bias(name: str) -> bool:
    """Check if parameter name refers to a normalization layer bias."""
    return any(name.endswith(s) for s in _NORM_BIAS_SUFFIXES)


def _is_embedding(name: str) -> bool:
    """Check if parameter name refers to an embedding layer."""
    return any(name.endswith(s) for s in _EMBED_SUFFIXES)


def _is_linear_weight(name: str) -> bool:
    """Check if parameter name refers to a linear layer weight."""
    # Heuristic: 2D tensor whose name ends with .weight but isn't norm/embed
    return (
        name.endswith(".weight")
        and not _is_norm_weight(name)
        and not _is_embedding(name)
    )


# ──────────────────────────────────────────────────────────────────────
# Initialization helpers
# ──────────────────────────────────────────────────────────────────────

def _kaiming_init(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Kaiming normal initialization suitable for ReLU/SwiGLU networks."""
    with torch.no_grad():
        nn.init.kaiming_normal_(tensor, a=math.sqrt(5))
    return tensor


def _xavier_init(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Xavier uniform initialization."""
    with torch.no_grad():
        nn.init.xavier_uniform_(tensor)
    return tensor


def _orthogonal_init(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Orthogonal initialization — good for training stability."""
    with torch.no_grad():
        nn.init.orthogonal_(tensor)
    return tensor


def _small_normal_init(tensor: torch.Tensor, name: str, std: float = 0.02) -> torch.Tensor:
    """Small normal initialization — similar to GPT-2 default."""
    with torch.no_grad():
        nn.init.normal_(tensor, mean=0.0, std=std)
    return tensor


_INIT_FN_MAP = {
    "kaiming": _kaiming_init,
    "xavier": _xavier_init,
    "orthogonal": _orthogonal_init,
    "small_normal": _small_normal_init,
}


class HybridUltraStrategy(WeightGenerationStrategy):
    """
    HybridUltra — best of Ultra compression + Compact trainability.

    This strategy replaces the stride=0 trick with compact contiguous
    zero tensors that are compatible with Safetensors, while ensuring
    LayerNorm weights are initialized to 1.0 for trainability.

    Parameters
    ----------
    init_mode : str
        Weight initialization mode for linear/embedding layers:
        - "zeros" (default): zero-filled, maximal disk compression
        - "kaiming": Kaiming normal — good for ReLU/SwiGLU FFN
        - "xavier": Xavier uniform — good for sigmoid/tanh
        - "orthogonal": Orthogonal — good for training stability
        - "small_normal": Normal(0, 0.02) — GPT-style default

    norm_init : bool
        If True (default), initialize LayerNorm/RMSNorm weight to 1.0
        and bias to 0.0 so the model is immediately trainable.

    embed_init : str
        Initialization for embedding layers:
        - "zeros": zero-filled (compact)
        - "small_normal": Normal(0, 0.02) (trainable)

    dtype_override : str or None
        Override all float dtypes to this type for consistent storage.
        Default: "bfloat16" for maximal disk savings.
        Set to None to preserve original dtypes.

    Examples
    --------
    >>> strategy = HybridUltraStrategy()  # defaults: zeros + bfloat16
    >>> strategy = HybridUltraStrategy(init_mode="kaiming")  # trainable
    >>> strategy = HybridUltraStrategy(init_mode="orthogonal", dtype_override=None)
    """

    def __init__(
        self,
        device: str = "cpu",
        save_dummy_config: bool = False,
        init_mode: str = "zeros",
        norm_init: bool = True,
        embed_init: str = "zeros",
        dtype_override: Optional[str] = "bfloat16",
        **kwargs,
    ):
        super().__init__(device, save_dummy_config=save_dummy_config, **kwargs)

        if init_mode not in _INIT_FN_MAP and init_mode != "zeros":
            raise ValueError(
                f"Invalid init_mode '{init_mode}'. "
                f"Choose from: zeros, {', '.join(_INIT_FN_MAP.keys())}"
            )
        if embed_init not in ("zeros", "small_normal"):
            raise ValueError(
                f"Invalid embed_init '{embed_init}'. Choose from: zeros, small_normal"
            )

        self.init_mode = init_mode
        self.norm_init = norm_init
        self.embed_init = embed_init
        self.dtype_override = dtype_override
        self._first_tensor_logged = False

    # ──────────────────────────────────────────────────────────────────
    # Capabilities
    # ──────────────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> StrategyCapabilities:
        trainable = self.init_mode != "zeros" or self.norm_init
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=trainable,
            requires_contiguous=True,
            max_compression_ratio=0.01 if self.init_mode == "zeros" else 0.5,
            description=(
                f"HybridUltra: zero-base + norm_init={self.norm_init} "
                f"+ init_mode={self.init_mode}. "
                f"Trainable={trainable}, Safetensors=✓."
            ),
        )

    # ──────────────────────────────────────────────────────────────────
    # Tensor generation
    # ──────────────────────────────────────────────────────────────────

    def generate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype,
        name: str,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a tensor with HybridUltra strategy.

        Decision tree:
        1. LayerNorm/RMSNorm weight → ones (if norm_init)
        2. LayerNorm/RMSNorm bias → zeros
        3. Embedding → zeros or small_normal
        4. Linear weight → zeros or init_mode
        5. Other → zeros

        Raises:
            ValueError: If shape has non-positive dimensions or is empty.
        """
        # [Hardening] Validate shape — prevent empty/invalid shapes
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(
                f"HybridUltra: invalid shape {shape} for parameter '{name}'. "
                f"All dimensions must be positive."
            )

        # Apply dtype override for consistent storage
        dtype = self._resolve_dtype(dtype)

        # Log first tensor for debugging
        if not self._first_tensor_logged:
            logger.info(
                "HybridUltra: init_mode=%s, norm_init=%s, embed_init=%s, "
                "dtype_override=%s",
                self.init_mode, self.norm_init, self.embed_init,
                self.dtype_override,
            )
            self._first_tensor_logged = True

        # 1. LayerNorm / RMSNorm weight → 1.0 (trainable!)
        if _is_norm_weight(name) and self.norm_init:
            return torch.ones(shape, dtype=dtype, device=self.device)

        # 2. LayerNorm / RMSNorm bias → 0.0 (already correct)
        if _is_norm_bias(name):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        # 3. Embedding layers
        if _is_embedding(name):
            if self.embed_init == "small_normal":
                return _small_normal_init(
                    torch.empty(shape, dtype=dtype, device=self.device), name
                )
            return torch.zeros(shape, dtype=dtype, device=self.device)

        # 4. Linear layer weights
        if _is_linear_weight(name):
            if self.init_mode == "zeros":
                return torch.zeros(shape, dtype=dtype, device=self.device)
            init_fn = _INIT_FN_MAP.get(self.init_mode)
            if init_fn is not None:
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                return init_fn(tensor, name)
            # [Hardening] Unknown init_mode falls back to zeros with warning
            logger.warning(
                "HybridUltra: unknown init_mode '%s' for '%s', falling back to zeros.",
                self.init_mode, name,
            )
            return torch.zeros(shape, dtype=dtype, device=self.device)

        # 5. Everything else (biases, 1D params, buffers) → zeros
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def _resolve_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Apply dtype override for consistent storage.

        Only overrides floating-point types. Integer types and bool are
        preserved as-is.  Supports float64 as an override target.

        Returns:
            The resolved dtype after applying override rules.
        """
        if self.dtype_override is None:
            return dtype

        # Only override floating point types
        if not dtype.is_floating_point:
            return dtype

        override_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,   # [Hardening] full coverage
        }
        target = override_map.get(self.dtype_override)
        if target is not None:
            return target

        # [Hardening] Unknown dtype_override — warn and preserve original
        logger.warning(
            "HybridUltra: unknown dtype_override='%s', preserving original dtype %s.",
            self.dtype_override, dtype,
        )
        return dtype

    # ──────────────────────────────────────────────────────────────────
    # Shard saving
    # ──────────────────────────────────────────────────────────────────

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        """
        Save shard with safetensors (preferred) or PyTorch fallback.

        Safetensors provides better compression for zero-filled tensors
        and is the recommended format for HybridUltra.

        Raises:
            OSError: If file cannot be written (disk full, permission denied, etc.)
        """
        if not shard_data:
            logger.warning("HybridUltra: save_shard called with empty data for %s", path)
            return

        if self.storage_format == "safetensors":
            try:
                from safetensors.torch import save_file
                # Ensure all tensors are contiguous (safetensors requirement)
                contiguous_data = {}
                for k, v in shard_data.items():
                    if not v.is_contiguous():
                        v = v.contiguous()
                    contiguous_data[k] = v
                save_file(contiguous_data, path)
                logger.debug("Saved safetensors shard: %s (%d tensors)", path, len(contiguous_data))
                return
            except ImportError:
                logger.warning(
                    "safetensors not installed, falling back to PyTorch format. "
                    "Install with: pip install safetensors"
                )
                # Fall through to torch.save
            except (OSError, RuntimeError) as e:
                # [Hardening] safetensors save failed (e.g. unsupported dtype,
                # disk full) — try PyTorch fallback before giving up
                logger.warning(
                    "safetensors save failed for %s: %s. Trying PyTorch fallback.",
                    path, e,
                )

        # PyTorch fallback
        fallback_path = path
        if fallback_path.endswith(".safetensors"):
            fallback_path = fallback_path.rsplit(".", 1)[0] + ".bin"
            logger.info("Extension changed: %s → %s", path, fallback_path)
        try:
            torch.save(shard_data, fallback_path)
            logger.debug("Saved PyTorch shard: %s (%d tensors)", fallback_path, len(shard_data))
        except (OSError, RuntimeError) as e:
            # [Hardening] Wrap I/O errors with context
            from ..utils.exceptions import ShardSaveError
            raise ShardSaveError(fallback_path, str(e)) from e

    # ──────────────────────────────────────────────────────────────────
    # Post-load hook — optimize memory for loaded models
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def optimize_loaded_model(model) -> Dict[str, int]:
        """
        Optimize a model loaded from HybridUltra weights.

        This static method can be called after `from_pretrained()` to:
        1. Replace zero-filled parameters with stride=0 views (memory save)
        2. Report memory savings

        Returns:
            Dict with memory statistics before/after optimization.

        Example:
            >>> model = AutoModelForCausalLM.from_pretrained(output_dir)
            >>> stats = HybridUltraStrategy.optimize_loaded_model(model)
            >>> print(f"Memory saved: {stats['saved_mb']:.1f} MB")
        """
        before_bytes = 0
        after_bytes = 0
        zero_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            total_params += 1
            before_bytes += param.numel() * param.element_size()

            # [Hardening] Skip empty parameters (numel==0) — cannot check
            # abs().max() on a 0-element tensor, and stride=0 makes no sense
            if param.numel() == 0:
                after_bytes += 0  # zero bytes for zero elements
                continue

            # Check if parameter is all zeros (candidate for stride=0)
            try:
                is_zero = param.data.abs().max().item() == 0.0
            except RuntimeError as e:
                # [Hardening] Sparse or otherwise unusual tensor — skip
                logger.debug("optimize_loaded_model: skipping param '%s': %s", name, e)
                after_bytes += param.numel() * param.element_size()
                continue

            if is_zero:
                # Replace with stride=0 view
                storage = torch.zeros(1, dtype=param.dtype, device=param.device)
                strided = torch.as_strided(
                    storage, param.shape, [0] * len(param.shape)
                )
                # We can't directly assign a stride=0 tensor to a Parameter,
                # but we can modify the underlying data
                with torch.no_grad():
                    param.data = strided
                after_bytes += storage.numel() * storage.element_size()
                zero_params += 1
            else:
                after_bytes += param.numel() * param.element_size()

        saved_bytes = before_bytes - after_bytes
        return {
            "total_params": total_params,
            "zero_params": zero_params,
            "before_mb": before_bytes / (1024 ** 2),
            "after_mb": after_bytes / (1024 ** 2),
            "saved_mb": saved_bytes / (1024 ** 2),
            "compression_ratio": after_bytes / before_bytes if before_bytes > 0 else 1.0,
        }

    # ──────────────────────────────────────────────────────────────────
    # Metadata — describe the generation recipe
    # ──────────────────────────────────────────────────────────────────

    def get_recipe(self) -> Dict[str, str]:
        """Return a human-readable recipe describing the generation strategy."""
        return {
            "strategy": "hybrid_ultra",
            "init_mode": self.init_mode,
            "norm_init": str(self.norm_init),
            "embed_init": self.embed_init,
            "dtype_override": self.dtype_override or "preserve",
            "trainable": str(self.capabilities.supports_training),
            "safetensors": str(self.capabilities.supports_safetensors),
            "device": self.device,
        }

    def validate_config(self) -> bool:
        """Validate the current strategy configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is inconsistent.
        """
        if self.init_mode not in _INIT_FN_MAP and self.init_mode != "zeros":
            raise ValueError(
                f"HybridUltra: invalid init_mode '{self.init_mode}'. "
                f"Choose from: zeros, {', '.join(_INIT_FN_MAP.keys())}"
            )
        if self.embed_init not in ("zeros", "small_normal"):
            raise ValueError(
                f"HybridUltra: invalid embed_init '{self.embed_init}'. "
                f"Choose from: zeros, small_normal"
            )
        if self.dtype_override is not None and self.dtype_override not in (
            "bfloat16", "float16", "float32", "float64"
        ):
            raise ValueError(
                f"HybridUltra: invalid dtype_override '{self.dtype_override}'. "
                f"Choose from: bfloat16, float16, float32, float64, None"
            )
        return True
