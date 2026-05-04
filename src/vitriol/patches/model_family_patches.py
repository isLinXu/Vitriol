"""
Model family-specific configuration patches.

This module provides a registry-based system for applying model-specific
patches to configurations, handling differences between model families.
"""

import logging
from typing import Any, Dict, List, Tuple, Callable

logger = logging.getLogger(__name__)

# Default RoPE parameters
_ROPE_DEFAULTS: Dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10000.0,
}


def _set_missing(obj: Any, **kv: Any) -> None:
    """Set attributes on an object if they don't already exist."""
    for k, v in kv.items():
        # Many HF configs define attributes with default None.
        # Treat "None" as missing so patches can inject sane defaults.
        if (not hasattr(obj, k)) or (getattr(obj, k, None) is None):
            try:
                setattr(obj, k, v)
            except Exception as e:
                logger.debug("Could not set attribute %s on config: %s", k, e)


def _ensure_rope_params(config: Any) -> None:
    """Ensure RoPE parameters are present with defaults."""
    existing = getattr(config, "rope_parameters", None)
    if not isinstance(existing, dict):
        setattr(config, "rope_parameters", dict(_ROPE_DEFAULTS))
    else:
        for k, v in _ROPE_DEFAULTS.items():
            existing.setdefault(k, v)


def _fix_rope_theta(config: Any) -> None:
    """Fix RoPE theta if it's a list or None."""
    theta = getattr(config, "rope_theta", None)
    if isinstance(theta, (list, tuple)):
        config.rope_theta = float(theta[0]) if theta else 10000.0
    elif theta is None:
        config.rope_theta = 10000.0
    
    if isinstance(getattr(config, "rope_scaling", None), (list, tuple)):
        config.rope_scaling = None


def _fix_rms_norm(config: Any) -> None:
    """Fix RMS norm epsilon if it's a list."""
    eps = getattr(config, "rms_norm_eps", None)
    if isinstance(eps, (list, tuple)):
        config.rms_norm_eps = float(eps[0]) if eps else 1e-6


class PatchRegistry:
    """
    Decorator-based registry mapping family keywords to patch functions.
    
    Match order: config class-name substring, then model_id substring.
    
    Example:
        >>> @PatchRegistry.register("qwen")
        ... def patch_qwen(config, model_id):
        ...     config.pad_token_id = 0
        ...
        >>> config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")
        >>> PatchRegistry.apply(config, "Qwen/Qwen2-7B")
    """
    
    _entries: List[Tuple[Tuple[str, ...], Callable]] = []
    
    @classmethod
    def register(cls, *keys: str) -> Callable:
        """
        Register a patch function for specific model keywords.
        
        Args:
            *keys: Keywords to match (case-insensitive)
        
        Returns:
            Decorator function
        
        Example:
            @PatchRegistry.register("qwen", "qwenvl")
            def patch_qwen_family(config, model_id):
                # Apply Qwen-specific patches
                pass
        """
        def decorator(fn: Callable) -> Callable:
            cls._entries.append((tuple(k.lower() for k in keys), fn))
            return fn
        return decorator
    
    @classmethod
    def apply(cls, config: Any, model_id: str) -> None:
        """
        Apply all matching patches to a config.
        
        Args:
            config: Model configuration object
            model_id: Model identifier (e.g., "Qwen/Qwen2-7B")
        """
        cname = type(config).__name__.lower()
        mid = model_id.lower()
        
        for keys, fn in cls._entries:
            if any(k in cname or k in mid for k in keys):
                try:
                    fn(config, model_id)
                except Exception as e:
                    logger.debug("Patch %s raised: %s", keys, e)


# ============================================================================
# Model Family Patches
# ============================================================================

@PatchRegistry.register("qwen")
def _patch_qwen(config: Any, _mid: str) -> None:
    """Patch Qwen family models."""
    _set_missing(
        config,
        pad_token_id=0,
        vocab_size=32000,
        intermediate_size=512,
        mlp_only_layers=[],
    )
    
    if not hasattr(config, "qkv_bias"):
        config.qkv_bias = getattr(config, "attention_bias", True)


@PatchRegistry.register("minimax")
def _patch_minimax(config: Any, _mid: str) -> None:
    """Patch MiniMax family models."""
    _ensure_rope_params(config)
    _set_missing(config, use_cache=False)


@PatchRegistry.register("step3p5", "step-3.5")
def _patch_step35(config: Any, _mid: str) -> None:
    """Patch Step 3.5 family models."""
    _set_missing(
        config,
        pad_token_id=0,
        vocab_size=32000,
        qkv_bias=True,
    )
    _fix_rope_theta(config)
    _fix_rms_norm(config)
    _ensure_rope_params(config)


@PatchRegistry.register("mistral")
def _patch_mistral(config: Any, _mid: str) -> None:
    """Patch Mistral family models."""
    _set_missing(
        config,
        pad_token_id=0,
        sliding_window=None,
        max_window_layers=getattr(config, "num_hidden_layers", 32),
    )
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("mixtral")
def _patch_mixtral(config: Any, _mid: str) -> None:
    """Patch Mixtral family models."""
    _patch_mistral(config, _mid)
    _set_missing(config, num_experts=8, num_experts_per_tok=2, router_aux_loss_coef=0.001)


@PatchRegistry.register("phi3", "phi-3", "phi_3", "phi4", "phi-4")
def _patch_phi(config: Any, _mid: str) -> None:
    """Patch Phi family models."""
    _set_missing(
        config,
        pad_token_id=32000,
        original_max_position_embeddings=4096,
        sliding_window=None,
    )
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("gemma")
def _patch_gemma(config: Any, _mid: str) -> None:
    """Patch Gemma family models."""
    _set_missing(config, pad_token_id=0)
    if not hasattr(config, "head_dim"):
        config.head_dim = getattr(config, "hidden_size", 2048) // max(
            getattr(config, "num_attention_heads", 8), 1
        )
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("falcon")
def _patch_falcon(config: Any, _mid: str) -> None:
    """Patch Falcon family models."""
    _set_missing(
        config,
        alibi=False,
        new_decoder_architecture=True,
        parallel_attn=True,
        multi_query=False,
    )
    _fix_rope_theta(config)


@PatchRegistry.register("llama")
def _patch_llama(config: Any, _mid: str) -> None:
    """Patch LLaMA family models."""
    _set_missing(
        config,
        pad_token_id=0,
    )
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("deepseek")
def _patch_deepseek(config: Any, _mid: str) -> None:
    """Patch DeepSeek family models."""
    _set_missing(
        config,
        pad_token_id=0,
    )
    _fix_rope_theta(config)
    _fix_rms_norm(config)
    _ensure_rope_params(config)
    
    # Handle DeepSeek-V3 MLA
    if hasattr(config, "q_lora_rank"):
        _set_missing(config, kv_lora_rank=getattr(config, "q_lora_rank", 512))


@PatchRegistry.register("kimi")
def _patch_kimi(config: Any, _mid: str) -> None:
    """Patch Kimi family models (DeepSeek-V3 architecture)."""
    _patch_deepseek(config, _mid)


@PatchRegistry.register("glm")
def _patch_glm(config: Any, _mid: str) -> None:
    """Patch GLM family models."""
    _set_missing(
        config,
        pad_token_id=0,
    )
    _fix_rope_theta(config)
    _fix_rms_norm(config)


@PatchRegistry.register("ernie")
def _patch_ernie(config: Any, _mid: str) -> None:
    """Patch ERNIE family models."""
    _set_missing(
        config,
        pad_token_id=0,
    )
    _fix_rope_theta(config)
    _fix_rms_norm(config)


@PatchRegistry.register("intern")
def _patch_intern(config: Any, _mid: str) -> None:
    """Patch Intern family models."""
    _set_missing(
        config,
        pad_token_id=0,
    )
    _fix_rope_theta(config)
    _fix_rms_norm(config)


@PatchRegistry.register("gpt2")
def _patch_gpt2(config: Any, _mid: str) -> None:
    """Patch GPT-2 family models."""
    _set_missing(
        config,
        pad_token_id=50256,
        use_cache=True,
    )


@PatchRegistry.register("gptneox", "gpt-neox")
def _patch_gpt_neox(config: Any, _mid: str) -> None:
    """Patch GPT-NeoX family models."""
    _patch_gpt2(config, _mid)


@PatchRegistry.register("bloom")
def _patch_bloom(config: Any, _mid: str) -> None:
    """Patch BLOOM family models."""
    _set_missing(config, pad_token_id=3, apply_residual_connection_post_layernorm=False)


@PatchRegistry.register("opt")
def _patch_opt(config: Any, _mid: str) -> None:
    """Patch OPT family models."""
    _set_missing(config, pad_token_id=1, do_layer_norm_before=True)


@PatchRegistry.register("yi", "internlm2", "internlm-2")
def _patch_yi(config: Any, _mid: str) -> None:
    """Patch Yi and InternLM2 family models."""
    _set_missing(config, pad_token_id=0)
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("baichuan")
def _patch_baichuan(config: Any, _mid: str) -> None:
    """Patch Baichuan family models."""
    _set_missing(config, pad_token_id=0, model_max_length=4096)


@PatchRegistry.register("chatglm")
def _patch_chatglm(config: Any, _mid: str) -> None:
    """Patch ChatGLM family models."""
    _set_missing(
        config,
        add_bias_linear=False,
        add_qkv_bias=True,
        apply_query_key_layer_scaling=False,
        post_layer_norm=True,
        rmsnorm=True,
    )


@PatchRegistry.register("cohere", "command-r")
def _patch_cohere(config: Any, _mid: str) -> None:
    """Patch Cohere Command-R family models."""
    _set_missing(config, pad_token_id=0, use_qk_norm=True)
    _fix_rope_theta(config)
    _ensure_rope_params(config)


@PatchRegistry.register("mamba")
def _patch_mamba(config: Any, _mid: str) -> None:
    """Patch Mamba family models."""
    _set_missing(config, d_state=16, d_conv=4, expand=2, dt_rank="auto", use_fast_path=False)
    if not hasattr(config, "d_model"):
        config.d_model = getattr(config, "hidden_size", 768)


@PatchRegistry.register("jamba")
def _patch_jamba(config: Any, _mid: str) -> None:
    """Patch Jamba family models."""
    _patch_mamba(config, _mid)
    _set_missing(config, attn_layer_offset=4, attn_layer_period=8, num_experts=16, num_experts_per_tok=2)


@PatchRegistry.register("qwen2-vl", "qwen2vl", "internvl", "llava", "cogvlm", "idefics")
def _patch_vlm(config: Any, _mid: str) -> None:
    """Patch multimodal VLM families with text and vision sub-configs."""
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        _set_missing(
            vision_config,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            intermediate_size=4096,
            patch_size=14,
            image_size=336,
        )

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        _set_missing(text_config, pad_token_id=0)
