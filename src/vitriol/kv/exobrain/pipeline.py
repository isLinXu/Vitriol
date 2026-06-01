"""ExoBrain inference pipeline."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from vitriol.utils.hf_loading import load_causallm, load_tokenizer

from .scheduler import KVPrefetcher
from .teacher import HeadDimProjection, InferenceResult, TeacherKVCache, TeacherKVExtractor

logger = logging.getLogger(__name__)


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
        trust_remote_code: bool = False,
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
        from . import (
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
                from . import ExoBrainBackend
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
