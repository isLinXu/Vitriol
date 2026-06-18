import datetime
import importlib.util
import json
import logging
import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Dict

import torch
from transformers import AutoConfig
from transformers.utils import cached_file

if TYPE_CHECKING:
    from .generator import MinimalWeightGenerator

logger = logging.getLogger(__name__)


def save_configs(generator, hf_config) -> None:
    logger.info("Saving config…")

    # ── Always save original HF config as meta-config.json ──────────
    # This preserves the full, unmodified model configuration from HuggingFace
    # so that visualization tools can render the real architecture even when
    # the active config.json has been shrunk or modified.
    try:
        meta_src = None
        if os.path.isdir(generator.model_id):
            local_cfg = os.path.join(generator.model_id, "config.json")
            if os.path.exists(local_cfg):
                meta_src = local_cfg
        if meta_src is None:
            meta_src = cached_file(
                generator.model_id,
                "config.json",
                _raise_exceptions_for_missing_entries=False,
                local_files_only=(
                    getattr(generator.config.security, "local_files_only", False)
                    or not getattr(generator.config.security, "allow_network", True)
                ),
            )
        if meta_src and os.path.exists(meta_src):
            shutil.copy2(meta_src, os.path.join(generator.output_dir, "meta-config.json"))
            logger.info("Saved raw config.json → meta-config.json")
        else:
            logger.warning("meta-config.json save skipped (config.json not found).")
    except Exception as e:
        logger.warning("meta-config.json save failed: %s", e)

    # ── Legacy: also save config_meta.json if save_dummy_config is set ──
    if getattr(generator.strategy, "save_dummy_config", False):
        meta_cfg_path = os.path.join(generator.output_dir, "meta-config.json")
        legacy_path = os.path.join(generator.output_dir, "config_meta.json")
        if os.path.exists(meta_cfg_path) and not os.path.exists(legacy_path):
            shutil.copy2(meta_cfg_path, legacy_path)
            logger.info("Copied meta-config.json → config_meta.json (legacy)")

    # ── Save the (shrink) active config ──────────────────────────────
    hf_config.save_pretrained(generator.output_dir)

    # ── Post-process: strip raw dict sub-configs from the saved config ──
    # Some adapters (e.g. Qwen35MoeAdapter) strip sub-config dicts in-memory
    # but save_pretrained() may re-serialise them.  Also, _patch_model_config
    # may have already removed them.  As a safety net, re-read the saved JSON
    # and strip any remaining raw dict sub-configs.
    #
    # IMPORTANT: For multimodal models (Gemma-4, etc.), we preserve
    # vision_config because the model class needs it to instantiate the
    # vision tower.  We only strip if it's a raw dict that would crash
    # PretrainedConfig deserialisation.  If vision_config has been
    # converted to a PretrainedConfig object by the adapter, it's safe
    # to keep.
    saved_cfg_path = os.path.join(generator.output_dir, "config.json")
    try:
        with open(saved_cfg_path) as f:
            saved_cfg = json.load(f)
        _stripped = False

        # Detect if this is a multimodal model
        _is_vlm = bool(
            saved_cfg.get("vision_soft_tokens_per_image")
            or saved_cfg.get("image_token_id") is not None
            or any("ConditionalGeneration" in a for a in saved_cfg.get("architectures", []))
        )

        def _is_transformers_known_model_type(model_type):
            if not isinstance(model_type, str) or not model_type:
                return False
            try:
                AutoConfig.for_model(model_type)
                return True
            except (KeyError, ValueError, OSError, AttributeError):
                return False

        # --- Align model_type and architectures with original meta-config ---
        meta_cfg_path = os.path.join(generator.output_dir, "meta-config.json")
        if os.path.exists(meta_cfg_path):
            try:
                with open(meta_cfg_path) as mf:
                    meta_cfg = json.load(mf)
                should_align_model_type = _is_transformers_known_model_type(meta_cfg.get("model_type"))
                if (
                    "model_type" in meta_cfg
                    and saved_cfg.get("model_type") != meta_cfg["model_type"]
                    and should_align_model_type
                ):
                    saved_cfg["model_type"] = meta_cfg["model_type"]
                    _stripped = True
                    logger.info("Aligned model_type to original: %s", meta_cfg["model_type"])
                elif "model_type" in meta_cfg and saved_cfg.get("model_type") != meta_cfg["model_type"]:
                    logger.info(
                        "Kept reloadable fallback model_type '%s'; original unknown model_type is preserved in meta-config.json: %s",
                        saved_cfg.get("model_type"),
                        meta_cfg["model_type"],
                    )
                if (
                    "architectures" in meta_cfg
                    and saved_cfg.get("architectures") != meta_cfg["architectures"]
                    and should_align_model_type
                ):
                    saved_cfg["architectures"] = meta_cfg["architectures"]
                    _stripped = True
                    logger.info("Aligned architectures to original: %s", meta_cfg["architectures"])
                if (
                    getattr(generator.config.security, "trust_remote_code", False)
                    and "auto_map" in meta_cfg
                    and saved_cfg.get("auto_map") != meta_cfg["auto_map"]
                ):
                    saved_cfg["auto_map"] = meta_cfg["auto_map"]
                    _stripped = True
                    logger.info("Aligned auto_map to original (trust_remote_code enabled).")
            except Exception as e:
                logger.warning("Failed to align model_type/architectures: %s", e)

        if generator.shrink_config and isinstance(saved_cfg.get("quantization_config"), dict):
            del saved_cfg["quantization_config"]
            _stripped = True

        for sub_key in ("text_config", "vision_config",
                        "encoder_config", "decoder_config",
                        "audio_config"):
            if not isinstance(saved_cfg.get(sub_key), dict):
                continue
            # For VLMs, keep vision_config if it looks like a valid
            # PretrainedConfig dict (has model_type or hidden_size).
            if sub_key == "vision_config" and _is_vlm:
                vc = saved_cfg[sub_key]
                if isinstance(vc, dict) and (
                    vc.get("model_type") or vc.get("hidden_size")
                ):
                    # Looks like a proper config dict — keep it
                    logger.info(
                        "Preserving vision_config dict in config.json for VLM model."
                    )
                    continue
            del saved_cfg[sub_key]
            _stripped = True
            logger.info("Stripped raw dict sub-config '%s' from saved config.json", sub_key)
        if _stripped:
            with open(saved_cfg_path, "w") as f:
                json.dump(saved_cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Post-process config strip failed: %s", e)

    if generator.shrink_config:
        # ── Reconcile config.json with actual generated weight shapes ──
        # The shrink process + model initialisation (which may fall back to
        # LlamaForCausalLM) can produce weights whose shapes do NOT match
        # the config's dimension fields (e.g. intermediate_size is 512 in
        # config but the actual gate_proj has shape [11008, 256]).
        # We scan the actual saved weights and patch config.json so that
        # `from_pretrained()` does not hit size-mismatch errors.
        reconcile_config_with_weights(generator, saved_cfg_path)

def reconcile_config_with_weights(generator, cfg_path: str) -> None:
    """Patch config.json so that its dimension fields match actual weight shapes.

    After shrink + model initialisation (which may fallback to Llama, etc.),
    the config may declare intermediate_size=512 while the generated weights
    actually have gate_proj shape [11008, 256].  This mismatch causes
    ``from_pretrained()`` to fail with RuntimeError (size mismatch).

    This method:
    1. Reads the first .bin shard to discover actual weight shapes.
    2. Infers vocab_size, hidden_size, intermediate_size, num_hidden_layers
       from the weight shapes.
    3. Patches config.json with the inferred values.
    """
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:
        logger.debug("Failed to load config for reconciliation: %s", e)
        return

    # Find weight shapes from the first few shards
    shapes: Dict[str, tuple] = {}
    idx_path = os.path.join(generator.output_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(idx_path):
        return
    try:
        with open(idx_path) as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
    except Exception as e:
        logger.debug("Failed to read index for reconciliation: %s", e)
        return

    # Load a representative sample of shards to get key shapes
    shards_to_read = set()
    for key in weight_map:
        if any(p in key for p in ("embed_tokens", "gate_proj", "q_proj", "lm_head")):
            shards_to_read.add(weight_map[key])
        if len(shards_to_read) >= 4:
            break

    for shard_name in shards_to_read:
        shard_path = os.path.join(generator.output_dir, shard_name)
        if os.path.exists(shard_path):
            try:
                try:
                    d = torch.load(shard_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    logger.warning(
                        "Skipping shape extraction for %s: torch.load(weights_only=True) failed "
                        "and unsafe pickle fallback is disabled. Convert this shard to safetensors "
                        "or regenerate it. Error: %s",
                        shard_path,
                        e,
                    )
                    continue
                for k, v in d.items():
                    shapes[k] = tuple(v.shape)
            except Exception as e:
                logger.debug("Failed to load shard %s for shape extraction: %s", shard_path, e)

    if not shapes:
        logger.debug("No weight shapes found — skipping config reconciliation.")
        return

    # Infer dimensions from weight shapes
    patched = False
    diff: Dict[str, Dict[str, Any]] = {}

    def _set(k: str, v):
        nonlocal patched
        if cfg.get(k) != v:
            diff[k] = {"before": cfg.get(k), "after": v}
            cfg[k] = v
            patched = True

    # vocab_size from embed_tokens.weight [vocab, hidden]
    for k, s in shapes.items():
        if "embed_tokens" in k and len(s) == 2:
            _set("vocab_size", s[0])
            _set("hidden_size", s[1])
            break

    # intermediate_size from gate_proj.weight [inter, hidden]
    for k, s in shapes.items():
        if "gate_proj" in k and len(s) == 2:
            _set("intermediate_size", s[0])
            break

    # num_hidden_layers from highest layer index
    max_layer = -1
    for k in weight_map:
        import re as _re
        m = _re.search(r"layers\.(\d+)\.", k)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    if max_layer >= 0:
        inferred_layers = max_layer + 1
        _set("num_hidden_layers", inferred_layers)

    # num_attention_heads / num_key_value_heads from q_proj/k_proj shapes
    cfg.get("hidden_size", 256)
    for k, s in shapes.items():
        if "self_attn.q_proj" in k and len(s) == 2:
            # q_proj: [num_heads * head_dim, hidden]
            head_dim = cfg.get("head_dim")
            if head_dim and head_dim > 0:
                n_heads = s[0] // head_dim
                if n_heads > 0:
                    _set("num_attention_heads", n_heads)
            break

    if patched:
        try:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            logger.info(
                "Reconciled config.json with actual weight shapes: "
                "vocab=%s, hidden=%s, inter=%s, layers=%s",
                cfg.get("vocab_size"), cfg.get("hidden_size"),
                cfg.get("intermediate_size"), cfg.get("num_hidden_layers"),
            )
        except Exception as e:
            logger.warning("Config reconciliation write failed: %s", e)
    else:
        logger.debug("Config already consistent with weight shapes — no patch needed.")

    diff_path = os.path.join(generator.output_dir, "vitriol-reconcile.json")
    try:
        with open(diff_path, "w") as f:
            json.dump({"patched": bool(patched), "diff": diff}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write reconcile diff: %s", e)

def save_tokenizer(generator) -> None:
    logger.info("Saving tokenizer…")
    try:
        from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

        local_files_only = bool(
            getattr(generator.config.security, "local_files_only", False)
            or not getattr(generator.config.security, "allow_network", True)
        )
        tok = hf_load_tokenizer(
            generator.model_id,
            security={
                "trust_remote_code": generator.config.security.trust_remote_code,
                "allow_network": not local_files_only,
                "local_files_only": local_files_only,
            },
        )
        tok.save_pretrained(generator.output_dir)
    except Exception as e:
        logger.warning("Tokenizer save failed: %s", e)
        copy_tokenizer_assets(generator)

def copy_tokenizer_assets(generator) -> None:
    local_files_only = bool(
        getattr(generator.config.security, "local_files_only", False)
        or not getattr(generator.config.security, "allow_network", True)
    )
    if os.path.isdir(generator.model_id):
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
        ):
            src = os.path.join(generator.model_id, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(generator.output_dir, name))
        return
    if local_files_only:
        return
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_files = set(list_repo_files(generator.model_id))
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "sentencepiece.bpe.model",
        ):
            if name not in repo_files:
                continue
            src = hf_hub_download(
                repo_id=generator.model_id,
                filename=name,
                local_files_only=local_files_only,
            )
            shutil.copy2(src, os.path.join(generator.output_dir, name))
    except Exception as copy_exc:
        logger.warning("Tokenizer asset fallback failed: %s", copy_exc)

# ──────────────────────────────────────────────────────────────────────
# README
# ──────────────────────────────────────────────────────────────────────

def add_readme_metadata(generator) -> None:
    has_viz   = generate_visualization(generator, 
        os.path.join(generator.output_dir, "architecture.png"))
    viz_block = (
        "## 🔍 Architecture Visualization\n\n"
        "![Block Diagram](architecture.png)\n"
        "![Detailed Architecture](architecture_detail.png)\n\n"
        "*Interactive: [architecture.html](architecture.html)*\n"
    ) if has_viz else "## 🔍 Architecture Visualization\n*(Unavailable)*\n"

    # [B6 fix] datetime instead of subprocess call
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    content = f"""---
library_name: transformers
tags:
- vitriol
- minimal-weights
- architectural-test
---
# Vitriol-Compressed: {generator.model_id}

⚠️ **IMPORTANT WARNING**: This model contains **minimal/dummy weights** generated by
[Vitriol](https://github.com/isLinXu/Vitriol). It is **NOT** intended for inference.

## 📊 Model Details
| Field | Value |
|-------|-------|
| Original | `{generator.model_id}` |
| Strategy | `{generator.config.strategy}` |
| Shrunk | `{generator.shrink_config}` |
| Generated | {ts} |

## 🛠️ How to Use
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

path = "{generator.output_dir}"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
trust_remote_code = False  # set True only for trusted custom-code model repos
cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
for k in ("text_config", "vision_config", "encoder_config", "decoder_config"):
    v = getattr(cfg, k, None)
    if isinstance(v, dict):
        setattr(cfg, k, PretrainedConfig.from_dict(v))
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code, config=cfg)
out = model.generate(**tok("hello", return_tensors="pt"), max_new_tokens=8)
print(tok.decode(out[0]))
```

{viz_block}

## 🔗 About Vitriol
- **GitHub**: https://github.com/isLinXu/Vitriol

---
*Generated with ❤️ by Vitriol*
"""
    readme = os.path.join(generator.output_dir, "README.md")
    with open(readme, "w") as f:
        f.write(content)
    logger.info("README.md written.")

# ──────────────────────────────────────────────────────────────────────
# Architecture visualisation
# ──────────────────────────────────────────────────────────────────────

def generate_visualization(generator, viz_path: str) -> bool:
    try:
        from ..arch_viz.visualizer import ArchitectureVisualizer
    except ImportError:
        logger.debug("arch_viz unavailable; skipping visualization.")
        return False

    cfg_json    = os.path.join(generator.output_dir, "config.json")
    cfg_meta    = os.path.join(generator.output_dir, "meta-config.json")
    cfg_meta_legacy = os.path.join(generator.output_dir, "config_meta.json")
    cfg_shrunk  = os.path.join(generator.output_dir, "config_shrunk.json")
    temp_swap   = False

    # Resolve meta config path (prefer meta-config.json, fallback config_meta.json)
    meta_path = cfg_meta if os.path.exists(cfg_meta) else (
        cfg_meta_legacy if os.path.exists(cfg_meta_legacy) else None
    )

    try:
        # Strategy A: meta config exists → swap to expose original for visualization
        if meta_path:
            os.rename(cfg_json, cfg_shrunk)
            shutil.copy2(meta_path, cfg_json)
            temp_swap = True

        # Strategy B: shrink mode, no meta → fetch from Hub
        elif generator.shrink_config:
            try:
                from .config_loader import load_hf_config

                orig = load_hf_config(generator)
                generator._patch_model_config(orig)
                with tempfile.TemporaryDirectory() as tmp:
                    orig.save_pretrained(tmp)
                    os.rename(cfg_json, cfg_shrunk)
                    shutil.move(os.path.join(tmp, "config.json"), cfg_json)
                temp_swap = True
                logger.info("Fetched original config for visualization.")
            except Exception as e:
                logger.warning("Config fetch for viz failed: %s. Using shrunk.", e)

        viz = ArchitectureVisualizer(
            generator.output_dir,
            trust_remote_code=bool(getattr(generator.config.security, "trust_remote_code", False)),
            local_files_only=True,
        )
        viz.generate_block_diagram(os.path.join(generator.output_dir, "architecture.png"))
        viz.generate_detailed_diagram(os.path.join(generator.output_dir, "architecture_detail.png"))
        viz.generate_interactive_html(os.path.join(generator.output_dir, "architecture.html"))
        return True

    except Exception as e:
        logger.warning("Visualization failed: %s", e)
        return False

    finally:
        # Restore: shrunk → config.json; meta-config.json stays as-is
        if temp_swap and os.path.exists(cfg_shrunk):
            try:
                if os.path.exists(cfg_json):
                    os.remove(cfg_json)
                # Restore shrunk config as the active config.json
                os.rename(cfg_shrunk, cfg_json)
            except Exception as e:
                logger.error("Config restore in finally block failed: %s", e)