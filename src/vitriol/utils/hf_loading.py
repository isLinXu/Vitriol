"""
HuggingFace loading facade.

Goals:
1) Normalize trust_remote_code / allow_network / local_files_only semantics.
2) Provide a single source of truth for transformers Auto* loading (auditable/testable).
3) Avoid importing transformers eagerly; use lazy imports inside functions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from vitriol.config.manager import SecurityOptions

logger = logging.getLogger(__name__)


class RawConfig:
    """Lightweight attribute-style config used when transformers is unavailable."""

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", {})
        for key, value in dict(data).items():
            self._data[key] = self._wrap(value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawConfig":
        return cls(dict(data))

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return RawConfig(value)
        if isinstance(value, list):
            return [RawConfig._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(RawConfig._wrap(item) for item in value)
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, RawConfig):
            return value.to_dict()
        if isinstance(value, list):
            return [RawConfig._unwrap(item) for item in value]
        if isinstance(value, tuple):
            return [RawConfig._unwrap(item) for item in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {key: self._unwrap(value) for key, value in self._data.items()}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = self._wrap(value)

    def __repr__(self) -> str:  # pragma: no cover
        keys = ", ".join(sorted(self._data))
        return f"RawConfig({keys})"


def _coerce_security(security: Optional[SecurityOptions | Dict[str, Any]]) -> SecurityOptions:
    if security is None:
        default_trust_remote_code = True
        default_allow_network = True
        default_local_files_only = False
        return SecurityOptions(
            trust_remote_code=default_trust_remote_code,
            allow_network=default_allow_network,
            local_files_only=default_local_files_only,
        )

    if isinstance(security, SecurityOptions):
        return security

    if is_dataclass(security):
        data = asdict(security)
    else:
        data = dict(security)

    return SecurityOptions(
        trust_remote_code=bool(data.get("trust_remote_code", True)),
        allow_network=bool(data.get("allow_network", True)),
        local_files_only=bool(data.get("local_files_only", False)),
    )


def hf_kwargs(
    security: Optional[SecurityOptions | Dict[str, Any]] = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert SecurityOptions into kwargs for transformers from_pretrained().

    Rules:
    - allow_network=False => force local_files_only=True (prevents accidental network access)
    - trust_remote_code is passed through as-is (explicitly controlled by the caller)
    """
    # P3: single source of truth. Do not re-implement offline/network semantics in hf_kwargs;
    # fully delegate to the SecurityContext resolver (with provenance + invariants).
    sec = _coerce_security(security)
    from vitriol.security.context import resolve_security_context

    ctx = resolve_security_context(base=sec, explicit={})
    local_files_only = bool(ctx.local_files_only)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(ctx.trust_remote_code),
        "local_files_only": local_files_only,
    }
    if extra:
        kwargs.update(extra)
    return kwargs


def load_config(model_id: str, *, security: Optional[SecurityOptions | Dict[str, Any]] = None, **kwargs: Any):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id, **hf_kwargs(security, extra=kwargs))


def _is_unknown_model_type_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "model type" in message and ("not recognize" in message or "unknown" in message)


def load_raw_config_dict(
    model_id: str,
    *,
    security: Optional[SecurityOptions | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = Path(model_id)
    if path.is_dir():
        for name in ("meta-config.json", "config_meta.json", "config.json"):
            candidate = path / name
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No config file found under {path}")

    from transformers.utils import cached_file

    kwargs = hf_kwargs(security)
    config_path = cached_file(
        model_id,
        "config.json",
        _raise_exceptions_for_missing_entries=False,
        local_files_only=bool(kwargs.get("local_files_only", False)),
    )
    if not config_path:
        raise FileNotFoundError(f"config.json not found for {model_id}")
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def build_config_object(config_dict: Dict[str, Any]):
    try:
        from transformers import PretrainedConfig
    except Exception:
        return RawConfig.from_dict(config_dict)
    try:
        return PretrainedConfig.from_dict(config_dict)
    except Exception:
        return RawConfig.from_dict(config_dict)


def load_config_or_raw(
    model_id: str,
    *,
    security: Optional[SecurityOptions | Dict[str, Any]] = None,
    **kwargs: Any,
):
    try:
        return load_config(model_id, security=security, **kwargs)
    except Exception as exc:
        path = Path(model_id)
        if not (_is_unknown_model_type_error(exc) or (path.is_dir() and isinstance(exc, ModuleNotFoundError))):
            raise
        config_dict = load_raw_config_dict(model_id, security=security)
        return build_config_object(config_dict)


def _maybe_patch_dynamic_modules(model_id: str, *, security: Optional[SecurityOptions | Dict[str, Any]] = None) -> None:
    try:
        path = Path(model_id)
        if not path.exists():
            return

        kwargs = hf_kwargs(security)
        if not bool(kwargs.get("trust_remote_code", False)):
            return

        config_dict = load_raw_config_dict(model_id, security=security)
        auto_map = config_dict.get("auto_map")
        if not isinstance(auto_map, dict) or not auto_map:
            return

        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import sys

        from vitriol.patches import patch_remote_module

        for key in ("AutoModelForCausalLM", "AutoModel", "AutoConfig"):
            target = auto_map.get(key)
            if not isinstance(target, str) or "." not in target:
                continue
            try:
                cls = get_class_from_dynamic_module(target, model_id)
                mod = sys.modules.get(cls.__module__)
                if mod is not None:
                    patch_remote_module(mod)
                    return
            except Exception:
                continue
    except Exception:
        return


def _load_config_with_patches(model_id: str, *, security: Optional[SecurityOptions | Dict[str, Any]] = None):
    from transformers import AutoConfig
    import sys

    cfg = AutoConfig.from_pretrained(model_id, **hf_kwargs(security))
    try:
        from vitriol.patches import patch_remote_module

        mod = sys.modules.get(cfg.__class__.__module__)
        if mod is not None:
            patch_remote_module(mod)
    except Exception:
        logger.debug("Failed to patch remote module for %s", cfg.__class__.__module__)
    return cfg


def load_tokenizer(model_id: str, *, security: Optional[SecurityOptions | Dict[str, Any]] = None, **kwargs: Any):
    from transformers import AutoTokenizer

    try:
        _maybe_patch_dynamic_modules(model_id, security=security)
        return AutoTokenizer.from_pretrained(model_id, **hf_kwargs(security, extra=kwargs))
    except Exception as exc:
        if _is_unknown_model_type_error(exc) or "max_position_embeddings" in str(exc) or "deepseek_v4" in str(exc):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load tokenizer normally due to unsupported model type, falling back to local files without config validation: {exc}")
            
            # When AutoTokenizer fails because it tries to load AutoConfig first and fails on deepseek_v4
            # We can bypass the config validation by just passing the directory if it's local
            if Path(model_id).exists():
                from transformers import PreTrainedTokenizerFast
                tokenizer_file = Path(model_id) / "tokenizer.json"
                if tokenizer_file.exists():
                    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
                # Fallback
                return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        raise


def load_model(model_id: str, *, security: Optional[SecurityOptions | Dict[str, Any]] = None, **kwargs: Any):
    """Generic AutoModel.from_pretrained wrapper."""
    from transformers import AutoModel

    if Path(model_id).exists() and bool(hf_kwargs(security).get("trust_remote_code", False)):
        cfg = _load_config_with_patches(model_id, security=security)
        extra = dict(kwargs)
        extra["config"] = cfg
        return AutoModel.from_pretrained(model_id, **hf_kwargs(security, extra=extra))

    _maybe_patch_dynamic_modules(model_id, security=security)
    return AutoModel.from_pretrained(model_id, **hf_kwargs(security, extra=kwargs))


def load_causallm(
    model_id: str,
    *,
    security: Optional[SecurityOptions | Dict[str, Any]] = None,
    torch_dtype: Any = None,
    device: Any = None,
    **kwargs: Any,
):
    from transformers import AutoModelForCausalLM

    extra = dict(kwargs)
    if torch_dtype is not None:
        extra["torch_dtype"] = torch_dtype

    if Path(model_id).exists() and bool(hf_kwargs(security).get("trust_remote_code", False)):
        cfg = _load_config_with_patches(model_id, security=security)
        extra["config"] = cfg
        model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs(security, extra=extra))
    else:
        _maybe_patch_dynamic_modules(model_id, security=security)
        model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs(security, extra=extra))
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_from_config(config: Any, *, security: Optional[SecurityOptions | Dict[str, Any]] = None, **kwargs: Any):
    from transformers import AutoModel

    extra = hf_kwargs(security, extra=kwargs)
    extra.pop("local_files_only", None)
    return AutoModel.from_config(config, **extra)


def load_causallm_from_config(
    config: Any, *, security: Optional[SecurityOptions | Dict[str, Any]] = None, device: Any = None, **kwargs: Any
):
    from transformers import AutoModelForCausalLM

    extra = hf_kwargs(security, extra=kwargs)
    extra.pop("local_files_only", None)
    model = AutoModelForCausalLM.from_config(config, **extra)
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model
