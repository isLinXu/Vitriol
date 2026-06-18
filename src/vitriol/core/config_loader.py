"""HuggingFace config loading for :class:`~vitriol.core.generator.MinimalWeightGenerator`."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig
from transformers.utils import cached_file

from ..adapters.registry import AdapterRegistry
from ..utils.exceptions import GenerationError
from ._generator_utils import find_best_alias

if TYPE_CHECKING:
    from .generator import MinimalWeightGenerator

logger = logging.getLogger(__name__)


def load_hf_config(generator: "MinimalWeightGenerator") -> Any:
    """Load and normalise a HuggingFace config for weight generation."""
    local_files_only = bool(
        getattr(generator.config.security, "local_files_only", False)
        or not getattr(generator.config.security, "allow_network", True)
    )
    try:
        AdapterRegistry._load_builtin_adapters()
        for adapter_cls in AdapterRegistry._adapters:
            try:
                adapter_cls().register_classes()
            except Exception as e:
                logger.debug("Adapter %s register_classes failed: %s", adapter_cls, e)
    except Exception as e:
        logger.debug("Adapter pre-registration failed: %s", e)

    try:
        from ..utils.hf_loading import load_config as hf_load_config

        return hf_load_config(
            generator.model_id,
            security={
                "trust_remote_code": generator.config.security.trust_remote_code,
                "allow_network": not local_files_only,
                "local_files_only": local_files_only,
            },
        )
    except Exception as e:
        err_str = str(e).lower()
        is_unknown = "model type" in err_str or "not recognize" in err_str
        if not is_unknown:
            raise GenerationError(f"Config load error: {e}") from e

    logger.warning("Unknown model_type or config load error. Trying adapter/alias fallback…")
    try:
        config_path = cached_file(
            generator.model_id,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            local_files_only=local_files_only,
        )
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError("config.json not found")

        with open(config_path) as f:
            config_dict = json.load(f)
        original_type = config_dict.get("model_type", "unknown")
        logger.info("Original model_type: %s", original_type)

        config_dict_no_type = {k: v for k, v in config_dict.items() if k != "model_type"}

        try:
            cfg = AutoConfig.for_model(original_type, **config_dict_no_type)
            logger.info("Loaded config as original type '%s'", original_type)
            return cfg
        except Exception as e:
            logger.debug("for_model construction failed for '%s': %s", original_type, e)

        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if original_type in CONFIG_MAPPING:
                cfg_cls = CONFIG_MAPPING[original_type]
                logger.info("Direct CONFIG_MAPPING construction: %s", cfg_cls.__name__)
                return cfg_cls(**config_dict_no_type)
        except Exception as e:
            logger.debug("CONFIG_MAPPING construction failed for '%s': %s", original_type, e)

        for alias in find_best_alias(original_type):
            try:
                logger.info("Retrying as alias '%s'…", alias)
                cfg = AutoConfig.for_model(alias, **config_dict_no_type)
                logger.info("Loaded config via alias '%s' → %s", alias, type(cfg).__name__)
                return cfg
            except Exception as fe:
                logger.debug("Alias '%s' failed: %s", alias, fe)

        logger.warning("All typed loaders failed — falling back to raw PretrainedConfig.")
        from transformers import PretrainedConfig

        return PretrainedConfig.from_dict(config_dict)

    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"Failed to load raw config.json: {e}") from e
