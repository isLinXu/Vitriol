from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from ..utils.hf_loading import build_config_object, load_config_or_raw

logger = logging.getLogger(__name__)

class ConfigParser:
    """Parses model configuration from HuggingFace ID or local path.

    When loading from a local directory, prioritizes meta-config.json
    (the original unmodified HF config) over config.json (which may
    be shrunk or modified by the generation process).
    """

    @staticmethod
    def load_config(model_id_or_path: str, trust_remote_code: bool = False, local_files_only: bool = False) -> Any:
        path = Path(model_id_or_path)

        try:
            from ..adapters.registry import AdapterRegistry
            AdapterRegistry._load_builtin_adapters()
            for adapter_cls in AdapterRegistry._adapters:
                try:
                    adapter_cls().register_classes()
                except Exception:
                    logger.debug("Failed to register adapter classes")
        except Exception:
            logger.debug("Failed to load builtin adapters")

        # For local directories: prefer meta-config.json → config_meta.json → config.json
        if path.is_dir():
            for meta_name in ('meta-config.json', 'config_meta.json'):
                meta_path = path / meta_name
                if meta_path.exists():
                    try:
                        logger.info(f"Loading original config from {meta_name}")
                        # Read the meta config and try to parse it via AutoConfig
                        meta_dict = json.loads(meta_path.read_text())
                        # Write it temporarily as config.json for AutoConfig
                        with tempfile.TemporaryDirectory() as tmp:
                            tmp_config = Path(tmp) / "config.json"
                            tmp_config.write_text(json.dumps(meta_dict, indent=2))
                            try:
                                return load_config_or_raw(
                                    tmp,
                                    security={
                                        "trust_remote_code": trust_remote_code,
                                        "allow_network": False,
                                        "local_files_only": True,
                                    },
                                )
                            except Exception as exc:
                                logger.debug("Config load from %s failed, falling back: %s", meta_name, exc)
                                return build_config_object(meta_dict)
                    except Exception as e:
                        logger.warning(f"Failed to load {meta_name}, falling back: {e}")

        # Default path: load from model_id or local config.json
        try:
            return load_config_or_raw(
                model_id_or_path,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": not local_files_only,
                    "local_files_only": local_files_only,
                },
            )
        except Exception as e:
            if path.is_dir():
                cfg_path = path / "config.json"
                if cfg_path.exists():
                    try:
                        return build_config_object(json.loads(cfg_path.read_text(encoding="utf-8")))
                    except Exception:
                        logger.debug("Failed to load config.json from local path")
            logger.error(f"Failed to load config for {model_id_or_path}: {e}")
            raise e
