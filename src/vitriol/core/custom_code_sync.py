"""Custom HuggingFace repo code sync for generated model directories."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Set

from ._generator_utils import (
    _CUSTOM_CODE_DEFAULT_MAX_FILES,
    _CUSTOM_CODE_MAX_FILES_ENV,
    custom_code_file_matches_auto_map,
    custom_repo_file_size_limit,
    extract_auto_map_modules,
    is_allowed_custom_repo_file,
    positive_int_env,
)

if TYPE_CHECKING:
    from .generator import MinimalWeightGenerator

logger = logging.getLogger(__name__)


def custom_code_modules_from_saved_config(generator: "MinimalWeightGenerator") -> Set[str] | None:
    """Return Python modules referenced by saved ``auto_map`` metadata, if present."""
    for config_name in ("meta-config.json", "config.json"):
        config_path = os.path.join(generator.output_dir, config_name)
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Could not inspect %s for auto_map custom code: %s", config_path, e)
            continue
        modules = extract_auto_map_modules(config_data.get("auto_map"))
        if modules:
            return modules
    return None


def copy_custom_code_files(generator: "MinimalWeightGenerator") -> None:
    """Download/copy custom Python files when trust_remote_code is enabled."""
    try:
        if not getattr(generator.config.security, "trust_remote_code", False):
            return
        if os.path.isdir(generator.model_id):
            return
        if bool(
            getattr(generator.config.security, "local_files_only", False)
            or not getattr(generator.config.security, "allow_network", True)
        ):
            return
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id = generator.model_id
        files = list(list_repo_files(repo_id))
        auto_map_modules = custom_code_modules_from_saved_config(generator)
        target_files = []
        for file_name in files:
            if is_allowed_custom_repo_file(file_name):
                if (
                    file_name.lower().endswith(".py")
                    and auto_map_modules is not None
                    and not custom_code_file_matches_auto_map(file_name, auto_map_modules)
                ):
                    logger.warning("Skipping custom Python file not referenced by auto_map: %s", file_name)
                    continue
                target_files.append(file_name)
            elif file_name.endswith(".py"):
                logger.warning("Skipping non-whitelisted custom Python file: %s", file_name)

        if not target_files:
            return
        max_files = positive_int_env(_CUSTOM_CODE_MAX_FILES_ENV, _CUSTOM_CODE_DEFAULT_MAX_FILES)
        if len(target_files) > max_files:
            logger.warning(
                "Refusing to sync all custom-code files: %d allowed files exceeds limit %d; "
                "syncing the first %d only.",
                len(target_files),
                max_files,
                max_files,
            )
            target_files = target_files[:max_files]

        logger.info("Downloading %d custom code/asset files for trust_remote_code...", len(target_files))
        real_root = os.path.realpath(generator.output_dir)
        for file_name in target_files:
            try:
                if os.path.isabs(file_name) or ".." in file_name.replace("\\", "/").split("/"):
                    logger.warning(
                        "Refusing suspicious custom-code filename (path traversal): %s",
                        file_name,
                    )
                    continue
                file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
                file_size = os.path.getsize(file_path)
                max_file_size = custom_repo_file_size_limit(file_name)
                if file_size > max_file_size:
                    logger.warning(
                        "Skipping oversized custom-code file %s (%d bytes > %d bytes)",
                        file_name,
                        file_size,
                        max_file_size,
                    )
                    continue
                dest_path = os.path.join(generator.output_dir, file_name)
                real_dest = os.path.realpath(dest_path)
                if not (real_dest == real_root or real_dest.startswith(real_root + os.sep)):
                    logger.warning(
                        "Refusing custom-code filename that escapes output_dir: %s",
                        file_name,
                    )
                    continue
                dest_dir = os.path.dirname(dest_path)
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                logger.debug("Copied custom code file: %s", file_name)
            except Exception as e:
                logger.warning("Failed to copy %s: %s", file_name, e)
    except Exception as e:
        logger.warning("Could not sync custom code files: %s", e)
