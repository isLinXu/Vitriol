
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

logger = logging.getLogger(__name__)

def load_weights(
    model_dir: str,
    pattern: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Load weights from model directory with filtering.

    Args:
        model_dir: Path to model directory
        pattern: Regex pattern to filter layer names (e.g. "layer.0|layer.1")
        limit: Max number of tensors to load

    Returns:
        Dictionary mapping parameter names to tensors
    """
    model_path = Path(model_dir)
    weights = {}

    if not model_path.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return weights

    # List weight files
    safetensor_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    bin_files = sorted([f for f in os.listdir(model_path) if f.endswith(".bin")])

    files = []
    is_safetensors = False

    if safetensor_files:
        files = safetensor_files
        is_safetensors = True
    elif bin_files:
        files = bin_files
    else:
        logger.warning(f"No weight files (.safetensors or .bin) found in {model_dir}")
        return weights

    loaded_count = 0
    regex = re.compile(pattern) if pattern else None

    for file_name in files:
        file_path = model_path / file_name
        try:
            shard_weights = {}
            if is_safetensors:
                if load_file is None:
                    logger.error("safetensors not installed")
                    return weights
                shard_weights = load_file(file_path)
            else:
                shard_weights = torch.load(file_path, map_location="cpu", weights_only=True)

            for name, tensor in shard_weights.items():
                if regex and not regex.search(name):
                    continue

                weights[name] = tensor
                loaded_count += 1

                if limit and loaded_count >= limit:
                    return weights

        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")

    return weights
