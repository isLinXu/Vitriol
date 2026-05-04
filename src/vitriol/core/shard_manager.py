"""
Shard manager for splitting large models into multiple files.

This module handles the logic for splitting model weights into
multiple shards to handle large models that don't fit in memory.
"""

import logging
from typing import Dict, Iterator, List, Tuple

logger = logging.getLogger(__name__)


class ShardManager:
    """
    Manage model sharding for large models.
    
    This class handles:
    - Planning how to split parameters into shards
    - Naming shards consistently
    - Tracking shard sizes
    
    Example:
        >>> manager = ShardManager(max_shard_size="5GB")
        >>> for shard_name, params in manager.plan_shards(param_names, param_sizes):
        ...     print(f"{shard_name}: {len(params)} parameters")
    """
    
    def __init__(self, max_shard_size: str = "5GB"):
        """
        Initialize shard manager.
        
        Args:
            max_shard_size: Maximum size per shard (e.g., "5GB", "500MB")
        """
        self.max_shard_size = self._parse_size(max_shard_size)
        logger.info(f"ShardManager initialized with max_shard_size={max_shard_size}")
    
    def plan_shards(
        self,
        param_names: List[str],
        param_sizes: Dict[str, int]
    ) -> Iterator[Tuple[str, Dict[str, int]]]:
        """
        Plan how to split parameters into shards.
        
        Args:
            param_names: List of parameter names in order
            param_sizes: Dict mapping parameter names to their sizes in bytes
        
        Yields:
            Tuple of (shard_filename, dict of param_name -> size)
        """
        current_shard: Dict[str, int] = {}
        current_size = 0
        shard_idx = 0
        
        for name in param_names:
            size = param_sizes.get(name, 0)
            
            # If adding this param would exceed max size, yield current shard
            if current_size + size > self.max_shard_size and current_shard:
                shard_name = self._format_shard_name(shard_idx)
                logger.debug(
                    f"Yielding shard {shard_idx}: {len(current_shard)} params, "
                    f"{current_size / 1e9:.2f} GB"
                )
                yield shard_name, current_shard
                
                shard_idx += 1
                current_shard = {}
                current_size = 0
            
            current_shard[name] = size
            current_size += size
        
        # Yield final shard
        if current_shard:
            shard_name = self._format_shard_name(shard_idx)
            logger.debug(
                f"Yielding final shard {shard_idx}: {len(current_shard)} params, "
                f"{current_size / 1e9:.2f} GB"
            )
            yield shard_name, current_shard
    
    def plan_shards_from_model(
        self,
        model,
        dtype_size: int = 2  # bfloat16 = 2 bytes
    ) -> Iterator[Tuple[str, Dict[str, int]]]:
        """
        Plan shards directly from a model instance.
        
        Args:
            model: PyTorch model
            dtype_size: Size of each element in bytes
        
        Yields:
            Tuple of (shard_filename, dict of param_name -> size)
        """
        param_names = []
        param_sizes = {}
        
        for name, param in model.named_parameters():
            param_names.append(name)
            param_sizes[name] = param.numel() * dtype_size
        
        return self.plan_shards(param_names, param_sizes)
    
    def _format_shard_name(self, shard_idx: int, format: str = "pytorch") -> str:
        """
        Format shard filename.
        
        Args:
            shard_idx: Shard index
            format: Storage format ("pytorch" or "safetensors")
        
        Returns:
            Formatted shard filename
        """
        if format == "safetensors":
            return f"model-{shard_idx+1:05d}-of-XXXXX.safetensors"
        else:
            return f"pytorch_model-{shard_idx+1:05d}-of-XXXXX.bin"
    
    def estimate_total_shards(
        self,
        param_names: List[str],
        param_sizes: Dict[str, int]
    ) -> int:
        """
        Estimate total number of shards.
        
        Args:
            param_names: List of parameter names
            param_sizes: Dict mapping parameter names to sizes
        
        Returns:
            Estimated number of shards
        """
        count = 0
        for _ in self.plan_shards(param_names, param_sizes):
            count += 1
        return count
    
    @staticmethod
    def _parse_size(size_str: str) -> int:
        """
        Parse size string to bytes.
        
        Args:
            size_str: Size string like "5GB", "500MB", "1024"
        
        Returns:
            Size in bytes
        
        Raises:
            ValueError: If size string is invalid
        """
        size_str = size_str.strip().upper()
        
        # Multipliers
        multipliers = {
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4,
        }
        
        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                try:
                    num = float(size_str[:-len(suffix)])
                    return int(num * mult)
                except ValueError:
                    raise ValueError(f"Invalid size string: {size_str}")
        
        # No suffix - assume bytes
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Invalid size string: {size_str}")
    
    def get_shard_index_map(
        self,
        param_names: List[str],
        param_sizes: Dict[str, int]
    ) -> Dict[str, str]:
        """
        Get a map from parameter names to shard filenames.
        
        Args:
            param_names: List of parameter names
            param_sizes: Dict mapping parameter names to sizes
        
        Returns:
            Dict mapping parameter name to shard filename
        """
        index_map = {}
        
        for shard_name, params in self.plan_shards(param_names, param_sizes):
            for param_name in params:
                index_map[param_name] = shard_name
        
        return index_map
