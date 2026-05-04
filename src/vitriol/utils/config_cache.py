"""
Configuration cache for avoiding repeated downloads.

This module provides caching for model configurations to avoid
repeated downloads from HuggingFace Hub, especially useful for
users with slow or unstable network connections.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from hashlib import md5

from .hf_loading import build_config_object

logger = logging.getLogger(__name__)


class ConfigCache:
    """
    Cache model configs to avoid repeated downloads.
    
    This class provides a simple file-based cache for model configurations,
    with automatic expiration and cache invalidation.
    
    Example:
        >>> cache = ConfigCache()
        >>> config = cache.get("Qwen/Qwen2.5-7B")
        >>> if config is None:
        ...     config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
        ...     cache.set("Qwen/Qwen2.5-7B", config)
    """
    
    def __init__(
        self,
        cache_dir: str = "~/.cache/vitriol/configs",
        max_age_days: int = 7
    ):
        """
        Initialize config cache.
        
        Args:
            cache_dir: Directory to store cached configs
            max_age_days: Maximum age of cached configs in days
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_days * 86400
        
        logger.info(f"ConfigCache initialized at {self.cache_dir}")
    
    def get(self, model_id: str) -> Optional[Any]:
        """
        Get cached config if exists and not expired.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Cached config or None if not found/expired
        """
        cache_key = self._get_cache_key(model_id)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            logger.debug(f"Cache miss for {model_id}")
            return None
        
        # Check age
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > self.max_age_seconds:
            logger.debug(f"Cache expired for {model_id} (age: {file_age/86400:.1f} days)")
            cache_file.unlink()
            return None
        
        # Load cached config
        try:
            with open(cache_file, "r") as f:
                config_dict = json.load(f)
            
            config = build_config_object(config_dict)
            logger.info(f"Loaded cached config for {model_id}")
            return config
        
        except Exception as e:
            logger.warning(f"Failed to load cached config: {e}")
            return None
    
    def set(self, model_id: str, config: Any) -> None:
        """
        Cache a configuration.
        
        Args:
            model_id: Model identifier
            config: Configuration to cache
        """
        cache_key = self._get_cache_key(model_id)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            elif isinstance(config, dict):
                config_dict = dict(config)
            else:
                raise TypeError(f"Unsupported config type for caching: {type(config)!r}")
            
            with open(cache_file, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Cached config for {model_id}")
        
        except Exception as e:
            logger.warning(f"Failed to cache config: {e}")
    
    def clear(self) -> None:
        """Clear all cached configs."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cleared config cache")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            file_age = current_time - cache_file.stat().st_mtime
            
            if file_age > self.max_age_seconds:
                cache_file.unlink()
                removed += 1
        
        if removed > 0:
            logger.info(f"Removed {removed} expired cache entries")
        
        return removed
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats (size, count, oldest entry)
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        
        if not cache_files:
            return {
                "count": 0,
                "total_size_mb": 0,
                "oldest_age_days": 0
            }
        
        total_size = sum(f.stat().st_size for f in cache_files)
        current_time = time.time()
        oldest_age = max(current_time - f.stat().st_mtime for f in cache_files)
        
        return {
            "count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_age_days": oldest_age / 86400
        }
    
    @staticmethod
    def _get_cache_key(model_id: str) -> str:
        """
        Generate cache key from model ID.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Cache key (MD5 hash)
        """
        # Use MD5 for short, consistent filenames
        return md5(model_id.encode()).hexdigest()


class ModelInfoCache:
    """
    Cache for model metadata and statistics.
    
    Stores additional information about models like parameter counts,
    architecture types, and generation history.
    """
    
    def __init__(self, cache_dir: str = "~/.cache/vitriol/models"):
        """
        Initialize model info cache.
        
        Args:
            cache_dir: Directory to store cached info
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.info_file = self.cache_dir / "model_info.json"
        self._cache: Dict[str, Any] = self._load()
    
    def _load(self) -> Dict:
        """Load cache from disk."""
        if self.info_file.exists():
            try:
                with open(self.info_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        """Save cache to disk."""
        with open(self.info_file, "w") as f:
            json.dump(self._cache, f, indent=2)
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get cached model info.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model info dict or None
        """
        return self._cache.get(model_id)
    
    def set_model_info(
        self,
        model_id: str,
        total_params: int,
        arch_type: str,
        features: list,
        **kwargs
    ):
        """
        Cache model information.
        
        Args:
            model_id: Model identifier
            total_params: Total parameter count
            arch_type: Architecture type (e.g., "decoder-only")
            features: List of features (e.g., ["GQA", "RoPE"])
            **kwargs: Additional info to cache
        """
        self._cache[model_id] = {
            "total_params": total_params,
            "arch_type": arch_type,
            "features": features,
            "last_accessed": time.time(),
            **kwargs
        }
        self._save()
    
    def get_recently_used(self, limit: int = 10) -> list:
        """
        Get recently used models.
        
        Args:
            limit: Maximum number of models to return
        
        Returns:
            List of model IDs sorted by last accessed time
        """
        sorted_models = sorted(
            self._cache.items(),
            key=lambda x: x[1].get("last_accessed", 0),
            reverse=True
        )
        return [model_id for model_id, _ in sorted_models[:limit]]
