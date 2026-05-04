"""
Unified Configuration Management System for Vitriol.

Provides hierarchical configuration with:
- Environment-specific settings
- User preferences
- Project-level overrides
- Runtime modifications
- Validation and type checking
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging

from vitriol.version import __version__

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class GenerationDefaults:
    """Default generation settings (app/experiment-level; not part of the generator public API)."""
    default_strategy: str = "compact"
    default_dtype: str = "bfloat16"
    max_shard_size: str = "5GB"
    parallel_workers: int = 4
    use_memory_mapping: bool = True
    compression_level: int = 6
    verify_checksums: bool = True
    cache_dir: str = "~/.cache/vitriol"
    temp_dir: str = "/tmp/vitriol"


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    default_algorithm: str = "evolutionary"
    population_size: int = 20
    n_iterations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    early_stopping_patience: int = 10
    checkpoint_interval: int = 10
    use_rl_agent: bool = True
    rl_learning_rate: float = 3e-4


@dataclass
class SystemConfig:
    """System-level configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    max_memory_gb: float = 32.0
    max_disk_gb: float = 100.0
    gpu_enabled: bool = True
    gpu_memory_fraction: float = 0.9
    cpu_affinity: Optional[List[int]] = None
    nice_level: int = 0


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    enable_encryption: bool = False
    encryption_key_path: Optional[str] = None
    verify_signatures: bool = True
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost"])
    api_key_required: bool = False
    api_keys: List[str] = field(default_factory=list)
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600


@dataclass
class VitriolConfig:
    """Main Vitriol configuration."""
    version: str = __version__
    environment: str = "development"
    
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    nas: NASConfig = field(default_factory=NASConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Custom user settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Features:
        - Hierarchical config (default -> file -> env -> runtime)
        - Hot reload support
        - Validation and type checking
        - Environment-specific configs
        - Secrets management
    
    Example:
        >>> config = ConfigManager()
        >>> config.load_from_file("config.yaml")
        >>> config.get("generation.default_strategy")
        'compact'
    """
    
    # Config file search paths
    CONFIG_PATHS = [
        "vitriol.yaml",
        "vitriol.json",
        "~/.config/vitriol/config.yaml",
        "~/.vitriol/config.yaml",
        "/etc/vitriol/config.yaml",
    ]
    
    def __init__(self):
        self._config = VitriolConfig()
        self._loaded_files: List[str] = []
        self._watchers: List[callable] = []
        
        # Load default config
        self._load_defaults()
        
        # Auto-discover config files
        self._auto_load()
    
    def _load_defaults(self):
        """Load default configuration."""
        self._config = VitriolConfig()
    
    def _auto_load(self):
        """Automatically load config from standard locations."""
        for path in self.CONFIG_PATHS:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                self.load_from_file(str(expanded_path))
                break
        
        # Load from environment
        self._load_from_env()
    
    def load_from_file(self, path: str) -> "ConfigManager":
        """
        Load configuration from file.
        
        Args:
            path: Path to config file (yaml or json)
            
        Returns:
            Self for chaining
        """
        path = Path(path).expanduser()
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return self
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._merge_config(data)
            self._loaded_files.append(str(path))
            logger.info(f"Loaded config from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
        
        return self
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "VITRIOL_ENV": "environment",
            "VITRIOL_LOG_LEVEL": "system.log_level",
            "VITRIOL_CACHE_DIR": "generation.cache_dir",
            "VITRIOL_GPU_ENABLED": "system.gpu_enabled",
            "VITRIOL_DEFAULT_STRATEGY": "generation.default_strategy",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                
                self.set(config_key, value)
    
    def _merge_config(self, data: Dict[str, Any]):
        """Merge dictionary into current config."""
        for key, value in data.items():
            if hasattr(self._config, key):
                if isinstance(value, dict):
                    # Merge nested dataclass
                    current = getattr(self._config, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(current, sub_key):
                            setattr(current, sub_key, sub_value)
                else:
                    setattr(self._config, key, value)
            else:
                # Store in custom
                self._config.custom[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Dot-separated key (e.g., "generation.default_strategy")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> "ConfigManager":
        """
        Set configuration value.
        
        Args:
            key: Dot-separated key
            value: Value to set
            
        Returns:
            Self for chaining
        """
        keys = key.split('.')
        target = self._config
        
        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            elif isinstance(target, dict):
                if k not in target:
                    target[k] = {}
                target = target[k]
            else:
                return self
        
        final_key = keys[-1]
        if hasattr(target, final_key):
            setattr(target, final_key, value)
        elif isinstance(target, dict):
            target[final_key] = value
        
        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.debug("Config watcher callback failed for key %s: %s", key, e)
        
        return self
    
    def save_to_file(self, path: str, format: str = "yaml"):
        """
        Save current configuration to file.
        
        Args:
            path: Output file path
            format: 'yaml' or 'json'
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved config to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "version": self._config.version,
            "environment": self._config.environment,
            "generation": asdict(self._config.generation),
            "nas": asdict(self._config.nas),
            "system": asdict(self._config.system),
            "security": asdict(self._config.security),
            "custom": self._config.custom,
        }
    
    def watch(self, callback: callable):
        """Register a config change watcher."""
        self._watchers.append(callback)
    
    def unwatch(self, callback: callable):
        """Unregister a watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def get_environment(self) -> ConfigEnvironment:
        """Get current environment."""
        return ConfigEnvironment(self._config.environment)
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self._config.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self._config.environment == "development"


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def init_config(path: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration."""
    global _config_instance
    _config_instance = ConfigManager()
    
    if path:
        _config_instance.load_from_file(path)
    
    return _config_instance
