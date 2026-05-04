"""
Model Registry and Version Management.

Provides:
- Model version control
- Metadata management
- Storage backends
- Search and discovery
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    created_at: str
    description: str
    tags: List[str]
    metadata: Dict[str, Any]
    files: List[str]
    size_bytes: int
    checksum: str


@dataclass
class ModelEntry:
    """Registry entry for a model."""
    id: str
    name: str
    description: str
    author: str
    created_at: str
    updated_at: str
    versions: Dict[str, ModelVersion]
    tags: List[str]
    stats: Dict[str, int]


class ModelRegistry:
    """
    Model registry for version management.
    
    Features:
        - Version control
        - Tag management
        - Metadata storage
        - Search and filtering
    """
    
    def __init__(self, storage_path: str = "~/.vitriol/registry"):
        """
        Initialize registry.
        
        Args:
            storage_path: Path to store registry data
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelEntry] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        index_path = self.storage_path / "index.json"
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                
                for model_id, entry_data in data.items():
                    versions = {
                        v: ModelVersion(**vd)
                        for v, vd in entry_data.get('versions', {}).items()
                    }
                    
                    self.models[model_id] = ModelEntry(
                        id=model_id,
                        name=entry_data['name'],
                        description=entry_data.get('description', ''),
                        author=entry_data.get('author', ''),
                        created_at=entry_data.get('created_at', datetime.now().isoformat()),
                        updated_at=entry_data.get('updated_at', datetime.now().isoformat()),
                        versions=versions,
                        tags=entry_data.get('tags', []),
                        stats=entry_data.get('stats', {'downloads': 0, 'views': 0})
                    )
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        index_path = self.storage_path / "index.json"
        
        data = {}
        for model_id, entry in self.models.items():
            data[model_id] = {
                'name': entry.name,
                'description': entry.description,
                'author': entry.author,
                'created_at': entry.created_at,
                'updated_at': entry.updated_at,
                'versions': {
                    v: asdict(vd)
                    for v, vd in entry.versions.items()
                },
                'tags': entry.tags,
                'stats': entry.stats
            }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model_id: str,
        name: str,
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None
    ) -> ModelEntry:
        """
        Register a new model.
        
        Args:
            model_id: Unique model identifier
            name: Human-readable name
            description: Model description
            author: Model author
            tags: Tags for categorization
            
        Returns:
            Model entry
        """
        now = datetime.now().isoformat()
        
        entry = ModelEntry(
            id=model_id,
            name=name,
            description=description,
            author=author,
            created_at=now,
            updated_at=now,
            versions={},
            tags=tags or [],
            stats={'downloads': 0, 'views': 0}
        )
        
        self.models[model_id] = entry
        self._save_registry()
        
        logger.info(f"Model registered: {model_id}")
        return entry
    
    def publish_version(
        self,
        model_id: str,
        version: str,
        files: List[str],
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[ModelVersion]:
        """
        Publish a new model version.
        
        Args:
            model_id: Model identifier
            version: Version string (e.g., "1.0.0")
            files: List of file paths
            description: Version description
            tags: Version tags
            metadata: Additional metadata
            
        Returns:
            Version info or None if model not found
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return None
        
        entry = self.models[model_id]
        
        # Copy files to registry
        version_dir = self.storage_path / model_id / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        copied_files = []
        total_size = 0
        
        for file_path in files:
            src = Path(file_path)
            if src.exists():
                dst = version_dir / src.name
                shutil.copy2(src, dst)
                copied_files.append(str(dst.relative_to(self.storage_path)))
                total_size += dst.stat().st_size
        
        # Compute checksum
        checksum = self._compute_checksum(copied_files)
        
        # Create version
        version_info = ModelVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or [],
            metadata=metadata or {},
            files=copied_files,
            size_bytes=total_size,
            checksum=checksum
        )
        
        entry.versions[version] = version_info
        entry.updated_at = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"Version {version} published for {model_id}")
        return version_info
    
    def _compute_checksum(self, files: List[str]) -> str:
        """Compute checksum for files."""
        hasher = hashlib.sha256()
        for file_path in sorted(files):
            path = self.storage_path / file_path
            if path.exists():
                with open(path, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()[:16]
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get model entry."""
        return self.models.get(model_id)
    
    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific version."""
        entry = self.models.get(model_id)
        if entry:
            return entry.versions.get(version)
        return None
    
    def list_models(
        self,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None
    ) -> List[ModelEntry]:
        """
        List models with filtering.
        
        Args:
            tags: Filter by tags
            author: Filter by author
            
        Returns:
            List of model entries
        """
        results = list(self.models.values())
        
        if tags:
            results = [
                m for m in results
                if any(t in m.tags for t in tags)
            ]
        
        if author:
            results = [
                m for m in results
                if m.author == author
            ]
        
        return results
    
    def search(self, query: str) -> List[ModelEntry]:
        """
        Search models.
        
        Args:
            query: Search query
            
        Returns:
            Matching models
        """
        query = query.lower()
        results = []
        
        for entry in self.models.values():
            if (query in entry.name.lower() or
                query in entry.description.lower() or
                any(query in t.lower() for t in entry.tags)):
                results.append(entry)
        
        return results
    
    def increment_stat(self, model_id: str, stat: str):
        """Increment model statistic."""
        if model_id in self.models:
            if stat not in self.models[model_id].stats:
                self.models[model_id].stats[stat] = 0
            self.models[model_id].stats[stat] += 1
            self._save_registry()
    
    def delete_model(self, model_id: str):
        """Delete model from registry."""
        if model_id in self.models:
            # Remove files
            model_dir = self.storage_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from index
            del self.models[model_id]
            self._save_registry()
            
            logger.info(f"Model deleted: {model_id}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_models = len(self.models)
        total_versions = sum(len(m.versions) for m in self.models.values())
        
        total_size = 0
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir() and model_dir.name != "index.json":
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        for file in version_dir.rglob("*"):
                            if file.is_file():
                                total_size += file.stat().st_size
        
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "total_size_gb": round(total_size / (1024**3), 2),
            "storage_path": str(self.storage_path)
        }


class ModelStore:
    """
    High-level model storage interface.
    
    Combines registry with storage backends.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
    
    def save_model(
        self,
        model_id: str,
        version: str,
        files: List[str],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save model to store.
        
        Args:
            model_id: Model identifier
            version: Version string
            files: Files to store
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        # Register if not exists
        if model_id not in self.registry.models:
            self.registry.register_model(model_id, model_id)
        
        # Publish version
        version_info = self.registry.publish_version(
            model_id=model_id,
            version=version,
            files=files,
            metadata=metadata
        )
        
        return version_info is not None
    
    def load_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Load model from store.
        
        Args:
            model_id: Model identifier
            version: Version (latest if None)
            
        Returns:
            Model data or None
        """
        entry = self.registry.get_model(model_id)
        if not entry:
            return None
        
        # Get version
        if version is None:
            # Get latest version
            if not entry.versions:
                return None
            version = max(entry.versions.keys())
        
        version_info = entry.versions.get(version)
        if not version_info:
            return None
        
        # Update stats
        self.registry.increment_stat(model_id, 'downloads')
        
        return {
            'model_id': model_id,
            'version': version,
            'files': [
                str(self.registry.storage_path / f)
                for f in version_info.files
            ],
            'metadata': version_info.metadata
        }


# Global instance
_registry: Optional[ModelRegistry] = None
_store: Optional[ModelStore] = None


def get_registry() -> ModelRegistry:
    """Get global registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_store() -> ModelStore:
    """Get global store."""
    global _store
    if _store is None:
        _store = ModelStore(get_registry())
    return _store
