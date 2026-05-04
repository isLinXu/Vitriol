"""
Model Fingerprinting System for Unique Model Identification.

This module implements a comprehensive fingerprinting system that creates
unique, deterministic identifiers for models based on their architecture
and weight characteristics. Useful for:
- Model versioning and tracking
- Verifying model integrity
- Detecting unauthorized modifications
- Model marketplace verification
"""

import logging
import hashlib
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelFingerprint:
    """
    Complete fingerprint of a model.
    
    Contains multiple hash types for different use cases:
    - architecture_hash: Identifies model architecture (structure only)
    - weights_hash: Identifies exact weight values
    - content_hash: Combined architecture + weights
    - signature: Cryptographic signature for verification
    """
    model_id: str
    architecture_hash: str
    weights_hash: str
    content_hash: str
    signature: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary."""
        return {
            "model_id": self.model_id,
            "architecture_hash": self.architecture_hash,
            "weights_hash": self.weights_hash,
            "content_hash": self.content_hash,
            "signature": self.signature,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelFingerprint":
        """Create fingerprint from dictionary."""
        return cls(**data)
    
    def verify(self, other: "ModelFingerprint") -> Dict[str, bool]:
        """
        Verify fingerprint against another.
        
        Returns:
            Dict with verification results
        """
        return {
            "same_architecture": self.architecture_hash == other.architecture_hash,
            "same_weights": self.weights_hash == other.weights_hash,
            "identical": self.content_hash == other.content_hash,
            "signature_valid": self.signature == other.signature
        }


class ArchitectureHasher:
    """
    Creates deterministic hashes of model architecture.
    
    Ignores weight values, only considers:
    - Layer types and configurations
    - Connectivity patterns
    - Hyperparameters
    """
    
    def hash(self, model: nn.Module) -> str:
        """
        Create architecture hash.
        
        Args:
            model: PyTorch model
            
        Returns:
            Hexadecimal hash string
        """
        architecture_info = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            
            # Extract layer info
            layer_info = self._extract_layer_info(name, module)
            if layer_info:
                architecture_info.append(layer_info)
        
        # Create deterministic string
        arch_str = json.dumps(architecture_info, sort_keys=True, separators=(',', ':'))
        
        # Hash
        return hashlib.sha256(arch_str.encode()).hexdigest()[:32]
    
    def _extract_layer_info(self, name: str, module: nn.Module) -> Optional[Dict]:
        """Extract serializable layer information."""
        info = {"name": name, "type": module.__class__.__name__}
        
        if isinstance(module, nn.Linear):
            info.update({
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None
            })
        
        elif isinstance(module, nn.Conv2d):
            info.update({
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "bias": module.bias is not None
            })
        
        elif isinstance(module, nn.Embedding):
            info.update({
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim
            })
        
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            info.update({
                "normalized_shape": getattr(module, 'normalized_shape', None),
                "num_features": getattr(module, 'num_features', None),
                "eps": module.eps,
                "elementwise_affine": module.elementwise_affine
            })
        
        else:
            return None
        
        return info


class WeightsHasher:
    """
    Creates deterministic hashes of model weights.
    
    Uses perceptual hashing techniques to be robust to
    minor numerical differences while detecting significant changes.
    """
    
    def __init__(self, precision: int = 6):
        """
        Initialize weights hasher.
        
        Args:
            precision: Decimal precision for rounding (higher = more sensitive)
        """
        self.precision = precision
    
    def hash(self, model: nn.Module) -> str:
        """
        Create weights hash.
        
        Args:
            model: PyTorch model
            
        Returns:
            Hexadecimal hash string
        """
        weight_stats = []
        
        for name, param in model.named_parameters():
            # Compute statistics that characterize the weights
            stats = self._compute_weight_stats(name, param)
            weight_stats.append(stats)
        
        # Create deterministic string
        stats_str = json.dumps(weight_stats, sort_keys=True, separators=(',', ':'))
        
        # Hash
        return hashlib.sha256(stats_str.encode()).hexdigest()[:32]
    
    def _compute_weight_stats(self, name: str, param: torch.Tensor) -> Dict:
        """Compute statistics for a weight tensor."""
        # Convert to numpy for processing
        weights = param.detach().cpu().numpy()
        
        # Round to specified precision for robustness
        weights = np.round(weights, self.precision)
        
        # Compute statistics
        stats = {
            "name": name,
            "shape": list(param.shape),
            "mean": round(float(np.mean(weights)), self.precision),
            "std": round(float(np.std(weights)), self.precision),
            "min": round(float(np.min(weights)), self.precision),
            "max": round(float(np.max(weights)), self.precision),
            "median": round(float(np.median(weights)), self.precision),
            "sparsity": round(float(np.mean(weights == 0)), self.precision),
            # Perceptual features
            "histogram": self._compute_histogram(weights),
            "gradient_stats": self._compute_gradient_stats(weights)
        }
        
        return stats
    
    def _compute_histogram(self, weights: np.ndarray, bins: int = 8) -> List[float]:
        """Compute binned histogram of weights."""
        hist, _ = np.histogram(weights, bins=bins, range=(-3, 3))
        # Normalize
        hist = hist / (np.sum(hist) + 1e-10)
        return [round(float(h), self.precision) for h in hist]
    
    def _compute_gradient_stats(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute gradient-like statistics."""
        if weights.ndim < 2:
            return {"dx": 0.0, "dy": 0.0}
        
        # Compute differences along dimensions
        dx = np.diff(weights, axis=0)
        dy = np.diff(weights, axis=1) if weights.ndim > 1 else np.array([0])
        
        return {
            "dx_mean": round(float(np.mean(np.abs(dx))), self.precision),
            "dy_mean": round(float(np.mean(np.abs(dy))), self.precision)
        }


class FingerprintEngine:
    """
    Main engine for model fingerprinting.
    
    Provides comprehensive fingerprinting capabilities including
    architecture hashing, weights hashing, and cryptographic signatures.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize fingerprint engine.
        
        Args:
            secret_key: Optional secret key for signing fingerprints
        """
        self.arch_hasher = ArchitectureHasher()
        self.weights_hasher = WeightsHasher()
        self.secret_key = secret_key or os.environ.get(
            "VITRIOL_FINGERPRINT_SECRET", "vitriol_default_key"
        )
    
    def fingerprint(
        self,
        model: nn.Module,
        model_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ModelFingerprint:
        """
        Create complete fingerprint of a model.
        
        Args:
            model: PyTorch model to fingerprint
            model_id: Optional model identifier
            metadata: Optional metadata dict
            
        Returns:
            ModelFingerprint object
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(model)
        
        # Compute hashes
        arch_hash = self.arch_hasher.hash(model)
        weights_hash = self.weights_hasher.hash(model)
        
        # Compute content hash (combination)
        content_str = f"{arch_hash}:{weights_hash}"
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:32]
        
        # Create signature
        signature = self._create_signature(model_id, content_hash)
        
        fingerprint = ModelFingerprint(
            model_id=model_id,
            architecture_hash=arch_hash,
            weights_hash=weights_hash,
            content_hash=content_hash,
            signature=signature,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        logger.info(f"Created fingerprint for model {model_id}")
        return fingerprint
    
    def _generate_model_id(self, model: nn.Module) -> str:
        """Generate unique model ID."""
        # Use architecture hash as base
        arch_hash = self.arch_hasher.hash(model)
        
        # Add timestamp for uniqueness
        timestamp = int(time.time())
        
        return f"vitriol_{arch_hash[:16]}_{timestamp}"
    
    def _create_signature(self, model_id: str, content_hash: str) -> str:
        """Create cryptographic signature."""
        signature_str = f"{model_id}:{content_hash}:{self.secret_key}"
        return hashlib.sha256(signature_str.encode()).hexdigest()[:32]
    
    def verify_signature(self, fingerprint: ModelFingerprint) -> bool:
        """
        Verify fingerprint signature.
        
        Args:
            fingerprint: Fingerprint to verify
            
        Returns:
            True if signature is valid
        """
        expected = self._create_signature(fingerprint.model_id, fingerprint.content_hash)
        return fingerprint.signature == expected
    
    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module
    ) -> Dict[str, Any]:
        """
        Compare two models and return similarity metrics.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Dict with comparison results
        """
        fp1 = self.fingerprint(model1)
        fp2 = self.fingerprint(model2)
        
        # Architecture comparison
        arch_match = fp1.architecture_hash == fp2.architecture_hash
        
        # Weights comparison
        weights_match = fp1.weights_hash == fp2.weights_hash
        
        # Detailed comparison
        comparison = {
            "identical": arch_match and weights_match,
            "same_architecture": arch_match,
            "same_weights": weights_match,
            "model1_id": fp1.model_id,
            "model2_id": fp2.model_id,
            "architecture_similarity": self._compute_arch_similarity(model1, model2),
            "weights_similarity": self._compute_weights_similarity(model1, model2) if not weights_match else 1.0
        }
        
        return comparison
    
    def _compute_arch_similarity(self, model1: nn.Module, model2: nn.Module) -> float:
        """Compute architecture similarity (0-1)."""
        layers1 = [(n, m.__class__.__name__) for n, m in model1.named_modules() if len(list(m.children())) == 0]
        layers2 = [(n, m.__class__.__name__) for n, m in model2.named_modules() if len(list(m.children())) == 0]
        
        if len(layers1) == 0 and len(layers2) == 0:
            return 1.0
        
        # Simple Jaccard similarity on layer types
        types1 = set(layer[1] for layer in layers1)
        types2 = set(layer[1] for layer in layers2)
        
        intersection = len(types1 & types2)
        union = len(types1 | types2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_weights_similarity(self, model1: nn.Module, model2: nn.Module) -> float:
        """Compute weights similarity (0-1)."""
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        
        common_params = set(params1.keys()) & set(params2.keys())
        
        if not common_params:
            return 0.0
        
        similarities = []
        for name in common_params:
            p1 = params1[name].detach().cpu().numpy().flatten()
            p2 = params2[name].detach().cpu().numpy().flatten()
            
            # Cosine similarity
            if len(p1) == len(p2):
                sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10)
                similarities.append(abs(sim))
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def save_fingerprint(self, fingerprint: ModelFingerprint, path: str):
        """Save fingerprint to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(fingerprint.to_dict(), f, indent=2)
        
        logger.info(f"Saved fingerprint to {path}")
    
    def load_fingerprint(self, path: str) -> ModelFingerprint:
        """Load fingerprint from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return ModelFingerprint.from_dict(data)


class FingerprintRegistry:
    """
    Registry for tracking multiple model fingerprints.
    
    Useful for model versioning, lineage tracking, and marketplace verification.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize registry.
        
        Args:
            storage_path: Optional path for persistent storage
        """
        self.fingerprints: Dict[str, ModelFingerprint] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        self.engine = FingerprintEngine()
        
        if self.storage_path and self.storage_path.exists():
            self._load_registry()
    
    def register(self, model: nn.Module, model_id: Optional[str] = None, metadata: Optional[Dict] = None) -> ModelFingerprint:
        """Register a model in the registry."""
        fingerprint = self.engine.fingerprint(model, model_id, metadata)
        self.fingerprints[fingerprint.model_id] = fingerprint
        
        if self.storage_path:
            self._save_registry()
        
        return fingerprint
    
    def verify(self, model: nn.Module) -> Dict[str, Any]:
        """Verify a model against registered fingerprints."""
        test_fp = self.engine.fingerprint(model)
        
        results = []
        for fp in self.fingerprints.values():
            verification = test_fp.verify(fp)
            results.append({
                "registered_id": fp.model_id,
                **verification
            })
        
        return {
            "test_model_id": test_fp.model_id,
            "matches": [r for r in results if r["identical"]],
            "architecture_matches": [r for r in results if r["same_architecture"]],
            "all_results": results
        }
    
    def get_lineage(self, model_id: str) -> List[Dict]:
        """Get model lineage (versions of same architecture)."""
        if model_id not in self.fingerprints:
            return []
        
        target_arch = self.fingerprints[model_id].architecture_hash
        
        lineage = []
        for fp in self.fingerprints.values():
            if fp.architecture_hash == target_arch:
                lineage.append({
                    "model_id": fp.model_id,
                    "timestamp": fp.timestamp,
                    "weights_hash": fp.weights_hash
                })
        
        # Sort by timestamp
        lineage.sort(key=lambda x: x["timestamp"])
        return lineage
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {
            model_id: fp.to_dict()
            for model_id, fp in self.fingerprints.items()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_registry(self):
        """Load registry from disk."""
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.fingerprints = {
            model_id: ModelFingerprint.from_dict(fp_data)
            for model_id, fp_data in data.items()
        }
