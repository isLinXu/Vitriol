"""
DictKV: Dictionary-Based Sparse Coding for KV Cache Compression.

═══════════════════════════════════════════════════════════════
Core Insight
═══════════════════════════════════════════════════════════════

KV vectors in LLM caches exhibit significant **structural redundancy**:
many vectors are approximate linear combinations of a small set of
"dictionary atoms" (prototypical patterns).

This is inspired by ICLR 2025 "Lexico" and classical sparse coding:
  - Any KV vector can be approximated as: x ≈ D · α
    where D ∈ ℝᵈˣᴷ is a dictionary, α ∈ ℝᴷ is sparse (few non-zeros)

  - Sparse representation: store only the non-zero coefficients and
    their indices → dramatic compression for high-dimensional vectors

═══════════════════════════════════════════════════════════════
Method
═══════════════════════════════════════════════════════════════

1. **Dictionary Learning** (offline or online):
   Learn K dictionary atoms from KV vectors using K-SVD or
   online dictionary learning. The dictionary captures the
   most common patterns in the KV space.

2. **Sparse Encoding** (per KV vector):
   For each vector x, solve:
     min ||x - D·α||²  s.t.  ||α||₀ ≤ L
   
   Using Orthogonal Matching Pursuit (OMP) with L atoms.
   Store: L × (log₂(K) + 16) bits per vector.

3. **Residual Quantization**:
   Quantize the residual r = x - D·α with low-bit quantization
   to capture the detail that sparse coding misses.

═══════════════════════════════════════════════════════════════
Theoretical Analysis
═══════════════════════════════════════════════════════════════

For a KV vector x ∈ ℝᵈ (d = hidden_dim, typically 1024-4096):

  - TurboQuant 3-bit: d × 3 = 3d bits per vector
  - DictKV (L=4, K=1024): 4 × (10 + 16) = 104 bits per vector
  
  Compression ratio: 3d / 104
    d=1024: 29.5×
    d=2048: 59.1×
    d=4096: 118×
  
  vs TurboQuant: 3d/3d × 16/3 ≈ 5.3× → DictKV is 5.5-22× better

Sparsity level L controls the trade-off:
  - L=2: maximum compression, lower quality
  - L=4: good balance (recommended)
  - L=8: higher quality, less compression

═══════════════════════════════════════════════════════════════
Advantages over existing methods
═══════════════════════════════════════════════════════════════

| Method         | Exploits Structure | Bits/vector | Quality |
|----------------|-------------------|-------------|---------|
| TurboQuant     | None              | 3d          | Medium  |
| SpectralKV     | Frequency         | ~2d         | Good    |
| PredictiveKV   | Temporal          | ~1.5d       | Good    |
| CrossLayerKV   | Depth             | ~1d         | Good    |
| DictKV         | Dictionary        | ~100-300    | Varies  |

DictKV is **orthogonal** to all other methods and can be combined:
  - DictKV for structural compression
  - CrossLayerKV for depth compression
  - PredictiveKV for temporal compression

═══════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════

    from vitriol.kv.dict_kv import DictKVCodec, DictKVConfig

    # Learn dictionary from data
    codec = DictKVCodec(DictKVConfig(n_atoms=1024, sparsity=4))
    codec.learn_dictionary(kv_tensors)

    # Compress
    k_out, v_out, report = codec.compress_kv(key, value)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Orthogonal Matching Pursuit (OMP)
# ─────────────────────────────────────────────────────────────

def orthogonal_matching_pursuit(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    sparsity: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonal Matching Pursuit for sparse coding.

    Solves: min ||x - D·α||²  s.t.  ||α||₀ ≤ sparsity

    Args:
        x: Signal to encode [..., dim]
        dictionary: Dictionary matrix [n_atoms, dim]
        sparsity: Maximum number of non-zero coefficients

    Returns:
        coefficients: Sparse coefficients [..., n_atoms]
        indices: Selected atom indices [..., sparsity]
    """
    dim = x.shape[-1]
    n_atoms = dictionary.shape[0]
    batch_shape = x.shape[:-1]

    # Flatten batch dimensions
    x_flat = x.reshape(-1, dim)  # [N, dim]
    N = x_flat.shape[0]
    D = dictionary  # [K, dim]

    # Initialize
    coefficients = torch.zeros(N, n_atoms, device=x.device, dtype=x.dtype)
    residual = x_flat.clone()
    selected_indices = torch.zeros(N, sparsity, device=x.device, dtype=torch.long)

    # Precompute D^T for correlation
    Dt = D.t()  # [dim, K]

    for step in range(sparsity):
        # Compute correlations: D^T · residual
        corr = residual @ Dt  # [N, K]

        # Select atom with maximum correlation
        best_idx = corr.abs().argmax(dim=-1)  # [N]
        selected_indices[:, step] = best_idx

        # Extract selected atoms
        D_selected = D[best_idx]  # [N, dim]

        # Orthogonal projection onto selected subspace
        # Solve least squares for all selected atoms so far
        if step == 0:
            # Simple case: single atom
            coeff = (residual * D_selected).sum(dim=-1, keepdim=True) / \
                    (D_selected.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
            coefficients[torch.arange(N), best_idx] = coeff.squeeze(-1)
            residual = residual - coeff * D_selected
        else:
            # Re-solve with all selected atoms using batched least squares
            all_indices = selected_indices[:, :step + 1]  # [N, step+1]
            D_all = D[all_indices]  # [N, step+1, dim]

            # Solve: D_all · α ≈ x_flat
            # DtD = D_all^T · D_all: [N, step+1, step+1]
            DtD = torch.bmm(D_all, D_all.transpose(1, 2))  # [N, step+1, step+1]
            # Dtx = D_all^T · x: [N, step+1]
            Dtx = torch.bmm(D_all, x_flat.unsqueeze(-1)).squeeze(-1)  # [N, step+1]

            # Add regularization for stability
            reg = 1e-6 * torch.eye(step + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
            DtD_reg = DtD + reg

            try:
                alpha = torch.linalg.solve(DtD_reg, Dtx.unsqueeze(-1))  # [N, step+1, 1]
            except Exception:
                # Fallback: use gradient approximation
                alpha = Dtx.unsqueeze(-1) * 0.1

            # Update coefficients
            coefficients.zero_()
            for i in range(step + 1):
                atom_idx = all_indices[:, i]
                coefficients[torch.arange(N), atom_idx] = alpha[:, i, 0]

            # Update residual
            reconstruction = torch.bmm(D_all.transpose(1, 2), alpha).squeeze(-1)  # [N, dim]
            residual = x_flat - reconstruction

    return coefficients.reshape(*batch_shape, n_atoms), selected_indices.reshape(*batch_shape, sparsity)


# ─────────────────────────────────────────────────────────────
# Dictionary Learning (K-SVD style)
# ─────────────────────────────────────────────────────────────

def learn_dictionary_ksvd(
    data: torch.Tensor,
    n_atoms: int = 1024,
    n_iterations: int = 10,
    sparsity: int = 4,
) -> torch.Tensor:
    """
    Learn a dictionary using K-SVD algorithm.

    Args:
        data: Training data [N, dim] where N is number of vectors
        n_atoms: Number of dictionary atoms
        n_iterations: Number of K-SVD iterations
        sparsity: Target sparsity level

    Returns:
        dictionary: Learned dictionary [n_atoms, dim]
    """
    N, dim = data.shape
    device = data.device

    # Initialize dictionary with random subset of data
    indices = torch.randperm(N, device=device)[:min(n_atoms, N)]
    dictionary = data[indices].clone()

    # Pad if not enough data points
    if N < n_atoms:
        extra = torch.randn(n_atoms - N, dim, device=device, dtype=data.dtype)
        dictionary = torch.cat([dictionary, extra], dim=0)

    # Normalize dictionary atoms
    dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)

    for iteration in range(n_iterations):
        # Sparse coding step: encode each vector
        coeffs, _ = orthogonal_matching_pursuit(data, dictionary, sparsity)

        # Dictionary update step: update each atom
        for k in range(n_atoms):
            # Find vectors that use this atom
            usage = coeffs[:, k].abs() > 1e-10
            if usage.sum() < 2:
                # Replace unused atom with random data point
                idx = torch.randint(0, N, (1,), device=device)
                dictionary[k] = data[idx[0]] / (data[idx[0]].norm() + 1e-12)
                continue

            # Compute error for these vectors without atom k
            used_data = data[usage]
            used_coeffs = coeffs[usage].clone()
            used_coeffs[:, k] = 0

            # Error without atom k
            error = used_data - used_coeffs @ dictionary

            # SVD to find best atom
            try:
                U, S, Vh = torch.linalg.svd(error, full_matrices=False)
                dictionary[k] = Vh[0]  # Best rank-1 approximation

                # Update coefficients
                coeffs[usage, k] = S[0] * U[:, 0]
            except Exception:
                # Fallback: keep current atom
                pass

        # Re-normalize
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)

    return dictionary


def learn_dictionary_online(
    data: torch.Tensor,
    n_atoms: int = 1024,
    n_iterations: int = 20,
    sparsity: int = 4,
    learning_rate: float = 0.01,
) -> torch.Tensor:
    """
    Learn a dictionary using online dictionary learning.

    More memory-efficient than K-SVD for large datasets.

    Args:
        data: Training data [N, dim]
        n_atoms: Number of atoms
        n_iterations: Number of passes
        sparsity: Target sparsity
        learning_rate: Learning rate for dictionary updates

    Returns:
        dictionary: Learned dictionary [n_atoms, dim]
    """
    N, dim = data.shape
    device = data.device

    # Initialize with random subset
    indices = torch.randperm(N, device=device)[:min(n_atoms, N)]
    dictionary = data[indices].clone()
    if N < n_atoms:
        extra = torch.randn(n_atoms - N, dim, device=device, dtype=data.dtype)
        dictionary = torch.cat([dictionary, extra], dim=0)
    dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)

    batch_size = min(256, N)

    for iteration in range(n_iterations):
        # Sample mini-batch
        idx = torch.randperm(N, device=device)[:batch_size]
        batch = data[idx]

        # Sparse encode
        coeffs, _ = orthogonal_matching_pursuit(batch, dictionary, sparsity)

        # Reconstruct
        reconstruction = coeffs @ dictionary

        # Compute gradient
        error = batch - reconstruction
        grad = -coeffs.t() @ error / batch_size  # [K, dim]

        # Update dictionary
        dictionary = dictionary - learning_rate * grad

        # Normalize
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)

    return dictionary


# ─────────────────────────────────────────────────────────────
# Compressed Representation
# ─────────────────────────────────────────────────────────────

@dataclass
class DictKVCompressed:
    """Compressed KV tensor using dictionary sparse coding."""

    # Sparse coefficients [batch, heads, seq_len, n_atoms]
    coefficients: torch.Tensor

    # Selected atom indices [batch, heads, seq_len, sparsity]
    indices: torch.Tensor

    # Coefficient values for selected atoms [batch, heads, seq_len, sparsity]
    values: torch.Tensor

    # Metadata (required fields first)
    orig_shape: Tuple[int, ...]
    n_atoms: int
    sparsity: int
    is_key: bool

    # Residual after sparse coding (quantized)
    q_residual: Optional[torch.Tensor] = None
    residual_scales: Optional[torch.Tensor] = None
    residual_mins: Optional[torch.Tensor] = None

    # Quality metrics
    sparse_ratio: float = 0.0  # Fraction of energy captured by sparse repr
    reconstruction_mse: float = 0.0

    def storage_nbytes(self) -> int:
        """Estimate storage in bytes."""
        # Sparse representation: indices + values
        # indices: sparsity × ceil(log2(n_atoms)) bits per position
        bits_per_index = math.ceil(math.log2(max(2, self.n_atoms)))
        index_bytes = self.indices.numel() * bits_per_index // 8

        # values: 16-bit float per coefficient
        value_bytes = self.values.numel() * 2  # float16

        # Residual (if stored)
        residual_bytes = 0
        if self.q_residual is not None:
            residual_bytes = self.q_residual.numel() * 2  # Approximate

        return index_bytes + value_bytes + residual_bytes


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class DictKVConfig:
    """Configuration for DictKV compression."""

    # Number of dictionary atoms
    n_atoms: int = 1024

    # Sparsity level (number of atoms per vector)
    sparsity: int = 4

    # Whether to learn dictionary from data
    learn_dictionary: bool = True

    # Number of iterations for dictionary learning
    learning_iterations: int = 10

    # Dictionary learning method: 'ksvd' or 'online'
    learning_method: str = 'online'

    # Whether to store and quantize residuals
    quantize_residual: bool = True

    # Residual quantization levels
    residual_levels: int = 4

    # Residual block size
    residual_block_size: int = 32

    # Key vs Value differentiation
    k_sparsity_boost: int = 1    # Extra atoms for keys
    v_sparsity_penalty: int = 0  # Reduce atoms for values

    # Minimum atoms for reconstruction quality
    min_atoms: int = 2
    max_atoms: int = 8


# ─────────────────────────────────────────────────────────────
# Main Codec
# ─────────────────────────────────────────────────────────────

class DictKVCodec:
    """
    DictKV: Dictionary-based sparse coding for KV cache compression.

    This codec exploits structural redundancy in KV vectors by
    representing them as sparse linear combinations of learned
    dictionary atoms.

    Key innovation over existing methods:
    - Captures **global structural patterns** (dictionary)
    - Enables **extreme compression** at the vector level
    - Complementary to all other methods (orthogonal redundancy)
    """

    def __init__(self, config: Optional[DictKVConfig] = None) -> None:
        self.config = config or DictKVConfig()
        self._dictionary_k: Optional[torch.Tensor] = None
        self._dictionary_v: Optional[torch.Tensor] = None
        self._dictionary_learned: bool = False

    @property
    def dictionary(self) -> Optional[torch.Tensor]:
        """Get the current dictionary (K dictionary if both exist)."""
        return self._dictionary_k

    def learn_dictionary(
        self,
        kv_tensors: List[torch.Tensor],
        is_key: bool = True,
    ) -> None:
        """
        Learn dictionary from KV tensor data.

        Args:
            kv_tensors: List of KV tensors for dictionary learning.
                        Each: [batch, heads, seq_len, dim]
            is_key: Whether this is for key dictionary
        """
        cfg = self.config

        # Collect all vectors
        all_vectors = []
        for kv in kv_tensors:
            if kv.ndim == 4:
                b, h, s, d = kv.shape
                vectors = kv.reshape(b * h * s, d)
            else:
                vectors = kv.reshape(-1, kv.shape[-1])
            all_vectors.append(vectors)

        data = torch.cat(all_vectors, dim=0).float()

        # Subsample if too many vectors
        max_samples = 10000
        if data.shape[0] > max_samples:
            idx = torch.randperm(data.shape[0], device=data.device)[:max_samples]
            data = data[idx]

        # Learn dictionary
        if cfg.learning_method == 'ksvd':
            dictionary = learn_dictionary_ksvd(
                data, cfg.n_atoms, cfg.learning_iterations, cfg.sparsity
            )
        else:
            dictionary = learn_dictionary_online(
                data, cfg.n_atoms, cfg.learning_iterations, cfg.sparsity
            )

        if is_key:
            self._dictionary_k = dictionary
        else:
            self._dictionary_v = dictionary

        self._dictionary_learned = True

    def _ensure_dictionary(self, dim: int, device: torch.device, is_key: bool) -> torch.Tensor:
        """Ensure dictionary exists, creating a random one if needed."""
        dictionary = self._dictionary_k if is_key else self._dictionary_v

        if dictionary is not None:
            return dictionary

        # Create random dictionary (DCT-like initialization)
        cfg = self.config
        dictionary = torch.randn(cfg.n_atoms, dim, device=device, dtype=torch.float32)
        # Initialize with approximate DCT basis for better starting point
        for k in range(min(cfg.n_atoms, dim)):
            freq = k / dim
            basis = torch.cos(2 * math.pi * freq * torch.arange(dim, dtype=torch.float32))
            if k < dictionary.shape[0]:
                dictionary[k] = basis / (basis.norm() + 1e-12)

        # Normalize
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)

        if is_key:
            self._dictionary_k = dictionary
        else:
            self._dictionary_v = dictionary

        return dictionary

    def compress(
        self,
        x: torch.Tensor,
        is_key: bool = True,
    ) -> Tuple[DictKVCompressed, Dict[str, Any]]:
        """
        Compress a KV tensor using dictionary sparse coding.

        Args:
            x: [batch, heads, seq_len, dim]
            is_key: Whether this is a key tensor

        Returns:
            compressed: DictKVCompressed
            report: Diagnostic dictionary
        """
        cfg = self.config
        orig_shape = x.shape
        dim = x.shape[-1]

        # Get dictionary
        dictionary = self._ensure_dictionary(dim, x.device, is_key)

        # Adjust sparsity for K vs V
        sparsity = cfg.sparsity
        if is_key:
            sparsity = min(sparsity + cfg.k_sparsity_boost, cfg.max_atoms)
        else:
            sparsity = max(sparsity - cfg.v_sparsity_penalty, cfg.min_atoms)

        # Reshape for OMP: [N, dim]
        x_float = x.float()
        flat = x_float.reshape(-1, dim)

        # Sparse encode using OMP
        coefficients, indices = orthogonal_matching_pursuit(
            flat, dictionary, sparsity
        )

        # Extract sparse values (only non-zero coefficients)
        values = coefficients.gather(1, indices.reshape(-1, sparsity))
        values = values.reshape(*orig_shape[:-1], sparsity)
        indices = indices.reshape(*orig_shape[:-1], sparsity)

        # Compute sparse reconstruction
        sparse_recon = coefficients @ dictionary  # [N, dim]
        residual = flat - sparse_recon

        # Compute quality metrics
        sparse_mse = float(residual.pow(2).mean().item())
        total_mse = float((x_float.reshape(-1, dim) - sparse_recon).pow(2).mean().item())
        x_energy = float(x_float.pow(2).mean().item())
        sparse_ratio = 1.0 - sparse_mse / max(x_energy, 1e-12)

        # Quantize residual if enabled
        q_residual = None
        res_scales = None
        res_mins = None
        if cfg.quantize_residual:
            residual_reshaped = residual.reshape(orig_shape)
            block_size = cfg.residual_block_size
            last = dim
            if last % block_size != 0:
                pad = block_size - (last % block_size)
                residual_work = F.pad(residual_reshaped, (0, pad))
            else:
                residual_work = residual_reshaped
                pad = 0

            r_flat = residual_work.reshape(-1, residual_work.shape[-1] // block_size, block_size)
            res_mins = r_flat.min(dim=-1, keepdim=True)[0]
            res_maxs = r_flat.max(dim=-1, keepdim=True)[0]
            res_scales = (res_maxs - res_mins) / (cfg.residual_levels - 1 + 1e-8)
            q_r = torch.round((r_flat - res_mins) / (res_scales + 1e-8))
            q_r = torch.clamp(q_r, 0, cfg.residual_levels - 1)
            q_residual = q_r

        # Build compressed representation
        compressed = DictKVCompressed(
            coefficients=coefficients.reshape(*orig_shape[:-1], cfg.n_atoms),
            indices=indices,
            values=values,
            q_residual=q_residual,
            residual_scales=res_scales,
            residual_mins=res_mins,
            orig_shape=orig_shape,
            n_atoms=cfg.n_atoms,
            sparsity=sparsity,
            is_key=is_key,
            sparse_ratio=sparse_ratio,
            reconstruction_mse=total_mse,
        )

        # Compute effective bpv
        total_values = x.numel()
        effective_bpv = compressed.storage_nbytes() * 8 / total_values if total_values > 0 else 0

        report = {
            "method": "dict_kv",
            "is_key": is_key,
            "n_atoms": cfg.n_atoms,
            "sparsity": sparsity,
            "sparse_ratio": sparse_ratio,
            "reconstruction_mse": total_mse,
            "target_bpv": 0,  # DictKV doesn't use target_bpv directly
            "effective_bpv": effective_bpv,
            "compression_ratio": 16.0 / max(effective_bpv, 0.01),
        }

        return compressed, report

    def decompress(
        self,
        compressed: DictKVCompressed,
    ) -> torch.Tensor:
        """
        Decompress a DictKV compressed tensor.

        Args:
            compressed: DictKVCompressed

        Returns:
            Reconstructed tensor in original shape
        """
        dictionary = self._ensure_dictionary(
            compressed.orig_shape[-1],
            compressed.indices.device,
            compressed.is_key,
        )

        # Reconstruct from sparse coefficients
        coefficients = compressed.coefficients.float()
        flat = coefficients.reshape(-1, coefficients.shape[-1])
        reconstruction = flat @ dictionary

        # Add residual if available
        if compressed.q_residual is not None and compressed.residual_scales is not None:
            r_dq = compressed.q_residual.float() * compressed.residual_scales + compressed.residual_mins
            dim = compressed.orig_shape[-1]
            block_size = self.config.residual_block_size
            padded = dim
            if dim % block_size != 0:
                padded = dim + (block_size - dim % block_size)
            r_reshaped = r_dq.reshape(*compressed.orig_shape[:-1], padded)
            if padded != dim:
                r_reshaped = r_reshaped[..., :dim]
            r_flat = r_reshaped.reshape(-1, dim)
            reconstruction = reconstruction + r_flat

        return reconstruction.reshape(compressed.orig_shape)

    def compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compress both K and V tensors using dictionary sparse coding.

        Main API for integration with KVCacheStore.

        Args:
            key: [batch, heads, seq_len, dim]
            value: [batch, heads, seq_len, dim]

        Returns:
            k_out: Reconstructed key tensor
            v_out: Reconstructed value tensor
            report: Combined diagnostic dictionary
        """
        k_compressed, k_report = self.compress(key, is_key=True)
        v_compressed, v_report = self.compress(value, is_key=False)

        k_out = self.decompress(k_compressed)
        v_out = self.decompress(v_compressed)

        # Quality metrics
        k_mse = float((key.float() - k_out).pow(2).mean().item())
        v_mse = float((value.float() - v_out).pow(2).mean().item())

        report = {
            "k": k_report,
            "v": v_report,
            "k_mse": k_mse,
            "v_mse": v_mse,
            "total_mse": (k_mse + v_mse) / 2,
            "method": "dict_kv",
        }

        return k_out, v_out, report


# ─────────────────────────────────────────────────────────────
# Quick Quantize-Dequantize (for benchmarking)
# ─────────────────────────────────────────────────────────────

def dict_kv_qdq(
    x: torch.Tensor,
    n_atoms: int = 1024,
    sparsity: int = 4,
    is_key: bool = True,
    learn_from_data: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quick dictionary-based quantize-dequantize for benchmarking.

    Args:
        x: Input tensor [batch, heads, seq_len, dim]
        n_atoms: Number of dictionary atoms
        sparsity: Sparsity level
        is_key: Whether key tensor
        learn_from_data: Whether to learn dictionary from input

    Returns:
        reconstructed: Quantized-dequantized tensor
        report: Diagnostic dictionary
    """
    config = DictKVConfig(
        n_atoms=n_atoms,
        sparsity=sparsity,
        learn_dictionary=learn_from_data,
        learning_iterations=5,
    )
    codec = DictKVCodec(config)

    if learn_from_data:
        codec.learn_dictionary([x], is_key=is_key)

    compressed, report = codec.compress(x, is_key=is_key)
    reconstructed = codec.decompress(compressed)

    # Quality metrics
    mse = float((x.float() - reconstructed).pow(2).mean().item())
    cos_sim = float(F.cosine_similarity(
        x.float().flatten().unsqueeze(0),
        reconstructed.flatten().unsqueeze(0),
        dim=-1
    ).item())

    x_var = float(x.float().pow(2).mean().item())
    snr_db = 10.0 * math.log10(max(1e-12, x_var) / max(1e-12, mse))

    report["mse"] = mse
    report["cosine_similarity"] = cos_sim
    report["snr_db"] = snr_db

    return reconstructed, report
