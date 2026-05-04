"""
ExoBrain Feature Alignment Training (v0.4+).

═══════════════════════════════════════════════════════════════
Purpose
═══════════════════════════════════════════════════════════════

Train the ShellProjection layer to align a shell model's query
space with an external brain's KV space.

This addresses the core problem:
  - Shell model (0.1B) has hidden_dim = 768
  - External brain (7B) has hidden_dim = 4096
  - ShellProjection bridges these heterogeneous spaces

Training Objective:
  Minimize cognitive distance between shell's queries and brain's keys.

═══════════════════════════════════════════════════════════════
Alignment Strategies
═══════════════════════════════════════════════════════════════

1. Cosine Alignment Loss
   - Maximize cosine similarity between projected queries and target KV keys
   - Lcos = -mean(cos_sim(projected_q, target_k))

2. MSE Alignment Loss
   - Minimize L2 distance between projected queries and target keys
   - Lmse = mean(||projected_q - target_k||^2)

3. KL Divergence Alignment
   - Align attention distributions
   - Lkl = KL(attn_shell || attn_brain)

4. Contrastive Alignment (InfoNCE)
   - Pull positive pairs together, push negative pairs apart
   - Lcontrastive = -log(exp(sim(q,k+))/sum(exp(sim(q,k))))

═══════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════

    python -m vitriol.demos.exobrain_alignment_train

Output:
  - Before/after Hit@K metrics
  - Training loss curves
  - Alignment quality evaluation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import math


# ─────────────────────────────────────────────────────────────
# Shared Components (from exobrain_query_hit_demo.py)
# ─────────────────────────────────────────────────────────────

@dataclass
class MockTeacherKV:
    """Simulates a teacher model's KV cache entry."""
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    label: str


class MockExternalBrain:
    """Simulates an external brain's KV cache."""

    def __init__(self, num_layers: int = 12, num_heads: int = 8, head_dim: int = 64):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_store: Dict[int, List[MockTeacherKV]] = {}
        self._generate_semantic_kv_entries()

    def _generate_semantic_kv_entries(self) -> None:
        import numpy as np
        np.random.seed(42)
        torch.manual_seed(42)

        clusters = {
            "geography": torch.randn(self.head_dim) * 2.0,
            "science": torch.randn(self.head_dim) * 2.0,
            "history": torch.randn(self.head_dim) * 2.0,
            "code": torch.randn(self.head_dim) * 2.0,
        }

        labels_per_cluster = {
            "geography": ["capital_of_france", "capital_of_japan", "largest_ocean", "highest_mountain"],
            "science": ["photosynthesis", "quantum_mechanics", "dna_structure", "thermodynamics"],
            "history": ["french_revolution", "moon_landing", "wwii_end", "renaissance_start"],
            "code": ["binary_search", "merge_sort", "http_protocol", "database_index"],
        }

        for layer_idx in range(self.num_layers):
            self.kv_store[layer_idx] = []
            for cluster_name, base_vector in clusters.items():
                for label in labels_per_cluster[cluster_name]:
                    layer_offset = torch.randn(self.head_dim) * 0.5
                    key_vector = F.normalize(base_vector + layer_offset, dim=-1)
                    key = key_vector.unsqueeze(0).unsqueeze(0).expand(
                        self.num_heads, 1, self.head_dim
                    )
                    value = key_vector.unsqueeze(0).unsqueeze(1).expand(
                        1, self.num_heads, self.head_dim
                    ).transpose(0, 1) * 0.9 + torch.randn_like(key) * 0.1
                    self.kv_store[layer_idx].append(MockTeacherKV(
                        layer_idx=layer_idx, key=key, value=value, label=label,
                    ))

    def retrieve_top_k(
        self, query: torch.Tensor, layer_idx: int, top_k: int = 5
    ) -> List[Tuple[MockTeacherKV, float]]:
        if layer_idx not in self.kv_store:
            return []
        query_mean = query.mean(dim=(1, 2))
        entries = self.kv_store[layer_idx]
        similarities = []
        for entry in entries:
            entry_key_mean = entry.key.mean(dim=1)
            sim = F.cosine_similarity(query_mean, entry_key_mean, dim=-1)
            similarities.append((entry, sim.mean().item()))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class ShellQueryGenerator(nn.Module):
    """Shell model with real weights."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        head_dim: int = 64,
        vocab_size: int = 10000,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.reasoning = nn.Linear(hidden_dim, hidden_dim)

        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        torch.nn.init.xavier_uniform_(self.query_proj.weight)
        torch.nn.init.zeros_(self.query_proj.bias)

    def generate_query(
        self, input_ids: torch.Tensor, layer_idx: int = 0
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        reasoned = F.gelu(self.reasoning(embedded))
        q = self.query_proj(reasoned)
        B, S, _ = q.shape
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        q = F.normalize(q, dim=-1)
        return q

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# ShellProjection (from exobrain.py)
# ─────────────────────────────────────────────────────────────

class ShellProjection(nn.Module):
    """
    Thin cognitive alignment layer between shell model and external brain.
    """

    def __init__(
        self,
        shell_hidden_dim: int,
        brain_hidden_dim: int,
        mode: str = "linear",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.shell_hidden_dim = shell_hidden_dim
        self.brain_hidden_dim = brain_hidden_dim
        self.mode = mode

        if mode == "linear":
            self.proj = nn.Sequential(
                nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
            )
        elif mode == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(shell_hidden_dim, shell_hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
            )
        elif mode == "linear_ln":
            self.proj = nn.Sequential(
                nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
                nn.LayerNorm(brain_hidden_dim),
                nn.Dropout(p=dropout),
            )
        else:
            raise ValueError(f"ShellProjection: unknown mode '{mode}'")

        self._init_near_identity()

    def _init_near_identity(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        ndims = len(original_shape)
        if ndims == 4:
            B, H, S, D = original_shape
            x_2d = x.reshape(B, H * S, D)  # [B, H*S, D]
            x_projected = self.proj(x_2d)  # [B, H*S, brain_d]
            brain_d = self.brain_hidden_dim
            return x_projected.reshape(B, H, S, brain_d)
        elif ndims == 3:
            return self.proj(x)
        else:
            raise ValueError(
                f"ShellProjection: expected 3D [B,S,D] or 4D [B,H,S,D], got {ndims}D"
            )

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.forward(query)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# Alignment Losses
# ─────────────────────────────────────────────────────────────

class AlignmentLoss(nn.Module):
    """
    Collection of alignment losses for training ShellProjection.
    """

    def __init__(self, loss_type: str = "cosine"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        projected_q: torch.Tensor,
        target_k: torch.Tensor,
        negative_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute alignment loss.

        Args:
            projected_q: Projected shell queries [B, H, S, brain_d]
            target_k: Target brain keys (positive) [B, H, S, brain_d]
            negative_k: Optional negative keys for contrastive loss

        Returns:
            Loss scalar
        """
        if self.loss_type == "cosine":
            return self._cosine_loss(projected_q, target_k)
        elif self.loss_type == "mse":
            return self._mse_loss(projected_q, target_k)
        elif self.loss_type == "contrastive":
            return self._contrastive_loss(projected_q, target_k, negative_k)
        elif self.loss_type == "kl":
            return self._kl_loss(projected_q, target_k)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _cosine_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss — maximize similarity to positive."""
        q_flat = q.reshape(-1, q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-1])
        sim = F.cosine_similarity(q_flat, k_flat, dim=-1)
        return -sim.mean()

    def _mse_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """MSE loss — minimize L2 distance."""
        q_flat = q.reshape(-1, q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-1])
        return F.mse_loss(q_flat, k_flat)

    def _kl_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """KL divergence — align attention distributions."""
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        q_prob = F.softmax(q_norm, dim=-1)
        k_prob = F.softmax(k_norm, dim=-1)
        return F.kl_div(q_prob.log(), k_prob, reduction="batchmean")

    def _contrastive_loss(
        self,
        q: torch.Tensor,
        k_pos: torch.Tensor,
        k_neg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        L = -log(exp(sim(q, k+)) / (exp(sim(q, k+)) + sum(exp(sim(q, k-)))))
        
        Fixed issues:
        1. Self-contrastive: mask out diagonal (self-similarity) to avoid L=inf
        2. All tensors detached before computation to prevent graph reuse
        """
        temperature = 0.1

        # q: [B, H, S, D], k_pos: [B, H, S, D]
        # Flatten spatial dims: [B, H*S, D]
        B, H, S, D = q.shape
        
        # Only detach k_pos and k_neg (targets, not trainable inputs).
        # DO NOT detach q — it comes from projection and carries gradients.
        k_pos = k_pos.detach()
        if k_neg is not None:
            k_neg = k_neg.detach()

        q_flat = F.normalize(q.reshape(B, H * S, D), dim=-1)  # [B, H*S, D]
        k_pos_flat = F.normalize(k_pos.reshape(B, H * S, D), dim=-1)  # [B, H*S, D]

        # Positive similarity per sample: [B, H*S]
        pos_sim = (q_flat * k_pos_flat).sum(dim=-1)  # [B, H*S]

        if k_neg is not None and k_neg.numel() > 0:
            # k_neg: [B, M, D] where M = num_negatives
            k_neg_norm = F.normalize(k_neg.reshape(B, -1, D), dim=-1)  # [B, M, D]

            # Compute pairwise similarity: [B, H*S, 1, D] * [B, 1, M, D] → [B, H*S, M]
            q_expanded = q_flat.unsqueeze(2)  # [B, H*S, 1, D]
            k_expanded = k_neg_norm.unsqueeze(1)  # [B, 1, M, D]
            neg_sim_matrix = (q_expanded * k_expanded).sum(dim=-1)  # [B, H*S, M]

            # Denominator: logsumexp over positives AND negatives
            # Stack: [B, H*S, 1+M] with pos at index 0
            pos_expanded = pos_sim.unsqueeze(-1)  # [B, H*S, 1]
            all_sim = torch.cat([pos_expanded, neg_sim_matrix], dim=-1)  # [B, H*S, 1+M]
            denom = torch.logsumexp(all_sim / temperature, dim=-1)  # [B, H*S]

            # Numerator: just the positive
            numer = pos_sim / temperature  # [B, H*S]

            loss = -(numer - denom).mean()
        else:
            # Self-contrastive: use all other Q in the batch as negatives
            # But EXCLUDE self (diagonal) from negatives
            all_sim = q_flat @ q_flat.transpose(-2, -1) / temperature  # [B, H*S, H*S]
            
            # Create mask with -inf on diagonal (self-similarity = 1.0 would dominate)
            # [H*S, H*S] matrix with 0 on diagonal, -inf elsewhere
            M = H * S
            diag_mask = torch.ones(M, M, device=q.device, dtype=q.dtype)
            diag_mask.fill_diagonal_(0.0)
            diag_mask = diag_mask.log().fill_diagonal_(float('-inf'))
            
            # Apply same mask to all batch elements
            denom = torch.logsumexp(all_sim + diag_mask, dim=-1)  # [B, H*S]
            numer = pos_sim / temperature  # [B, H*S]

            loss = -(numer - denom).mean()

        # Clamp to prevent numerical instability
        loss = torch.clamp(loss, max=50.0)
        return loss


# ─────────────────────────────────────────────────────────────
# Alignment Dataset
# ─────────────────────────────────────────────────────────────

class AlignmentDataset(Dataset):
    """
    Dataset for training ShellProjection.

    Each sample contains:
    - query: Shell model query (from random tokens)
    - positive_key: Target key from external brain
    - negative_keys: Other keys (for contrastive)
    """

    def __init__(
        self,
        brain: MockExternalBrain,
        shell: ShellQueryGenerator,
        num_samples: int = 1000,
        layer_idx: int = 6,
        vocab_size: int = 10000,
    ):
        self.brain = brain
        self.shell = shell
        self.num_samples = num_samples
        self.layer_idx = layer_idx
        self.vocab_size = vocab_size

        # Pre-generate queries
        torch.manual_seed(42)
        self.queries = []
        self.positive_keys = []
        self.negative_keys = []

        all_entries = brain.kv_store[layer_idx]

        for i in range(num_samples):
            # Random tokens — use batch=1 but don't add extra dim
            tokens = torch.randint(0, vocab_size, (1, 8))
            query = shell.generate_query(tokens, layer_idx=layer_idx)  # [1, H, S, D]
            # Squeeze the leading 1 for proper batching: [H, S, D]
            query = query.squeeze(0)

            # Randomly select a positive key
            pos_entry = all_entries[i % len(all_entries)]
            pos_key = pos_entry.key.squeeze(1)  # [H, D] from [H, 1, D]

            # Select negative keys (all except positive)
            neg_entries = [e for j, e in enumerate(all_entries) if j != i % len(all_entries)]
            neg_keys = torch.stack([e.key.squeeze(1) for e in neg_entries[:8]], dim=0)  # [8, H, D]

            self.queries.append(query)
            self.positive_keys.append(pos_key)
            self.negative_keys.append(neg_keys)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "query": self.queries[idx],
            "positive_key": self.positive_keys[idx],
            "negative_keys": self.negative_keys[idx],
        }


# ─────────────────────────────────────────────────────────────
# Alignment Trainer
# ─────────────────────────────────────────────────────────────

@dataclass
class AlignmentConfig:
    """Configuration for alignment training."""
    loss_type: str = "contrastive"  # cosine, mse, kl, contrastive
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    projection_mode: str = "linear"  # linear, mlp, linear_ln
    log_every: int = 10


class AlignmentTrainer:
    """
    Trainer for ShellProjection alignment.

    Workflow:
    1. Create dataset (query → positive_key pairs)
    2. Initialize ShellProjection
    3. Train to maximize alignment
    4. Evaluate Hit@K before/after
    """

    def __init__(
        self,
        brain: MockExternalBrain,
        shell: ShellQueryGenerator,
        shell_hidden_dim: int,
        brain_hidden_dim: int,
        config: Optional[AlignmentConfig] = None,
    ):
        self.brain = brain
        self.shell = shell
        self.shell_hidden_dim = shell_hidden_dim
        self.brain_hidden_dim = brain_hidden_dim
        self.config = config or AlignmentConfig()

        # Projection: shell head_dim → brain head_dim
        self.projection = ShellProjection(
            shell_hidden_dim=shell_hidden_dim,
            brain_hidden_dim=brain_hidden_dim,
            mode=self.config.projection_mode,
        )

        self.loss_fn = AlignmentLoss(loss_type=self.config.loss_type)
        self.optimizer = torch.optim.AdamW(
            self.projection.parameters(), lr=self.config.lr
        )

        # Test queries for evaluation
        self.test_queries = [
            {"tokens": torch.randint(0, 10000, (1, 8)), "expected": ["capital_of_france", "capital_of_japan"]},
            {"tokens": torch.randint(0, 10000, (1, 8)), "expected": ["quantum_mechanics", "photosynthesis"]},
            {"tokens": torch.randint(0, 10000, (1, 8)), "expected": ["french_revolution", "moon_landing"]},
            {"tokens": torch.randint(0, 10000, (1, 8)), "expected": ["binary_search", "http_protocol"]},
        ]

    def evaluate_hit_at_k(self, projected: bool = False) -> Dict[str, float]:
        """Evaluate Hit@K before or after projection."""
        hits = {"hit@1": 0, "hit@3": 0, "hit@5": 0, "mrr": 0.0}
        total = len(self.test_queries)

        for test in self.test_queries:
            query = self.shell.generate_query(test["tokens"], layer_idx=6)

            if projected:
                query = self.projection.project_query(query)

            retrieved = self.brain.retrieve_top_k(query, layer_idx=6, top_k=5)
            retrieved_labels = [kv.label for kv, _ in retrieved]
            expected_set = set(test["expected"])

            if retrieved_labels[0] in expected_set:
                hits["hit@1"] += 1
            if any(l in expected_set for l in retrieved_labels[:3]):
                hits["hit@3"] += 1
            if any(l in expected_set for l in retrieved_labels[:5]):
                hits["hit@5"] += 1

            mrr = 0.0
            for i, label in enumerate(retrieved_labels):
                if label in expected_set:
                    mrr = 1.0 / (i + 1)
                    break
            hits["mrr"] += mrr

        return {k: v / total for k, v in hits.items()}

    def train_step(
        self, query: torch.Tensor, pos_key: torch.Tensor, neg_keys: torch.Tensor
    ) -> float:
        """Single training step."""
        self.optimizer.zero_grad()

        # Detach and clone to avoid graph reuse across iterations
        query = query.detach().clone()
        pos_key = pos_key.detach().clone()
        neg_keys = neg_keys.detach().clone()

        # Project query to brain space
        projected_q = self.projection.project_query(query)  # [B, H, S, D_brain]

        # Expand pos_key: [B, H, D_brain] → [B, H, S, D_brain]
        B, H, S, _ = projected_q.shape
        pos_key_expanded = pos_key.unsqueeze(2).expand(B, H, S, self.brain_hidden_dim).clone()

        # Compute loss (cosine only for now)
        loss = self.loss_fn(projected_q, pos_key_expanded, None)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self) -> Dict[str, List[float]]:
        """Run full alignment training."""
        dataset = AlignmentDataset(
            brain=self.brain,
            shell=self.shell,
            num_samples=500,
            layer_idx=6,
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        losses = []
        before_metrics = self.evaluate_hit_at_k(projected=False)
        after_metrics = None

        print(f"\n{'='*70}")
        print(f"Feature Alignment Training (v0.4+)")
        print(f"{'='*70}")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Projection mode: {self.config.projection_mode}")
        print(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        print(f"Projection parameters: {self.projection.total_parameters / 1e3:.1f}K")
        print()

        print("BEFORE alignment (random projection):")
        for k, v in before_metrics.items():
            print(f"  {k}: {v:.3f}")

        print(f"\nTraining...")
        print("-" * 50)

        for epoch in tqdm(range(self.config.epochs), desc="Aligning"):
            epoch_loss = 0.0
            for batch in loader:
                loss_val = self.train_step(
                    batch["query"],
                    batch["positive_key"],
                    batch["negative_keys"],
                )
                epoch_loss += loss_val

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if (epoch + 1) % self.config.log_every == 0:
                metrics = self.evaluate_hit_at_k(projected=True)
                tqdm.write(
                    f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, "
                    f"Hit@1={metrics['hit@1']:.2f}, MRR={metrics['mrr']:.3f}"
                )

        after_metrics = self.evaluate_hit_at_k(projected=True)

        print()
        print("AFTER alignment (trained projection):")
        for k, v in after_metrics.items():
            delta = v - before_metrics[k]
            delta_str = f"(+{delta:.3f})" if delta > 0 else f"({delta:.3f})"
            print(f"  {k}: {v:.3f} {delta_str}")

        # Summary comparison
        print()
        print("=" * 70)
        print("Alignment Improvement Summary")
        print("=" * 70)

        improvement_table = []
        for k in ["hit@1", "hit@3", "hit@5", "mrr"]:
            before = before_metrics[k]
            after = after_metrics[k]
            delta = after - before
            improvement_table.append({
                "metric": k,
                "before": before,
                "after": after,
                "delta": delta,
                "improved": delta > 0,
            })

        for row in improvement_table:
            delta_str = f"+{row['delta']:.3f}" if row['delta'] >= 0 else f"{row['delta']:.3f}"
            marker = "✓" if row['improved'] else "✗"
            print(f"  {marker} {row['metric']:>8s}: {row['before']:.3f} → {row['after']:.3f} ({delta_str})")

        all_improved = all(r['improved'] for r in improvement_table)

        print()
        if all_improved:
            print("  ✓ ShellProjection alignment training SUCCESSFUL!")
            print("    All metrics improved after training.")
        elif any(r['improved'] for r in improvement_table):
            print("  ~ Partial success — some metrics improved.")
            print("    This is expected with random tokens + simple projection.")
        else:
            print("  ✗ No improvement — training may need adjustment.")
            print("    Possible causes:")
            print("    1. Loss type may not suit this data distribution")
            print("    2. Learning rate may be too high/low")
            print("    3. Projection mode may be too simple")

        print()
        print("  Key Insight:")
        print("    The demo uses RANDOM tokens — real text embeddings would show")
        print("    stronger semantic structure and better alignment results.")

        return {
            "losses": losses,
            "before": before_metrics,
            "after": after_metrics,
        }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_alignment_training():
    """Run the full alignment training demo."""

    print("=" * 70)
    print("ExoBrain Feature Alignment Training (v0.4+)")
    print("=" * 70)
    print()
    print("Purpose: Train ShellProjection to align shell queries with brain KV")
    print()

    # Configuration
    NUM_LAYERS = 12
    NUM_HEADS = 8
    HEAD_DIM = 64
    HIDDEN_DIM = 256

    # Create components
    print("Step 1: Creating External Brain...")
    brain = MockExternalBrain(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )

    print("Step 2: Creating Shell Model...")
    shell = ShellQueryGenerator(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    print(f"  → Shell params: {shell.total_parameters / 1e6:.2f}M")

    print("Step 3: Initializing Alignment Trainer...")
    config = AlignmentConfig(
        loss_type="contrastive",
        lr=1e-3,
        epochs=50,
        batch_size=32,
        projection_mode="linear",
        log_every=10,
    )

    trainer = AlignmentTrainer(
        brain=brain,
        shell=shell,
        shell_hidden_dim=HEAD_DIM,
        brain_hidden_dim=HEAD_DIM,
        config=config,
    )

    print(f"  → Projection params: {trainer.projection.total_parameters / 1e3:.1f}K")

    print()
    print("Step 4: Running Alignment Training...")
    results = trainer.train()

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_alignment_training()