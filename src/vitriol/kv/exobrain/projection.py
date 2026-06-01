"""ShellProjection — cognitive alignment layer (shell hidden -> brain hidden)."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class ShellProjection(torch.nn.Module):
    """
    Thin cognitive alignment layer between shell model and external brain.

    PURPOSE:
    When the shell model (0.1B) and external brain (7B+) have different
    hidden dimensions, ShellProjection provides a lightweight learned mapping
    so that the shell's queries can semantically align with the brain's KV space.

    This is NOT optional — without cognitive alignment, the shell's query
    (e.g., 768-dim) cannot meaningfully attend to brain's KV (e.g., 4096-dim).

    MODES:
    - "linear": Single linear layer (thin, ~3M params for 768→4096)
    - "mlp": Linear → GELU → Linear (slightly more expressive)
    - "linear_ln": Linear → LayerNorm (stable for training)

    DESIGN PRINCIPLE:
    Keep it thin! The projection should be ~1% of shell model size.
    Weights are trained via Feature Alignment Distillation.

    EXAMPLE:
        shell_hidden_dim = 768    # 0.1B model
        brain_hidden_dim = 4096   # 7B model
        projection = ShellProjection(768, 4096, mode="linear")
        # ~3M parameters (768 * 4096 / projection_ratio)
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
            self.proj = torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias)
        elif mode == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(shell_hidden_dim, shell_hidden_dim, bias=bias),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
            )
        elif mode == "linear_ln":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
                torch.nn.LayerNorm(brain_hidden_dim),
                torch.nn.Dropout(p=dropout),
            )
        else:
            raise ValueError(f"ShellProjection: unknown mode '{mode}'. Use: linear, mlp, linear_ln")

        # Initialize with small std (near identity mapping is a good start)
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Initialize projection near identity for stable training start."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                # Small std — reduces initial distortion
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project shell hidden states → brain hidden space.

        Args:
            x: Shell hidden states [batch, seq, shell_hidden_dim]
               or [batch, heads, seq, head_dim]

        Returns:
            Projected tensor [batch, seq, brain_hidden_dim]
             or [batch, heads, seq, brain_head_dim]
        """
        original_shape = x.shape
        ndims = len(original_shape)

        # Normalize to [batch*heads, seq, dim] for projection
        if ndims == 4:
            # [B, H, S, D] → [B, H*S, D] → project → [B, H*S, brain_d]
            B, H, S, D = original_shape
            x = x.reshape(B, H * S, D)
            x = self.proj(x)
            # Return [B, H, S, brain_d]
            brain_d = self.brain_hidden_dim
            return x.reshape(B, H, S, brain_d)
        elif ndims == 3:
            # [B, S, D] → project → [B, S, brain_d]
            return self.proj(x)
        else:
            raise ValueError(
                f"ShellProjection: expected 3D [B,S,D] or 4D [B,H,S,D], got {ndims}D"
            )

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        """Convenience: project query tensor to brain space."""
        return self.forward(query)

    def project_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Convenience: project KV tensor to brain space."""
        return self.forward(kv)

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters in this projection."""
        return sum(p.numel() for p in self.parameters())

    @property
    def parameter_count_str(self) -> str:
        """Human-readable parameter count."""
        n = self.num_parameters
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)


# ─────────────────────────────────────────────────────────────
# Knowledge Sources — Protocols & Implementations
# ─────────────────────────────────────────────────────────────
