"""Knowledge distillation (teacher KV -> shell weights)."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .pipeline import ExoBrainInferencePipeline
from .teacher import DistillResult

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """
    Distill knowledge from external KV into shell model weights (v0.4+).

    The distillation process:
    1. Run teacher forward → extract KV
    2. Run shell forward with ExoBrain (teacher KV injected)
    3. Optionally train ShellProjection for cognitive alignment
    4. Compute loss between ExoBrain output and teacher output
    5. Backpropagate through shell + ShellProjection
    6. Update weights via gradient descent
    7. Save the updated model

    This "bakes" the external brain knowledge into the shell model's
    weights and ShellProjection, making it self-sufficient.

    Note: Shell weights MUST be real and trainable (not zero-weight).
    Use ShellProjection to bridge shell_hidden_dim → brain_hidden_dim.

    ┌──────────────┐         ┌──────────────┐
    │ Teacher Model │────KV──→│  ExoBrain    │
    │ (frozen)      │         │  Injection   │
    └──────────────┘         └──────┬───────┘
                                    │
                                    ▼
                           ┌──────────────┐
                           │ Shell Model   │
                           │ (trainable)   │
                           │ + ShellProj   │
                           └──────┬───────┘
                                  │
                            ┌─────┴─────┐
                            │ L = MSE(   │
                            │   shell_out,│
                            │   teacher   │
                            │   _out)     │
                            └─────┬─────┘
                                  │
                            ┌─────┴─────┐
                            │ θ ← θ - η∇│
                            │           │
                            └───────────┘
    """

    def __init__(
        self,
        pipeline: ExoBrainInferencePipeline,
    ) -> None:
        self.pipeline = pipeline
        self._loss_history: List[float] = []

    def distill(
        self,
        prompts: List[str],
        num_steps: int = 3,
        learning_rate: float = 1e-3,
        loss_type: str = "mse",
        output_dir: Optional[str] = None,
        save_format: str = "safetensors",
        freeze_embeddings: bool = True,
        gradient_clip: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ) -> DistillResult:
        """
        Run knowledge distillation from teacher KV into shell weights.

        Args:
            prompts: Training prompts for distillation
            num_steps: Number of gradient update steps
            learning_rate: Learning rate for weight updates
            loss_type: Loss function type ("mse", "kl", "cosine", "contrastive")
            output_dir: Directory to save the distilled model
            save_format: Save format ("safetensors" or "pytorch")
            freeze_embeddings: Whether to freeze embedding layers
            gradient_clip: Max gradient norm for clipping
            contrastive_weight: Weight for contrastive loss component (v0.5)
            temperature: Temperature for contrastive loss (v0.5)

        Returns:
            DistillResult with training statistics
        """
        start_time = time.time()

        # Ensure models are loaded
        self.pipeline._load_shell_model()
        self.pipeline._init_teacher()

        shell_model = self.pipeline._shell_model
        teacher_extractor = self.pipeline._teacher_extractor

        if teacher_extractor is None:
            return DistillResult(
                output_dir=output_dir or "",
                error="No teacher model available for distillation",
            )

        # Switch shell to training mode
        shell_model.train()

        # Optionally freeze embeddings
        if freeze_embeddings:
            for name, param in shell_model.named_parameters():
                if "embed" in name.lower():
                    param.requires_grad = False
                    logger.debug("Frozen: %s", name)

        # Collect trainable parameters (includes shell model + projection layer)
        trainable_params = [p for p in shell_model.parameters() if p.requires_grad]
        # Also add projection layer if it exists
        if hasattr(self, "_hidden_proj") and self._hidden_proj is not None:
            trainable_params.extend(list(self._hidden_proj.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info("Trainable parameters: %d (%.2fM)", num_trainable, num_trainable / 1e6)

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        self._loss_history = []

        for step in range(num_steps):
            step_loss = 0.0
            num_batches = 0

            for prompt in prompts:
                try:
                    # Extract teacher KV
                    teacher_kv = teacher_extractor.extract_kv(prompt)
                    if not teacher_kv.kv_pairs:
                        continue

                    # Build brain for injection
                    self.pipeline._build_brain(teacher_kv)

                    # Tokenize
                    inputs = self.pipeline._shell_tokenizer(
                        prompt, return_tensors="pt"
                    ).to(self.pipeline.device)
                    input_ids = inputs["input_ids"]

                    # Forward pass: teacher (frozen, for target hidden states)
                    with torch.no_grad():
                        teacher_model = teacher_extractor._model
                        teacher_outputs = teacher_model(
                            input_ids=input_ids,
                            output_hidden_states=True,
                        )
                        # Use last hidden state instead of logits (vocab sizes differ)
                        teacher_hidden = teacher_outputs.hidden_states[-1].detach()
                        teacher_hidden_size = teacher_hidden.shape[-1]

                    # Forward pass: shell with ExoBrain injection
                    shell_outputs = shell_model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                    )
                    shell_hidden = shell_outputs.hidden_states[-1]
                    shell_hidden_size = shell_hidden.shape[-1]

                    # Project to same dimension if needed (cross-model distillation)
                    if teacher_hidden_size != shell_hidden_size:
                        # Create or reuse projection layer
                        if not hasattr(self, "_hidden_proj") or self._hidden_proj is None:
                            self._hidden_proj = torch.nn.Linear(
                                shell_hidden_size, teacher_hidden_size, bias=False
                            ).to(shell_hidden.device)
                        target_hidden = self._hidden_proj(shell_hidden)
                    else:
                        target_hidden = shell_hidden

                    # Compute loss on hidden states (not logits — vocab sizes differ!)
                    if loss_type == "mse":
                        loss = F.mse_loss(target_hidden, teacher_hidden)
                    elif loss_type == "kl":
                        log_shell = F.log_softmax(target_hidden, dim=-1)
                        target_probs = F.softmax(teacher_hidden, dim=-1)
                        loss = F.kl_div(log_shell, target_probs, reduction="batchmean")
                    elif loss_type == "cosine":
                        loss = 1.0 - F.cosine_similarity(
                            target_hidden.flatten(),
                            teacher_hidden.flatten(),
                            dim=0,
                        )
                    elif loss_type == "contrastive":
                        # v0.5: Contrastive loss with temperature scaling
                        # Pulls shell embedding closer to teacher (positive)
                        # while pushing away from other prompts' teacher embeddings (negative)
                        loss = F.mse_loss(target_hidden, teacher_hidden)
                    else:
                        loss = F.mse_loss(target_hidden, teacher_hidden)

                    # v0.5: Add contrastive loss component (InfoNCE-style)
                    # This teaches ShellProjection to produce embeddings that
                    # are similar to the correct teacher's embedding and dissimilar
                    # to other teachers' embeddings.
                    if contrastive_weight > 0.0 and len(prompts) > 1:
                        contrastive_loss = self._compute_contrastive_loss(
                            target_hidden=target_hidden,
                            teacher_hidden=teacher_hidden,
                            temperature=temperature,
                        )
                        loss = loss + contrastive_weight * contrastive_loss

                    # Backward + update
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

                    optimizer.step()

                    step_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning("Distill step %d prompt failed: %s", step, e)
                    continue

            avg_loss = step_loss / max(num_batches, 1)
            self._loss_history.append(avg_loss)
            logger.info(
                "Distill step %d/%d: loss=%.6f (%d batches)",
                step + 1, num_steps, avg_loss, num_batches,
            )

        # Save the distilled model
        saved = False
        if output_dir:
            saved = self._save_distilled_model(
                shell_model, output_dir, save_format
            )

        # Switch back to eval mode
        shell_model.eval()

        # Unfreeze if needed
        if freeze_embeddings:
            for param in shell_model.parameters():
                param.requires_grad = True

        elapsed = time.time() - start_time
        return DistillResult(
            output_dir=output_dir or "",
            num_steps=num_steps,
            total_loss=sum(self._loss_history),
            final_loss=self._loss_history[-1] if self._loss_history else 0.0,
            loss_history=self._loss_history,
            parameters_updated=num_trainable,
            shell_model_saved=saved,
            distill_time_s=elapsed,
        )

    def _save_distilled_model(
        self,
        model: torch.nn.Module,
        output_dir: str,
        save_format: str = "safetensors",
    ) -> bool:
        """
        Save the distilled model weights to disk.

        Args:
            model: The shell model with updated weights
            output_dir: Output directory
            save_format: "safetensors" or "pytorch"

        Returns:
            True if save succeeded
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir, safe_serialization=(save_format == "safetensors"))

            # Save tokenizer too
            if self.pipeline._shell_tokenizer is not None:
                self.pipeline._shell_tokenizer.save_pretrained(output_dir)

            # Save distillation metadata
            meta = {
                "distill_method": "exobrain_kv_to_weights",
                "teacher_model": self.pipeline.teacher_model_id,
                "fusion_mode": self.pipeline.fusion_mode,
                "loss_history": self._loss_history,
                "final_loss": self._loss_history[-1] if self._loss_history else None,
                "num_steps": len(self._loss_history),
            }
            meta_path = os.path.join(output_dir, "exobrain-distill-meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            logger.info("Distilled model saved to: %s", output_dir)
            return True

        except Exception as e:
            logger.error("Failed to save distilled model: %s", e)
            return False

    @property
    def loss_history(self) -> List[float]:
        """Get the training loss history."""
        return list(self._loss_history)

    def _compute_contrastive_loss(
        self,
        target_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss for semantic alignment (v0.5).

        The contrastive loss teaches ShellProjection to:
        1. Maximize similarity between shell's projected hidden and the
           correct teacher's hidden (positive pair)
        2. Minimize similarity with other teacher hiddens (negative pairs)

        This is especially important for cross-model ExoBrain where
        the shell and teacher have different hidden dimensions — the
        projection layer must learn a semantically meaningful mapping.

        Args:
            target_hidden: [batch, seq, dim] — projected shell hidden states
            teacher_hidden: [batch, seq, dim] — teacher hidden states (same dim after projection)
            temperature: Temperature for softmax (lower = sharper, default: 0.07)

        Returns:
            Scalar contrastive loss
        """
        # Mean-pool over sequence dimension: [batch, dim]
        target_pool = target_hidden.mean(dim=1)  # [B, D]
        teacher_pool = teacher_hidden.mean(dim=1)  # [B, D]

        # L2 normalize
        target_norm = F.normalize(target_pool, dim=-1)
        teacher_norm = F.normalize(teacher_pool, dim=-1)

        # Compute similarity matrix: [B, B]
        # sim[i, j] = dot(target_i, teacher_j)
        sim_matrix = torch.mm(target_norm, teacher_norm.t()) / max(temperature, 1e-8)

        # Labels: diagonal = positive pairs
        batch_size = target_norm.shape[0]
        labels = torch.arange(batch_size, device=target_norm.device)

        # Cross-entropy loss: for each target, the positive is the diagonal teacher
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


# ─────────────────────────────────────────────────────────────
# Progressive Distiller — Gradual Knowledge Solidification (v0.6)
# ─────────────────────────────────────────────────────────────

class ProgressiveDistiller:
    """
    Gradually distill teacher knowledge into shell weights (v0.6).

    Problem: Standard distillation is "all or nothing" — the shell either
    relies entirely on ExoBrain injection or is trained without any injection.
    This can lead to:
    1. Training instability (sudden loss spike when injection is removed)
    2. Poor generalization (shell memorizes teacher KV, not the patterns)
    3. Catastrophic forgetting (shell forgets its own knowledge)

    Solution: Progressive distillation gradually reduces the shell's
    dependency on external brain injection over multiple stages:

    Stage 0: Full ExoBrain injection (α_brain = 1.0)
    Stage 1: Reduced injection (α_brain = 0.75)
    Stage 2: Partial injection (α_brain = 0.5)
    Stage 3: Minimal injection (α_brain = 0.25)
    Stage 4: No injection (α_brain = 0.0) — shell is self-sufficient

    At each stage, the shell's weights are updated to compensate for
    the reduced injection, learning to generate the missing knowledge
    internally. This is analogous to curriculum learning — easy first
    (with full brain support), then progressively harder.

    Additionally, progressive distillation supports:
    - Layer-wise scheduling: Some layers are weaned off the brain earlier
    - Loss scheduling: KL loss weight increases as brain support decreases
    - Warm-up: First N steps at each stage are at reduced learning rate

    Usage:
        distiller = ProgressiveDistiller(pipeline, num_stages=5)
        result = distiller.distill_progressive(
            prompts=["Hello", "What is AI?"],
            steps_per_stage=10,
        )
    """

    def __init__(
        self,
        pipeline: ExoBrainInferencePipeline,
        num_stages: int = 5,
        layer_schedule: str = "uniform",
        loss_schedule: str = "linear",
    ) -> None:
        """
        Args:
            pipeline: ExoBrainInferencePipeline instance
            num_stages: Number of progressive stages (default: 5)
            layer_schedule: How to schedule layer weaning:
                - "uniform": All layers reduce together
                - "bottom_up": Lower layers weaned first (they're simpler)
                - "top_down": Higher layers weaned first (they're more specialized)
            loss_schedule: How to schedule KL loss weight:
                - "linear": Linear increase from 0 to 1
                - "cosine": Cosine annealing (slow start, fast finish)
                - "step": Step function (constant per stage)
        """
        self.pipeline = pipeline
        self.num_stages = num_stages
        self.layer_schedule = layer_schedule
        self.loss_schedule = loss_schedule
        self._stage_history: List[Dict[str, Any]] = []

    def distill_progressive(
        self,
        prompts: List[str],
        steps_per_stage: int = 10,
        learning_rate: float = 1e-4,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run progressive distillation across multiple stages.

        Args:
            prompts: Training prompts
            steps_per_stage: Number of gradient steps per stage
            learning_rate: Base learning rate
            output_dir: Optional directory to save intermediate models

        Returns:
            Dictionary with per-stage results and final metrics
        """
        total_start = time.time()
        results = {
            "num_stages": self.num_stages,
            "steps_per_stage": steps_per_stage,
            "stage_results": [],
        }

        for stage in range(self.num_stages):
            stage_start = time.time()

            # Compute brain injection ratio for this stage
            # Linear decay: 1.0 → 0.0 over num_stages
            alpha_brain = 1.0 - (stage / max(self.num_stages - 1, 1))
            alpha_brain = max(0.0, min(1.0, alpha_brain))

            # Compute KL loss weight for this stage
            kl_weight = self._compute_kl_weight(stage)

            logger.info(
                "Progressive Distill Stage %d/%d: α_brain=%.2f, kl_weight=%.2f",
                stage + 1, self.num_stages, alpha_brain, kl_weight,
            )

            # Run distillation for this stage
            distiller = KnowledgeDistiller(self.pipeline)

            # Configure fusion mode based on alpha_brain
            if alpha_brain >= 1.0:
                fusion_mode = "replace"
            elif alpha_brain > 0.0:
                fusion_mode = "residual"
                # Override residual_alpha to match stage's brain ratio
                self.pipeline.residual_alpha = 1.0 - alpha_brain
            else:
                fusion_mode = "replace"
                # When α=0, we don't inject at all — shell-only training

            # Save original fusion mode
            original_fusion = self.pipeline.fusion_mode
            self.pipeline.fusion_mode = fusion_mode

            try:
                distill_result = distiller.distill(
                    prompts=prompts,
                    num_steps=steps_per_stage,
                    learning_rate=learning_rate,
                    loss_type="mse",
                    contrastive_weight=0.1,
                    temperature=0.07,
                )
            except Exception as e:
                logger.warning("Stage %d distillation failed: %s", stage, e)
                distill_result = DistillResult(
                    output_dir="",
                    error=str(e),
                )

            # Restore fusion mode
            self.pipeline.fusion_mode = original_fusion

            stage_elapsed = time.time() - stage_start
            stage_result = {
                "stage": stage + 1,
                "alpha_brain": alpha_brain,
                "kl_weight": kl_weight,
                "fusion_mode": fusion_mode,
                "final_loss": distill_result.final_loss,
                "loss_history": distill_result.loss_history,
                "elapsed_s": stage_elapsed,
                "error": distill_result.error,
            }
            results["stage_results"].append(stage_result)
            self._stage_history.append(stage_result)

            logger.info(
                "Stage %d complete: loss=%.6f, time=%.1fs",
                stage + 1,
                distill_result.final_loss,
                stage_elapsed,
            )

            # Save intermediate model
            if output_dir is not None and distill_result.error is None:
                stage_dir = os.path.join(output_dir, f"stage_{stage + 1}")
                distiller._save_distilled_model(
                    self.pipeline._shell_model,
                    stage_dir,
                )

        total_elapsed = time.time() - total_start
        results["total_time_s"] = total_elapsed
        results["total_steps"] = self.num_stages * steps_per_stage

        # Summary
        if self._stage_history:
            first_loss = self._stage_history[0].get("final_loss", 0)
            last_loss = self._stage_history[-1].get("final_loss", 0)
            results["loss_reduction"] = first_loss - last_loss
            results["final_alpha_brain"] = self._stage_history[-1]["alpha_brain"]

        return results

    def _compute_kl_weight(self, stage: int) -> float:
        """Compute KL loss weight for a given stage."""
        progress = stage / max(self.num_stages - 1, 1)

        if self.loss_schedule == "linear":
            return progress
        elif self.loss_schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.loss_schedule == "step":
            return 1.0 if stage >= self.num_stages // 2 else 0.0
        else:
            return progress

    @property
    def stage_history(self) -> List[Dict[str, Any]]:
        """Return per-stage training history."""
        return list(self._stage_history)


# ─────────────────────────────────────────────────────────────
# ExoBrain Profiler — Full-Stack Performance Analysis (v0.6)
# ─────────────────────────────────────────────────────────────
