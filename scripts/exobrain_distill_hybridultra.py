"""
ExoBrain Knowledge Distillation on HybridUltra (kaiming init) model.
This script:
1. Loads the trainable HybridUltra model
2. Loads the teacher model
3. Runs distillation to bake external knowledge into shell weights
"""

import torch
import os
import shutil
from vitriol.kv.exobrain_inference import ExoBrainInferencePipeline, KnowledgeDistiller

# Paths
SHELL_MODEL = "output/tinyllama-hybrid-ultra-test"  # HybridUltra with kaiming init
TEACHER_MODEL = "output/qwen2.5-0.5b-ultra-dummy"  # Qwen teacher
OUTPUT_DIR = "output/tinyllama-distilled-hybridultra"

print("=" * 70)
print("ExoBrain Knowledge Distillation on HybridUltra (kaiming init)")
print("=" * 70)

# Clean output
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Detect device
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
    print(f"Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"Using CUDA")
else:
    print(f"Using CPU")

# Create ExoBrain Inference Pipeline
print(f"\n1. Creating ExoBrain Inference Pipeline...")
print(f"   Shell model: {SHELL_MODEL}")
print(f"   Teacher model: {TEACHER_MODEL}")

pipeline = ExoBrainInferencePipeline(
    shell_model_path=SHELL_MODEL,
    teacher_model_id="Qwen/Qwen2.5-0.5B",
    fusion_mode="replace",
    device=device,
    dtype=torch.float32,
    trust_remote_code=True,
    local_files_only=True,
    retrieval_top_k=5,
    head_dim_projection="pad_or_truncate",
)

# Create Knowledge Distiller
print(f"\n2. Creating Knowledge Distiller...")
distiller = KnowledgeDistiller(pipeline)

# Run distillation
print(f"\n3. Running distillation...")
prompts = [
    "The capital of France is",
    "Python is a programming language that",
    "Machine learning is a subset of",
]

result = distiller.distill(
    prompts=prompts,
    num_steps=5,
    learning_rate=1e-4,
    loss_type="mse",
    output_dir=OUTPUT_DIR,
    save_format="safetensors",
    freeze_embeddings=False,  # Train embeddings too for better adaptation
    gradient_clip=1.0,
)

print(f"\n" + "=" * 70)
print("DISTILLATION RESULTS")
print("=" * 70)
print(f"Steps: {result['steps']}")
print(f"Final loss: {result['final_loss']:.6f}")
print(f"Initial loss: {result['initial_loss']:.6f}")
print(f"Loss reduction: {result['initial_loss'] - result['final_loss']:.6f}")
print(f"Output saved to: {result['output_path']}")

# Verify output
if os.path.exists(result['output_path']):
    from safetensors.torch import load_file
    state_dict = load_file(result['output_path'])
    print(f"\nSaved model has {len(state_dict)} tensors")
    total_params = sum(v.numel() for v in state_dict.values())
    non_zero = sum((v.abs() > 1e-6).sum().item() for v in state_dict.values())
    print(f"Total params: {total_params:,}, Non-zero: {non_zero:,} ({100*non_zero/total_params:.1f}%)")