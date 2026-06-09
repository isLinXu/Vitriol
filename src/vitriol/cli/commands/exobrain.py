"""
ExoBrain CLI commands: `vitriol exobrain infer` and `vitriol exobrain distill`.

Usage:
    # Inference verification with external brain
    vitriol exobrain infer ./shell-model --teacher Qwen/Qwen2.5-0.5B --prompt "Hello"

    # Knowledge distillation (bake brain into weights)
    vitriol exobrain distill ./shell-model --teacher Qwen/Qwen2.5-0.5B \
        --output ./distilled-model --steps 3

    # Quick test without teacher (shell model only)
    vitriol exobrain infer ./shell-model --prompt "Test"
"""

import json

import click


@click.group("exobrain")
def exobrain_group() -> None:
    """ExoBrain: External brain inference and knowledge distillation for shell models."""
    pass


@exobrain_group.command("infer")
@click.argument("shell_model_path")
@click.option("--teacher", "teacher_model_id", type=str, default=None,
              help="Teacher model ID (HuggingFace) for KV extraction")
@click.option("--prompt", type=str, required=True, help="Input prompt text")
@click.option("--prompt-file", type=click.Path(exists=True), help="Read prompt from file")
@click.option("--fusion-mode", type=click.Choice(["replace", "residual", "gated"]),
              default="replace", show_default=True, help="ExoBrain fusion mode")
@click.option("--max-new-tokens", type=int, default=64, show_default=True,
              help="Maximum tokens to generate")
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
@click.option("--dtype", type=click.Choice(["float32", "float16", "bfloat16"]),
              default="float32", show_default=True, help="Model dtype")
@click.option("--retrieval-top-k", type=int, default=5, show_default=True,
              help="Number of top-K KV pairs to retrieve")
@click.option("--local-files-only", is_flag=True, default=False,
              help="Only use local cached files, no network access")
@click.option("--head-dim-projection", type=click.Choice(["pad_or_truncate", "learned"]),
              default="pad_or_truncate", show_default=True,
              help="Head-dim projection mode for cross-model inference")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]),
              default="text", show_default=True, help="Output format")
@click.pass_context
def infer_cmd(
    ctx,
    shell_model_path,
    teacher_model_id,
    prompt,
    prompt_file,
    fusion_mode,
    max_new_tokens,
    device,
    dtype,
    retrieval_top_k,
    local_files_only,
    head_dim_projection,
    fmt,
) -> None:
    """Run ExoBrain inference on a shell model with optional teacher KV injection."""
    from ...kv.exobrain_inference import ExoBrainInferencePipeline

    # Resolve prompt
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as f:
            prompt = f.read()

    # Map dtype string to torch dtype
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False)) if ctx.obj else False

    pipeline = ExoBrainInferencePipeline(
        shell_model_path=shell_model_path,
        teacher_model_id=teacher_model_id,
        fusion_mode=fusion_mode,
        device=device,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        retrieval_top_k=retrieval_top_k,
        max_new_tokens=max_new_tokens,
        head_dim_projection=head_dim_projection,
    )

    result = pipeline.infer(prompt)

    if fmt == "json":
        output = {
            "prompt": result.prompt,
            "generated_text": result.generated_text,
            "generated_tokens": result.generated_tokens,
            "prompt_tokens": result.prompt_tokens,
            "inference_time_s": round(result.inference_time_s, 4),
            "tokens_per_second": round(result.tokens_per_second, 2),
            "fusion_mode": result.fusion_mode,
            "brain_hit_rate": round(result.brain_hit_rate, 4),
            "device": result.device,
            "error": result.error,
        }
        click.echo(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if result.error:
            click.echo(f"Error: {result.error}", err=True)
        else:
            click.echo(result.generated_text)
        if result.generated_tokens > 0:
            click.echo(
                f"\n[{result.generated_tokens} tokens, "
                f"{result.tokens_per_second:.1f} tok/s, "
                f"brain hit rate: {result.brain_hit_rate:.1%}]",
                err=True,
            )


@exobrain_group.command("distill")
@click.argument("shell_model_path")
@click.option("--teacher", "teacher_model_id", type=str, required=True,
              help="Teacher model ID (HuggingFace) for KV extraction")
@click.option("--output", "output_dir", type=click.Path(), required=True,
              help="Output directory for distilled model")
@click.option("--prompts", type=str, multiple=True,
              help="Training prompts (can specify multiple)")
@click.option("--prompts-file", type=click.Path(exists=True),
              help="Read training prompts from file (one per line)")
@click.option("--steps", "num_steps", type=int, default=3, show_default=True,
              help="Number of distillation steps")
@click.option("--lr", "learning_rate", type=float, default=1e-3, show_default=True,
              help="Learning rate for weight updates")
@click.option("--loss", "loss_type", type=click.Choice(["mse", "kl", "cosine"]),
              default="mse", show_default=True, help="Loss function type")
@click.option("--fusion-mode", type=click.Choice(["replace", "residual", "gated"]),
              default="replace", show_default=True, help="ExoBrain fusion mode")
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
@click.option("--dtype", type=click.Choice(["float32", "float16", "bfloat16"]),
              default="float32", show_default=True, help="Model dtype")
@click.option("--save-format", type=click.Choice(["safetensors", "pytorch"]),
              default="safetensors", show_default=True, help="Model save format")
@click.option("--gradient-clip", type=float, default=1.0, show_default=True,
              help="Max gradient norm for clipping")
@click.option("--head-dim-projection", type=click.Choice(["pad_or_truncate", "learned"]),
              default="pad_or_truncate", show_default=True,
              help="Head-dim projection mode for cross-model distillation")
@click.pass_context
def distill_cmd(
    ctx,
    shell_model_path,
    teacher_model_id,
    output_dir,
    prompts,
    prompts_file,
    num_steps,
    learning_rate,
    loss_type,
    fusion_mode,
    device,
    dtype,
    save_format,
    gradient_clip,
    head_dim_projection,
) -> None:
    """Distill teacher knowledge into shell model weights via ExoBrain KV injection."""
    from ...kv.exobrain_inference import ExoBrainInferencePipeline, KnowledgeDistiller

    # Resolve prompts
    prompt_list = list(prompts)
    if prompts_file:
        with open(prompts_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompt_list.append(line)

    if not prompt_list:
        prompt_list = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
        ]
        click.echo(f"No prompts specified, using {len(prompt_list)} default prompts.", err=True)

    # Map dtype
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    trust_remote_code = bool(ctx.obj.get("trust_remote_code", False)) if ctx.obj else False

    # Create pipeline
    pipeline = ExoBrainInferencePipeline(
        shell_model_path=shell_model_path,
        teacher_model_id=teacher_model_id,
        fusion_mode=fusion_mode,
        device=device,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        head_dim_projection=head_dim_projection,
    )

    # Create distiller
    distiller = KnowledgeDistiller(pipeline=pipeline)

    click.echo(f"Starting distillation: {num_steps} steps, lr={learning_rate}, loss={loss_type}", err=True)
    click.echo(f"Teacher: {teacher_model_id}", err=True)
    click.echo(f"Prompts: {len(prompt_list)}", err=True)

    result = distiller.distill(
        prompts=prompt_list,
        num_steps=num_steps,
        learning_rate=learning_rate,
        loss_type=loss_type,
        output_dir=output_dir,
        save_format=save_format,
        gradient_clip=gradient_clip,
    )

    # Report results
    click.echo("\nDistillation complete:", err=True)
    click.echo(f"  Steps: {result.num_steps}", err=True)
    click.echo(f"  Final loss: {result.final_loss:.6f}", err=True)
    click.echo(f"  Parameters updated: {result.parameters_updated:,}", err=True)
    click.echo(f"  Time: {result.distill_time_s:.1f}s", err=True)
    click.echo(f"  Model saved: {'✓' if result.shell_model_saved else '✗'}", err=True)

    if result.loss_history:
        click.echo(f"  Loss history: {[round(loss, 6) for loss in result.loss_history]}", err=True)

    if result.shell_model_saved:
        click.echo(f"\nDistilled model saved to: {output_dir}", err=True)
    else:
        click.echo("\nWarning: Model save failed!", err=True)
