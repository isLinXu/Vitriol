from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import click

from vitriol.telemetry.run_context import new_run_id

logger = logging.getLogger(__name__)

def _token_ids_to_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    # `convert_ids_to_tokens` handles special tokens more faithfully than decode().
    # It may return strings with Ġ/▁ markers depending on tokenizer type; that's ok for trace replay.
    return [str(t) for t in tokenizer.convert_ids_to_tokens(token_ids)]


def _build_trace_v1(
    *,
    run_id: str,
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    prompt_token_ids: list[int],
    prompt_tokens: list[str],
    generated_token_ids: list[int],
    generated_tokens: list[str],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    # Minimal schema for tests + replay: keep it stable and small.
    return {
        "schema_version": "trace.v1",
        "run_id": str(run_id),
        "model_path": model_path,
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "attention_topk": {
            "enabled": True,
            "k": 12,
            "note": "Mean over heads; top-k per layer for last target position.",
        },
        "tokens": {
            "prompt_token_ids": [int(x) for x in prompt_token_ids],
            "prompt_tokens": list(prompt_tokens),
            "generated_token_ids": [int(x) for x in generated_token_ids],
            "generated_tokens": list(generated_tokens),
        },
        "events": events,
    }


def _try_register_llama_hooks(model: Any, recorder: dict[str, Any]) -> list[Any]:
    """Register forward hooks for Llama-like models and record execution nodes."""
    handles = []
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is None:
        return []

    def _mk(node_id: str):
        def _hook(_m, _inp, _out):
            recorder["nodes"].append(node_id)
        return _hook

    # Per layer: norm1 / attn(qkv,o) / norm2 / ffn(gate,up,down)
    for i, layer in enumerate(layers):
        norm1 = getattr(layer, "input_layernorm", None)
        if norm1 is not None and hasattr(norm1, "register_forward_hook"):
            handles.append(norm1.register_forward_hook(_mk(f"block:{i}:norm1")))

        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            # Submodules (best-effort)
            q = getattr(attn, "q_proj", None)
            k = getattr(attn, "k_proj", None)
            v = getattr(attn, "v_proj", None)
            o = getattr(attn, "o_proj", None)
            if q is not None and hasattr(q, "register_forward_hook"):
                handles.append(q.register_forward_hook(_mk(f"block:{i}:attn:q_proj")))
            if k is not None and hasattr(k, "register_forward_hook"):
                handles.append(k.register_forward_hook(_mk(f"block:{i}:attn:k_proj")))
            if v is not None and hasattr(v, "register_forward_hook"):
                handles.append(v.register_forward_hook(_mk(f"block:{i}:attn:v_proj")))
            if o is not None and hasattr(o, "register_forward_hook"):
                handles.append(o.register_forward_hook(_mk(f"block:{i}:attn:o_proj")))

            if hasattr(attn, "register_forward_hook"):
                handles.append(attn.register_forward_hook(_mk(f"block:{i}:attn")))

        norm2 = getattr(layer, "post_attention_layernorm", None)
        if norm2 is not None and hasattr(norm2, "register_forward_hook"):
            handles.append(norm2.register_forward_hook(_mk(f"block:{i}:norm2")))

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            gate = getattr(mlp, "gate_proj", None)
            up = getattr(mlp, "up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if gate is not None and hasattr(gate, "register_forward_hook"):
                handles.append(gate.register_forward_hook(_mk(f"block:{i}:ffn:gate_proj")))
            if up is not None and hasattr(up, "register_forward_hook"):
                handles.append(up.register_forward_hook(_mk(f"block:{i}:ffn:up_proj")))
            if down is not None and hasattr(down, "register_forward_hook"):
                handles.append(down.register_forward_hook(_mk(f"block:{i}:ffn:down_proj")))

            if hasattr(mlp, "register_forward_hook"):
                # Keep coarse node for compatibility with viz (will be normalized to ffn)
                handles.append(mlp.register_forward_hook(_mk(f"block:{i}:mlp")))

    # lm_head
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "register_forward_hook"):
        handles.append(lm_head.register_forward_hook(_mk("lm_head")))

    return handles


def _extract_attention_topk(
    *,
    attentions: Any,
    src_len: int,
    token_global_indices: list[int],
    topk: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract per-layer attention top-k from transformers attentions (mean over heads).

    Returns a dict:
      key: 'block:{i}:attn'
      value: [{'src': <token_global_index>, 'w': <float>}, ...]
    """
    if attentions is None:
        return {}
    if not isinstance(attentions, (list, tuple)) or not attentions:
        return {}

    try:
        import torch
    except Exception:
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    k = max(1, int(topk))

    for layer_idx, layer_attn in enumerate(attentions):
        # expected: (batch, heads, tgt, src)
        if layer_attn is None:
            continue
        try:
            t = layer_attn
            if not isinstance(t, torch.Tensor):
                continue
            if t.dim() != 4 or t.shape[0] < 1:
                continue
            # choose last target position
            w = t[0, :, -1, :src_len]  # (heads, src)
            if w.numel() == 0:
                continue
            w = w.float().mean(dim=0)  # (src,)
            # topk
            kk = min(k, w.shape[0])
            vals, idxs = torch.topk(w, kk)
            items: list[dict[str, Any]] = []
            for v, ix in zip(vals.tolist(), idxs.tolist()):
                if 0 <= int(ix) < len(token_global_indices):
                    items.append({"src": int(token_global_indices[int(ix)]), "w": float(v)})
            out[f"block:{layer_idx}:attn"] = items
        except Exception:
            continue

    return out


def _extract_attention_histogram_per_layer(
    *,
    attentions: Any,
    src_len: int,
    bins: int = 32,
) -> dict[str, Any]:
    """
    Compute a lightweight binned histogram of the full distribution (mean over heads, last target position), per layer.

    Returns:
      {
        "block:{i}:attn": {"bins": <int>, "values": [<float> * bins]},
        ...
      }
    """
    b = max(4, int(bins))
    src_len = max(0, int(src_len))
    if src_len == 0:
        return {}

    if attentions is None:
        return {}
    if not isinstance(attentions, (list, tuple)) or not attentions:
        return {}

    try:
        import torch
    except Exception:
        return {}

    out: dict[str, Any] = {}

    for layer_idx, layer_attn in enumerate(attentions):
        if not isinstance(layer_attn, torch.Tensor) or layer_attn.dim() != 4 or layer_attn.shape[0] < 1:
            continue
        w = layer_attn[0, :, -1, :src_len]  # (heads, src)
        if w.numel() == 0:
            continue
        w = w.float().mean(dim=0)  # (src,)

        values = [0.0 for _ in range(b)]
        for i in range(src_len):
            bucket = min(b - 1, int(i * b / src_len))
            values[bucket] += float(w[i].item())
        out[f"block:{layer_idx}:attn"] = {"bins": b, "values": values}

    return out


@click.command(name="trace")
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path, dir_okay=True, file_okay=True),
    default=Path("output/tinyllama-hybrid-ultra-test"),
    show_default=True,
    help="Local model directory (path passed to transformers.from_pretrained).",
)
@click.option("--prompt", type=str, default="hello", show_default=True, help="Input prompt.")
@click.option(
    "--max-new-tokens",
    type=int,
    default=8,
    show_default=True,
    help="generate() max_new_tokens.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    default=Path("trace.json"),
    show_default=True,
    help="Output trace.json path.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Inference device: cpu/cuda/auto.",
)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=None,
    help="Whether to allow executing custom code from the model repository (trust_remote_code).",
)
@click.pass_context
def trace(
    ctx: click.Context,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
    out: Path,
    device: str,
    trust_remote_code: Optional[bool],
) -> Any:
    """Run a single offline greedy decode and export a trace.v1 JSON for offline playback."""

    # Lazily import heavy deps to keep `vitriol.cli.main --help` lightweight.
    import torch

    from vitriol.utils.hf_loading import load_causallm, load_tokenizer

    effective_trust_remote_code = (
        bool(ctx.obj.get("trust_remote_code", False)) if getattr(ctx, "obj", None) else False
    )
    if trust_remote_code is not None:
        effective_trust_remote_code = bool(trust_remote_code)

    effective_device = device.strip().lower()
    if effective_device == "auto":
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"

    security = {
        "trust_remote_code": effective_trust_remote_code,
        "allow_network": False,
        "local_files_only": True,
    }
    tokenizer = load_tokenizer(
        str(model_path),
        security=security,
    )
    model = load_causallm(
        str(model_path),
        security=security,
        attn_implementation="eager",
    )
    model.eval()
    model.to(effective_device)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(effective_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(effective_device)

    prompt_ids = input_ids[0].tolist()
    prompt_tokens = _token_ids_to_tokens(tokenizer, prompt_ids)

    # Record module-level execution nodes triggered by forward hooks (best-effort).
    recorder: dict[str, Any] = {"nodes": []}
    handles = _try_register_llama_hooks(model, recorder)

    def _run_forward(*, step_input_ids, step_attention_mask=None, past_key_values=None, token_global_indices=None):
        recorder["nodes"] = ["embed"]
        out = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        node_path = list(recorder["nodes"])
        if not node_path or node_path[0] != "embed":
            node_path.insert(0, "embed")
        if node_path[-1] != "lm_head":
            node_path.append("lm_head")
        attention_topk = {}
        attention_histogram = {}
        if token_global_indices is not None:
            attention_topk = _extract_attention_topk(
                attentions=getattr(out, "attentions", None),
                src_len=len(token_global_indices),
                token_global_indices=list(token_global_indices),
                topk=12,
            )
            attention_histogram = _extract_attention_histogram_per_layer(
                attentions=getattr(out, "attentions", None),
                src_len=len(token_global_indices),
                bins=32,
            )
        return out, node_path, attention_topk, attention_histogram

    generated_ids: list[int] = []
    generated_tokens: list[str] = []
    events: list[dict[str, Any]] = []

    with torch.no_grad():
        # Prefill: full prompt
        prefill_global = list(range(len(prompt_ids)))
        out_prefill, node_path_prefill, attn_prefill, hist_prefill = _run_forward(
            step_input_ids=input_ids,
            step_attention_mask=attention_mask,
            past_key_values=None,
            token_global_indices=prefill_global,
        )

        # Prefill event (token-level: represent prefill using the last prompt token)
        if prompt_ids:
            events.append(
                {
                    "token_index": 0,
                    "token_global_index": len(prompt_ids) - 1,
                    "token_text": prompt_tokens[-1],
                    "phase": "prefill",
                    "attention_topk": attn_prefill,
                    "attention_histogram": hist_prefill,
                    "node_path": node_path_prefill,
                }
            )

        logits = out_prefill.logits
        past = getattr(out_prefill, "past_key_values", None)
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(next_token)
        generated_tokens.append(str(tokenizer.convert_ids_to_tokens([next_token])[0]))

        # Decode event 0: reuse the prefill forward path (produces logits for the first new token)
        if int(max_new_tokens) > 0:
            events.append(
                {
                    "token_index": 0,
                    "token_global_index": len(prompt_ids) + 0,
                    "token_text": generated_tokens[0],
                    "phase": "decode",
                    "attention_topk": attn_prefill,
                    "attention_histogram": hist_prefill,
                    "node_path": node_path_prefill,
                }
            )

        # Subsequent decode: feed the last generated token and reuse past_key_values
        for i in range(1, int(max_new_tokens)):
            step_ids = torch.tensor([[generated_ids[i - 1]]], device=input_ids.device)
            global_idx = list(range(len(prompt_ids) + i))
            out_step, node_path_step, attn_step, hist_step = _run_forward(
                step_input_ids=step_ids,
                step_attention_mask=None,
                past_key_values=past,
                token_global_indices=global_idx,
            )
            logits = out_step.logits
            past = getattr(out_step, "past_key_values", past)
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated_ids.append(next_token)
            generated_tokens.append(str(tokenizer.convert_ids_to_tokens([next_token])[0]))
            events.append(
                {
                    "token_index": int(i),
                    "token_global_index": len(prompt_ids) + int(i),
                    "token_text": generated_tokens[i],
                    "phase": "decode",
                    "attention_topk": attn_step,
                    "attention_histogram": hist_step,
                    "node_path": node_path_step,
                }
            )

    # Remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            logger.debug("Failed to remove handler hook")

    run_id = new_run_id()
    trace_payload = _build_trace_v1(
        run_id=run_id,
        model_path=str(model_path),
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        prompt_token_ids=prompt_ids,
        prompt_tokens=prompt_tokens,
        generated_token_ids=generated_ids[: int(max_new_tokens)],
        generated_tokens=generated_tokens[: int(max_new_tokens)],
        events=events,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    click.echo(f"trace saved: {out}")
