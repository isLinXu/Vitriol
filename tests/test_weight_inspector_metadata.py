from __future__ import annotations

import json
from pathlib import Path

from vitriol.viz.weight_inspector import generate_viz_data, inspect_weights


def _write_fake_safetensors(path: Path, tensors: dict[str, dict[str, object]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    dtype_bytes = {"F32": 4, "F16": 2}
    for name, meta in tensors.items():
        shape = list(meta["shape"])
        dtype = str(meta.get("dtype", "F32"))
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + (b"\0" * offset))


def test_weight_inspector_reads_safetensors_header_without_loading_torch(tmp_path: Path) -> None:
    model_dir = tmp_path / "gpt2ish"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 1000,
                "n_embd": 64,
                "n_layer": 2,
                "n_head": 4,
            }
        ),
        encoding="utf-8",
    )
    _write_fake_safetensors(
        model_dir / "model.safetensors",
        {
            "transformer.wte.weight": {"dtype": "F32", "shape": [1000, 64]},
            "transformer.h.0.attn.c_attn.weight": {"dtype": "F32", "shape": [64, 192]},
            "transformer.h.0.mlp.c_fc.weight": {"dtype": "F32", "shape": [64, 256]},
            "transformer.h.1.attn.c_attn.weight": {"dtype": "F32", "shape": [64, 192]},
            "transformer.h.1.mlp.c_proj.weight": {"dtype": "F32", "shape": [256, 64]},
            "lm_head.weight": {"dtype": "F32", "shape": [1000, 64]},
        },
    )

    inspected = inspect_weights(str(model_dir))
    assert inspected["total_tensors"] == 6
    assert inspected["layers"][0]["shape"] == [1000, 64]

    viz = generate_viz_data(str(model_dir), max_layers=4)
    assert viz["model_name"] == "gpt2"
    assert viz["num_layers"] == 2
    assert viz["layers"][0]["name"] == "transformer.wte.weight"
    block0 = next(layer for layer in viz["layers"] if layer.get("block_index") == 0)
    names = {item["name"] for item in block0["sub_layers"]}
    assert "transformer.h.0.attn.c_attn.weight" in names
    assert "transformer.h.0.mlp.c_fc.weight" in names
