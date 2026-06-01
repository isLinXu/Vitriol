from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

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


def test_weight_inspector_refuses_unsafe_torch_load_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vitriol.viz import weight_inspector as wi

    calls: list[dict[str, object]] = []

    def _fake_torch_load(_path, **kwargs):
        calls.append(dict(kwargs))
        raise RuntimeError("weights_only blocked")

    monkeypatch.setattr(wi, "torch", SimpleNamespace(load=_fake_torch_load))

    with pytest.raises(ValueError, match="Unsafe legacy PyTorch pickle fallback is disabled"):
        wi._load_shard(tmp_path / "pytorch_model.bin", is_safetensors=False)

    assert calls == [{"map_location": "cpu", "weights_only": True}]


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


def test_weight_inspector_generate_viz_data_uses_selective_safetensors_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from vitriol.viz import weight_inspector as wi

    model_dir = tmp_path / "selective"
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
            "unused.extra.weight": {"dtype": "F32", "shape": [8, 8]},
        },
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "transformer.wte.weight": "model.safetensors",
                    "transformer.h.0.attn.c_attn.weight": "model.safetensors",
                    "transformer.h.0.mlp.c_fc.weight": "model.safetensors",
                    "transformer.h.1.attn.c_attn.weight": "model.safetensors",
                    "transformer.h.1.mlp.c_proj.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                    "unused.extra.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    accessed: list[str] = []

    class FakeHandle:
        def __init__(self, path: str, **_kwargs):
            self._path = Path(path)
            self._header = wi._read_safetensors_metadata(self._path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._header.keys())

        def get_tensor(self, name: str):
            accessed.append(name)
            meta = self._header[name]
            return torch.zeros(tuple(meta["shape"]), dtype=torch.float32)

    monkeypatch.setattr(wi, "_safetensors_safe_open", lambda path, **kwargs: FakeHandle(path, **kwargs))
    monkeypatch.setattr(
        wi,
        "_safetensors_load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full shard load should not be used")),
    )

    viz = wi.generate_viz_data(str(model_dir), max_layers=2)
    assert viz["weight_stats_available"] is True
    assert "unused.extra.weight" not in accessed
    assert "transformer.wte.weight" in accessed


def test_weight_inspector_generate_viz_data_uses_header_metadata_to_pick_shards_without_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from vitriol.viz import weight_inspector as wi

    model_dir = tmp_path / "header_only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "vocab_size": 1000,
                "n_embd": 64,
                "n_layer": 1,
                "n_head": 4,
            }
        ),
        encoding="utf-8",
    )

    _write_fake_safetensors(
        model_dir / "model-00001-of-00003.safetensors",
        {
            "unused.extra.weight": {"dtype": "F32", "shape": [8, 8]},
        },
    )
    _write_fake_safetensors(
        model_dir / "model-00002-of-00003.safetensors",
        {
            "unused.more.weight": {"dtype": "F32", "shape": [4, 4]},
        },
    )
    _write_fake_safetensors(
        model_dir / "model-00003-of-00003.safetensors",
        {
            "transformer.wte.weight": {"dtype": "F32", "shape": [1000, 64]},
            "transformer.h.0.attn.c_attn.weight": {"dtype": "F32", "shape": [64, 192]},
            "transformer.h.0.mlp.c_fc.weight": {"dtype": "F32", "shape": [64, 256]},
            "lm_head.weight": {"dtype": "F32", "shape": [1000, 64]},
        },
    )

    opened_shards: list[str] = []
    accessed: list[str] = []

    class FakeHandle:
        def __init__(self, path: str, **_kwargs):
            self._path = Path(path)
            opened_shards.append(self._path.name)
            self._header = wi._read_safetensors_metadata(self._path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._header.keys())

        def get_tensor(self, name: str):
            accessed.append(name)
            meta = self._header[name]
            return torch.zeros(tuple(meta["shape"]), dtype=torch.float32)

    monkeypatch.setattr(wi, "_safetensors_safe_open", lambda path, **kwargs: FakeHandle(path, **kwargs))
    monkeypatch.setattr(
        wi,
        "_safetensors_load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full shard load should not be used")),
    )

    viz = wi.generate_viz_data(str(model_dir), max_layers=1)
    assert viz["weight_stats_available"] is True
    assert opened_shards == ["model-00003-of-00003.safetensors"]
    assert "transformer.wte.weight" in accessed
    assert "lm_head.weight" in accessed
    assert "unused.extra.weight" not in accessed


def test_weight_inspector_marks_config_meta_as_meta_source(tmp_path: Path) -> None:
    model_dir = tmp_path / "meta_source"
    model_dir.mkdir()
    (model_dir / "config_meta.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "vocab_size": 1000,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 256,
            }
        ),
        encoding="utf-8",
    )

    viz = generate_viz_data(str(model_dir), max_layers=2)
    assert viz["config_source"] == "meta-config.json"


def test_weight_inspector_prefers_safetensors_without_bin_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vitriol.viz import weight_inspector as wi

    model_dir = tmp_path / "prefer_safetensors"
    model_dir.mkdir()
    safetensor_path = model_dir / "model.safetensors"
    safetensor_path.write_bytes(b"")

    original_glob = Path.glob

    def _glob(self: Path, pattern: str):
        if self == model_dir and pattern == "*.bin":
            raise AssertionError("bin glob should not be queried when safetensors are present")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _glob)

    files, is_safetensors = wi._list_weight_files(model_dir)
    assert is_safetensors is True
    assert files == [safetensor_path]
