from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from vitriol.bench import runner as bench_runner


def test_run_smoke_collects_kv_stats_after_residual_append(monkeypatch) -> None:
    class FakeTokenizer:
        pass

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(num_hidden_layers=1, layer_types=["full_attention"])

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_prefill_decode(model, tokenizer, prompt, device, max_new_tokens):
        if not getattr(DynamicCache.update, "_vitriol_cache_hook_patched", False):
            return {
                "prompt_tokens": 4,
                "decode_tokens": int(max_new_tokens),
                "decode_toks_per_s": 10.0,
                "gen_token_ids": [7],
                "_final_past_key_values": None,
            }

        handle = DynamicCache()
        handle.layer_types = ["full_attention"]
        key_prefill = torch.randn(1, 2, 4, 32, dtype=torch.float16)
        value_prefill = torch.randn(1, 2, 4, 32, dtype=torch.float16)
        key_decode = torch.randn(1, 2, 1, 32, dtype=torch.float16)
        value_decode = torch.randn(1, 2, 1, 32, dtype=torch.float16)

        handle.update(key_prefill, value_prefill, 0, {})
        handle.update(key_decode, value_decode, 0, {})

        return {
            "prompt_tokens": 4,
            "decode_tokens": int(max_new_tokens),
            "decode_toks_per_s": 10.0,
            "gen_token_ids": [7],
            "_final_past_key_values": handle,
        }

    monkeypatch.setattr(bench_runner, "hf_load_tokenizer", lambda model_id, security=None, **kwargs: FakeTokenizer())
    monkeypatch.setattr(
        bench_runner,
        "hf_load_causallm",
        lambda model_id, security=None, torch_dtype=None, device=None, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(bench_runner, "build_long_prompt", lambda tokenizer, min_tokens: "prompt")
    monkeypatch.setattr(bench_runner, "prefill_decode", fake_prefill_decode)
    monkeypatch.setattr(bench_runner, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench_runner, "sync", lambda device: None)

    result = bench_runner.run_smoke(
        model_id="demo/model",
        preset="aggressive",
        prompt_tokens=8,
        max_new_tokens=1,
        calib_new_tokens=1,
        search_max_n=1,
    )

    assert result["ok"] is True
    assert result["tuned_exact"] is True
    assert result["tuned_memory"]["estimated_kv_bytes"] > 0
    assert result["tuned_memory"]["layer_stats"][0]["seq_len"] == 5


def test_run_smoke_forwards_trust_remote_code_to_hf_loaders(monkeypatch) -> None:
    captured = {"tokenizer": None, "model": None}

    class FakeTokenizer:
        pass

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(num_hidden_layers=1, layer_types=["full_attention"])

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_tokenizer_from_pretrained(model_id, security=None, **kwargs):
        captured["tokenizer"] = {**(security or {}), **kwargs}
        return FakeTokenizer()

    def fake_model_from_pretrained(model_id, security=None, torch_dtype=None, device=None, **kwargs):
        captured["model"] = {"torch_dtype": torch_dtype, "device": device, **(security or {}), **kwargs}
        return FakeModel()

    monkeypatch.setattr(bench_runner, "hf_load_tokenizer", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(bench_runner, "hf_load_causallm", fake_model_from_pretrained)
    monkeypatch.setattr(bench_runner, "build_long_prompt", lambda tokenizer, min_tokens: "prompt")
    monkeypatch.setattr(
        bench_runner,
        "prefill_decode",
        lambda model, tokenizer, prompt, device, max_new_tokens: {
            "prompt_tokens": 4,
            "decode_tokens": int(max_new_tokens),
            "decode_toks_per_s": 10.0,
            "gen_token_ids": [7],
            "_final_past_key_values": None,
        },
    )
    monkeypatch.setattr(bench_runner, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench_runner, "sync", lambda device: None)

    result = bench_runner.run_smoke(
        model_id="demo/model",
        preset="balanced",
        prompt_tokens=8,
        max_new_tokens=1,
        calib_new_tokens=1,
        search_max_n=1,
        trust_remote_code=False,
    )

    assert result["ok"] is True
    assert captured["tokenizer"]["trust_remote_code"] is False
    assert captured["model"]["trust_remote_code"] is False
