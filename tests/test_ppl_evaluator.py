from __future__ import annotations

from dataclasses import replace

import pytest


def _install_torch_stub(monkeypatch) -> None:
    """
    The unit test environment does not install torch (too large), so we provide a minimal stub
    for tests that only need imports to succeed.
    Note: these tests do not cover real numeric computation; they only cover pure-Python logic
    such as parameter propagation and report rendering.
    """
    import sys
    import types

    if "torch" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")

    class _DummyDevice:
        def __init__(self, type_: str):
            self.type = type_

        def __repr__(self) -> str:
            return f"device(type={self.type})"

    # dtype sentinels
    torch_stub.float16 = object()
    torch_stub.float32 = object()
    torch_stub.bfloat16 = object()

    def _device(type_: str) -> _DummyDevice:
        return _DummyDevice(type_)

    torch_stub.device = _device

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = _Cuda()

    def _no_grad():
        def _decorator(fn):
            return fn

        return _decorator

    torch_stub.no_grad = _no_grad
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_stub.functional = torch_nn_functional_stub
    torch_stub.nn = torch_nn_stub

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_stub)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch_nn_functional_stub)


def _install_transformers_stub(monkeypatch) -> None:
    """
    Similarly, to avoid installing transformers (large/complex dependencies), we provide a minimal stub
    so imports inside ppl_evaluator._load_model succeed and we can monkeypatch from_pretrained().
    """
    import sys
    import types

    if "transformers" in sys.modules:
        return

    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("should be monkeypatched in test")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("should be monkeypatched in test")

    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_ppl_result_report_renders_without_attribute_errors(monkeypatch) -> None:
    """Report rendering should be stable and must not raise AttributeError due to field name typos."""
    _install_torch_stub(monkeypatch)
    from vitriol.bench.ppl_evaluator import LayerPPLResult, PPLResult

    result = PPLResult(
        model_id="demo/model",
        device="cpu",
        kv_preset="balanced",
        ppl_baseline=10.0,
        ppl_tuned=11.0,
        ppl_ratio=1.1,
        # Set to 30% to cover the high-degradation branch in report(), preventing missing-field typos from slipping.
        ppl_degradation=30.0,
        token_exact_match_rate=0.5,
        token_prefix_match_avg=0.5,
        generated_text_baseline="hello",
        generated_text_tuned="hello",
        memory_kv_bytes_baseline=100,
        memory_kv_bytes_tuned=50,
        memory_savings_pct=50.0,
        decode_speed_toks_per_sec_base=10.0,
        decode_speed_toks_per_sec_tuned=12.0,
        speedup_ratio=1.2,
        layers=[
            LayerPPLResult(
                layer_idx=0,
                layer_type="full_attention",
                logit_kl_divergence=0.01,
                logit_cosine_similarity=0.99,
                attention_mse=0.001,
                kv_compression_ratio=0.5,
            )
        ],
        avg_logit_kl=0.01,
        worst_layer_kl=(0, 0.01),
        eval_time_seconds=0.1,
    )

    text = result.report()
    assert "PPL Evaluation Report" in text
    assert "PPL Degradation" in text


def test_ppl_evaluator_respects_trust_remote_code_flag(monkeypatch) -> None:
    """
    The PPL evaluator is a paper-grade evaluation component and must respect the trust_remote_code
    security flag. We monkeypatch transformers loading calls to verify parameter propagation.
    """
    _install_torch_stub(monkeypatch)
    _install_transformers_stub(monkeypatch)
    from vitriol.bench.ppl_evaluator import PPLConfig, PPLEvaluator

    captured = {"tokenizer": None, "model": None}

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, *args, **kwargs):
            raise RuntimeError("not used in this unit test")

    def fake_tokenizer_from_pretrained(model_id: str, **kwargs):
        captured["tokenizer"] = {"model_id": model_id, "kwargs": dict(kwargs)}
        return DummyTokenizer()

    def fake_model_from_pretrained(model_id: str, **kwargs):
        captured["model"] = {"model_id": model_id, "kwargs": dict(kwargs)}
        raise StopIteration("stop after capturing kwargs")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_model_from_pretrained)

    cfg = PPLConfig(model_id="demo/model", device="cpu", dtype="float32")
    # New field: trust_remote_code (test-first; should fail if not implemented).
    cfg = replace(cfg, trust_remote_code=False)

    evaluator = PPLEvaluator(cfg)
    with pytest.raises(StopIteration):
        evaluator._load_model()

    assert captured["tokenizer"]["kwargs"].get("trust_remote_code") is False
    assert captured["model"]["kwargs"].get("trust_remote_code") is False


def test_ppl_evaluator_uses_current_kv_hook_parameter_name() -> None:
    """The PPL path must call the KV hook helper with its current signature."""
    import inspect

    from vitriol.bench.ppl_evaluator import PPLEvaluator
    from vitriol.bench.runner import _apply_vitriol_universal

    hook_params = inspect.signature(_apply_vitriol_universal).parameters
    evaluate_source = inspect.getsource(PPLEvaluator.evaluate)

    assert "v_quantize_only_first_n_layers" in hook_params
    assert "v_quantize_only_first_n_layers=int(first_n)" in evaluate_source
    assert "v_quantize_only_first_n=int(first_n)" not in evaluate_source
