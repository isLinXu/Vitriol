from __future__ import annotations


def _install_transformers_stub(monkeypatch, captured: dict) -> None:
    import sys
    import types

    transformers_stub = types.ModuleType("transformers")

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["config"] = {"model_id": model_id, "kwargs": dict(kwargs)}
            return object()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["tokenizer"] = {"model_id": model_id, "kwargs": dict(kwargs)}
            return object()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["model"] = {"model_id": model_id, "kwargs": dict(kwargs)}

            class _M:
                def to(self, _device):
                    return self

                def eval(self):
                    return self

            return _M()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured["model_generic"] = {"model_id": model_id, "kwargs": dict(kwargs)}
            return object()

        @staticmethod
        def from_config(_config, **kwargs):
            captured["model_from_config"] = dict(kwargs)
            return object()

    class _PretrainedConfig:
        @staticmethod
        def from_dict(data):
            captured["pretrained_config_from_dict"] = dict(data)
            return {"raw_config": dict(data)}

    transformers_stub.AutoConfig = _AutoConfig
    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM
    transformers_stub.AutoModel = _AutoModel
    transformers_stub.PretrainedConfig = _PretrainedConfig

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_hf_loading_respects_security_flags(monkeypatch) -> None:
    captured: dict = {}
    _install_transformers_stub(monkeypatch, captured)

    from vitriol.config.manager import SecurityOptions

    # New module: hf_loading (test-first; should fail if not implemented).
    from vitriol.utils import hf_loading

    sec = SecurityOptions(
        trust_remote_code=False,
        allow_network=False,  # should force local_files_only=True
        local_files_only=False,
    )

    hf_loading.load_config("demo/model", security=sec)
    hf_loading.load_tokenizer("demo/model", security=sec)
    hf_loading.load_model("demo/model", security=sec)
    hf_loading.load_causallm("demo/model", security=sec, torch_dtype="float32", device="cpu")

    assert captured["config"]["kwargs"]["trust_remote_code"] is False
    assert captured["tokenizer"]["kwargs"]["trust_remote_code"] is False
    assert captured["model_generic"]["kwargs"]["trust_remote_code"] is False
    assert captured["model"]["kwargs"]["trust_remote_code"] is False

    # allow_network=False → local_files_only must be True
    assert captured["config"]["kwargs"]["local_files_only"] is True
    assert captured["tokenizer"]["kwargs"]["local_files_only"] is True
    assert captured["model_generic"]["kwargs"]["local_files_only"] is True
    assert captured["model"]["kwargs"]["local_files_only"] is True


def test_hf_loading_model_from_config_respects_trust_remote_code(monkeypatch) -> None:
    captured: dict = {}
    _install_transformers_stub(monkeypatch, captured)

    from vitriol.config.manager import SecurityOptions
    from vitriol.utils import hf_loading

    sec = SecurityOptions(trust_remote_code=False, allow_network=True, local_files_only=False)
    hf_loading.load_model_from_config(config=object(), security=sec)

    assert captured["model_from_config"]["trust_remote_code"] is False


def test_hf_loading_accepts_security_context(monkeypatch) -> None:
    captured: dict = {}
    _install_transformers_stub(monkeypatch, captured)

    from vitriol.security.context import resolve_security_context
    from vitriol.utils import hf_loading

    ctx = resolve_security_context(base={"trust_remote_code": False, "allow_network": False, "local_files_only": False})
    hf_loading.load_config("demo/model", security=ctx)
    assert captured["config"]["kwargs"]["trust_remote_code"] is False
    assert captured["config"]["kwargs"]["local_files_only"] is True


def test_hf_loading_load_config_or_raw_falls_back_for_unknown_model_type(monkeypatch, tmp_path) -> None:
    import importlib
    import sys
    import types

    config_dir = tmp_path / "hy3"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        '{"model_type":"hy_v3","hidden_size":4096,"num_hidden_layers":80,"vocab_size":120832}',
        encoding="utf-8",
    )

    transformers_stub = types.ModuleType("transformers")

    class _AutoConfig:
        @staticmethod
        def from_pretrained(_model_id: str, **_kwargs):
            raise ValueError(
                "The checkpoint you are trying to load has model type `hy_v3` "
                "but Transformers does not recognize this architecture."
            )

    class _PretrainedConfig:
        @staticmethod
        def from_dict(data):
            return {"raw_config": dict(data)}

    transformers_stub.AutoConfig = _AutoConfig
    transformers_stub.PretrainedConfig = _PretrainedConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    from vitriol.utils import hf_loading

    importlib.reload(hf_loading)

    loaded = hf_loading.load_config_or_raw(str(config_dir))

    assert loaded["raw_config"]["model_type"] == "hy_v3"
    assert loaded["raw_config"]["hidden_size"] == 4096


def test_hf_loading_load_config_or_raw_returns_raw_config_without_transformers(monkeypatch, tmp_path) -> None:
    import builtins
    import importlib
    import sys

    config_dir = tmp_path / "local-model"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        (
            '{"model_type":"hy_v3","hidden_size":4096,"num_hidden_layers":80,'
            '"text_config":{"vocab_size":120832,"hidden_size":4096}}'
        ),
        encoding="utf-8",
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    from vitriol.utils import hf_loading

    importlib.reload(hf_loading)

    loaded = hf_loading.load_config_or_raw(
        str(config_dir),
        security={"trust_remote_code": False, "allow_network": False, "local_files_only": True},
    )

    assert loaded.model_type == "hy_v3"
    assert loaded.text_config.hidden_size == 4096
    assert loaded.to_dict()["text_config"]["vocab_size"] == 120832


def test_build_config_object_falls_back_to_raw_config_when_pretrained_config_breaks(monkeypatch) -> None:
    import importlib
    import sys
    import types

    transformers_stub = types.ModuleType("transformers")

    class _PretrainedConfig:
        @staticmethod
        def from_dict(_data):
            raise AttributeError("'PreTrainedConfig' object has no attribute 'max_position_embeddings'")

    transformers_stub.PretrainedConfig = _PretrainedConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    from vitriol.utils import hf_loading

    importlib.reload(hf_loading)

    loaded = hf_loading.build_config_object(
        {
            "model_type": "deepseek_v4",
            "text_config": {
                "hidden_size": 4096,
                "max_position_embeddings": 1048576,
            },
        }
    )

    assert loaded.model_type == "deepseek_v4"
    assert loaded.text_config.max_position_embeddings == 1048576


def test_offline_custom_from_config_passes_policy_only_to_code_resolver(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import transformers.dynamic_module_utils as dynamic
    import transformers.models.auto.auto_factory as factory

    from vitriol.utils import hf_loading

    model = object()
    remote_class = MagicMock()
    remote_class._from_config.return_value = model
    resolver = MagicMock(return_value=remote_class)
    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", resolver)
    if hasattr(factory, "add_generation_mixin_to_remote_model"):
        monkeypatch.setattr(factory, "add_generation_mixin_to_remote_model", lambda cls: cls)
    config = SimpleNamespace(auto_map={"AutoModel": "other/repo--modeling.CustomModel"}, name_or_path="local/model")
    result = hf_loading.load_model_from_config(
        config, security={"trust_remote_code": True, "allow_network": False}, torch_dtype="float32")
    assert result is model
    resolver.assert_called_once_with("modeling.CustomModel", "other/repo", local_files_only=True)
    remote_class._from_config.assert_called_once_with(config, torch_dtype="float32")


def test_dynamic_module_patch_respects_offline_policy(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock

    import transformers.dynamic_module_utils as dynamic

    from vitriol.utils import hf_loading

    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "other/repo--modeling.CustomModel"}}))
    resolver = MagicMock(side_effect=OSError("not cached"))
    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", resolver)
    hf_loading._maybe_patch_dynamic_modules(str(tmp_path), security={"trust_remote_code": True, "allow_network": False})
    assert resolver.call_count == 1
    assert resolver.call_args.kwargs["local_files_only"] is True


def test_real_offline_custom_model_constructor_receives_no_hub_kwargs(tmp_path):
    import torch
    from transformers import PretrainedConfig

    from vitriol.utils import hf_loading

    (tmp_path / "modeling_custom.py").write_text(
        "import torch\n"
        "from transformers import PreTrainedModel, PretrainedConfig\n"
        "class LocalCustomModel(PreTrainedModel):\n"
        "    config_class = PretrainedConfig\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        self.weight = torch.nn.Parameter(torch.ones(2))\n"
    )
    config = PretrainedConfig()
    config.name_or_path = str(tmp_path)
    config.auto_map = {"AutoModel": "modeling_custom.LocalCustomModel"}
    model = hf_loading.load_model_from_config(
        config, security={"trust_remote_code": True, "allow_network": False})
    assert torch.equal(model.weight, torch.ones(2))
