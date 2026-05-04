"""Tests for vitriol.core.pipeline.steps.shard_map module."""
from vitriol.config.manager import GenerationConfig
from vitriol.core.pipeline.context import GenerationContext
from vitriol.core.pipeline.steps.shard_map import ResolveShardMapStep


class TestResolveShardMapStep:
    def test_step_name(self):
        step = ResolveShardMapStep()
        assert step.name == "resolve_shard_map"

    def test_empty_original_map(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        step = ResolveShardMapStep()
        step.run(ctx)
        assert ctx.original_shard_map == {}
        assert ctx.expected_shards == []

    def test_single_shard_normalization(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "pytorch_model.bin",
        }
        step = ResolveShardMapStep()
        step.run(ctx)
        assert ctx.original_shard_map == {"layer1.weight": "pytorch_model-00001-of-00001.bin"}
        assert ctx.expected_shards == ["pytorch_model-00001-of-00001.bin"]

    def test_multiple_shards_normalization(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "pytorch_model.bin",
            "layer2.weight": "model.safetensors",
            "layer3.weight": "pytorch_model.bin",
        }
        step = ResolveShardMapStep()
        step.run(ctx)

        # Two unique shards sorted alphabetically
        assert len(ctx.expected_shards) == 2
        assert ctx.expected_shards == sorted(ctx.expected_shards)

        # All values should be normalized
        for v in ctx.original_shard_map.values():
            assert v.startswith("pytorch_model-") or v.startswith("model-")
            assert "-of-" in v

    def test_safetensors_extension_preserved(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "model.safetensors",
        }
        step = ResolveShardMapStep()
        step.run(ctx)
        assert ctx.original_shard_map == {"layer1.weight": "model-00001-of-00001.safetensors"}

    def test_mixed_extensions(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "pytorch_model.bin",
            "layer2.weight": "model.safetensors",
        }
        step = ResolveShardMapStep()
        step.run(ctx)

        assert ctx.original_shard_map["layer1.weight"].endswith(".bin")
        assert ctx.original_shard_map["layer2.weight"].endswith(".safetensors")

    def test_multiple_unique_shards_count(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "a.weight": "shard1.bin",
            "b.weight": "shard2.bin",
            "c.weight": "shard3.bin",
        }
        step = ResolveShardMapStep()
        step.run(ctx)

        assert len(ctx.expected_shards) == 3
        for i, shard in enumerate(ctx.expected_shards, 1):
            assert f"-of-00003" in shard

    def test_use_ctx_original_map_only_flag(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "model.bin",
        }
        step = ResolveShardMapStep(_use_ctx_original_map_only=True)
        step.run(ctx)
        assert ctx.original_shard_map == {"layer1.weight": "model-00001-of-00001.bin"}

    def test_expected_shards_sorted(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "z.weight": "shard_b.bin",
            "a.weight": "shard_a.bin",
        }
        step = ResolveShardMapStep()
        step.run(ctx)

        assert ctx.expected_shards == sorted(ctx.expected_shards)

    def test_pytorch_model_prefix(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "pytorch_model.bin",
        }
        step = ResolveShardMapStep()
        step.run(ctx)
        assert "pytorch_model" in ctx.original_shard_map["layer1.weight"]

    def test_model_prefix(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = {
            "layer1.weight": "model.safetensors",
        }
        step = ResolveShardMapStep()
        step.run(ctx)
        assert "model" in ctx.original_shard_map["layer1.weight"]

    def test_original_map_none(self):
        config = GenerationConfig()
        ctx = GenerationContext(
            model_id="test/model",
            output_dir="/tmp/output",
            config=config,
        )
        ctx.original_shard_map = None
        step = ResolveShardMapStep()
        step.run(ctx)
        assert ctx.original_shard_map == {}
        assert ctx.expected_shards == []
