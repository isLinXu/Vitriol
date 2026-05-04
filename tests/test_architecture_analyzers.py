from __future__ import annotations

from vitriol.arch_viz.analyzer import ArchitectureAnalyzer
from vitriol.arch_viz.analyzers import (
    AnalyzerRegistry,
    BaichuanAnalyzer,
    BertAnalyzer,
    BloomAnalyzer,
    CohereAnalyzer,
    DeepSeekAnalyzer,
    FalconAnalyzer,
    GemmaAnalyzer,
    GPTNeoXAnalyzer,
    Hy3Analyzer,
    InternLMAnalyzer,
    LlamaAnalyzer,
    MistralAnalyzer,
    OPTAnalyzer,
    PhiAnalyzer,
    Qwen2MoeAnalyzer,
    SequenceMixerAnalyzer,
    StableLMAnalyzer,
    StarCoder2Analyzer,
    StarCoderAnalyzer,
    T5Analyzer,
    YiAnalyzer,
)
from vitriol.utils.hf_loading import RawConfig


def _cfg(data: dict) -> RawConfig:
    return RawConfig.from_dict(data)


def test_analyzer_registry_resolves_new_family_aliases() -> None:
    assert isinstance(AnalyzerRegistry.get("gemma4_text"), GemmaAnalyzer)
    assert isinstance(AnalyzerRegistry.get("phi4"), PhiAnalyzer)
    assert isinstance(AnalyzerRegistry.get("cohere2"), CohereAnalyzer)
    assert isinstance(AnalyzerRegistry.get("stablelm_epoch"), StableLMAnalyzer)
    assert isinstance(AnalyzerRegistry.get("mixtral"), MistralAnalyzer)
    assert isinstance(AnalyzerRegistry.get("qwen2_moe"), Qwen2MoeAnalyzer)
    assert isinstance(AnalyzerRegistry.get("llama"), LlamaAnalyzer)
    assert isinstance(AnalyzerRegistry.get("bert"), BertAnalyzer)
    assert isinstance(AnalyzerRegistry.get("t5"), T5Analyzer)
    assert isinstance(AnalyzerRegistry.get("bloom"), BloomAnalyzer)
    assert isinstance(AnalyzerRegistry.get("gpt_neox"), GPTNeoXAnalyzer)
    assert isinstance(AnalyzerRegistry.get("gpt_bigcode"), StarCoderAnalyzer)
    assert isinstance(AnalyzerRegistry.get("starcoder2"), StarCoder2Analyzer)
    assert isinstance(AnalyzerRegistry.get("falcon"), FalconAnalyzer)
    assert isinstance(AnalyzerRegistry.get("opt"), OPTAnalyzer)
    assert isinstance(AnalyzerRegistry.get("yi"), YiAnalyzer)
    assert isinstance(AnalyzerRegistry.get("internlm2"), InternLMAnalyzer)
    assert isinstance(AnalyzerRegistry.get("baichuan"), BaichuanAnalyzer)
    assert isinstance(AnalyzerRegistry.get("deepseek_v4"), DeepSeekAnalyzer)
    assert isinstance(AnalyzerRegistry.get("hy_v3"), Hy3Analyzer)
    assert isinstance(AnalyzerRegistry.get("mamba"), SequenceMixerAnalyzer)


def test_analyzer_registry_resolves_from_architectures_when_model_type_is_unknown() -> None:
    bert_cfg = _cfg(
        {
            "model_type": "unknown_encoder",
            "architectures": ["BertForMaskedLM"],
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "vocab_size": 30522,
        }
    )
    assert isinstance(AnalyzerRegistry.resolve(bert_cfg), BertAnalyzer)

    t5_cfg = _cfg(
        {
            "model_type": "unknown_seq2seq",
            "architectures": ["T5ForConditionalGeneration"],
            "d_model": 1024,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "d_ff": 2816,
            "d_kv": 64,
            "vocab_size": 32128,
            "is_encoder_decoder": True,
        }
    )
    assert isinstance(AnalyzerRegistry.resolve(t5_cfg), T5Analyzer)

    baichuan_cfg = _cfg(
        {
            "model_type": "unknown_decoder",
            "architectures": ["BaichuanForCausalLM"],
            "vocab_size": 125696,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
        }
    )
    assert isinstance(AnalyzerRegistry.resolve(baichuan_cfg), BaichuanAnalyzer)

    neox_cfg = _cfg(
        {
            "model_type": "unknown_decoder",
            "architectures": ["GPTNeoXForCausalLM"],
            "vocab_size": 50432,
            "hidden_size": 6144,
            "num_hidden_layers": 44,
            "num_attention_heads": 64,
            "intermediate_size": 24576,
        }
    )
    assert isinstance(AnalyzerRegistry.resolve(neox_cfg), GPTNeoXAnalyzer)


def test_llama_analyzer_reports_gqa_and_long_context() -> None:
    cfg = _cfg(
        {
            "model_type": "llama",
            "vocab_size": 128256,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
        }
    )

    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "llama"
    assert "LLaMA" in arch.features
    assert "GQA" in arch.features
    assert "Long Context" in arch.features
    assert "RoPE" in arch.features
    assert "RMSNorm" in arch.features
    assert "SwiGLU" in arch.features
    assert arch.parameters["num_kv_heads"] == 8
    assert arch.parameters["max_position"] == 131072
    assert arch.total_layers == 32


def test_mixtral_analyzer_reports_moe_and_sliding_window() -> None:
    cfg = _cfg(
        {
            "model_type": "mixtral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 4096,
            "sliding_window": 4096,
            "tie_word_embeddings": False,
        }
    )

    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "mixtral"
    assert "Mixtral" in arch.features
    assert "MoE" in arch.features
    assert "Sliding Window" in arch.features
    assert arch.parameters["num_experts"] == 8
    moe_layer = next(layer for layer in arch.layers if layer.name == "Block 0 - FFN")
    assert "Experts: 8" in moe_layer.description
    assert "Shared Inter: 4096" in moe_layer.description


def test_deepseek_v4_analyzer_reports_csa_hca_hash_and_fp8_metadata() -> None:
    cfg = _cfg(
        {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 6,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "max_position_embeddings": 1048576,
            "sliding_window": 128,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 2048,
            "num_hash_layers": 2,
            "compress_ratios": [0, 0, 4, 128, 4, 0],
            "compress_rope_theta": 160000,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "qk_rope_head_dim": 64,
            "rope_scaling": {"type": "yarn", "factor": 16},
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            },
            "topk_method": "noaux_tc",
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": False,
        }
    )

    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "deepseek_v4"
    assert "DeepSeek-V4" in arch.features
    assert "CSA/HCA" in arch.features
    assert "Hash Attention" in arch.features
    assert "Compressed Attention" in arch.features
    assert "Compressed RoPE" in arch.features
    assert "YARN RoPE" in arch.features
    assert "FP8" in arch.features
    assert "MQA" in arch.features
    assert arch.parameters["num_hash_layers"] == 2
    assert arch.parameters["compressed_attention_layers"] == 3
    assert arch.parameters["quant_method"] == "fp8"
    assert arch.parameters["index_topk"] == 512
    assert arch.parameters["num_kv_heads"] == 1
    assert arch.parameters["layer_types"] == [
        "hash_attention",
        "hash_attention",
        "compressed_attention",
        "compressed_attention",
        "compressed_attention",
        "sliding_window",
    ]
    assert "Hash Attention" in next(layer.description for layer in arch.layers if layer.name == "Block 0 - Attention")
    assert "CSA/HCA compressed x128" in next(layer.description for layer in arch.layers if layer.name == "Block 3 - Attention")


def test_sequence_mixer_analyzer_marks_no_kv_cache_for_mamba_like_models() -> None:
    cfg = _cfg(
        {
            "model_type": "mamba",
            "architectures": ["MambaForCausalLM"],
            "vocab_size": 50280,
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "state_size": 16,
            "conv_kernel": 4,
            "intermediate_size": 1536,
        }
    )

    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "mamba"
    assert "Mamba" in arch.features
    assert "Sequence Mixer" in arch.features
    assert "No KV Cache" in arch.features
    assert arch.parameters["supports_kv_cache"] is False
    assert arch.parameters["layer_types"] == ["other"] * 24
    assert any(layer.type == "sequence_mixer" for layer in arch.layers)


def test_gemma_multimodal_analyzer_adds_vision_stack() -> None:
    cfg = _cfg(
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 262144,
                "hidden_size": 2560,
                "num_hidden_layers": 28,
                "num_attention_heads": 20,
                "num_key_value_heads": 10,
                "intermediate_size": 10240,
                "max_position_embeddings": 32768,
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "patch_size": 14,
            },
        }
    )

    arch = ArchitectureAnalyzer().analyze(cfg)

    assert arch.model_type == "gemma4"
    assert "Gemma" in arch.features
    assert "Vision" in arch.features
    assert arch.parameters["vision_hidden_size"] == 1152
    assert arch.parameters["vision_layers"] == 27
    assert arch.layers[0].name == "Vision Encoder"
    assert arch.layers[1].name == "Vision Projector"
    assert arch.total_layers == 28
    assert len(arch.layers) > arch.total_layers


def test_phi_and_cohere_analyzers_surface_family_specific_features() -> None:
    phi_cfg = _cfg(
        {
            "model_type": "phi4",
            "vocab_size": 100352,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 10,
            "intermediate_size": 17920,
            "max_position_embeddings": 131072,
            "partial_rotary_factor": 0.5,
        }
    )
    phi_arch = ArchitectureAnalyzer().analyze(phi_cfg)
    assert "Phi" in phi_arch.features
    assert "Partial RoPE" in phi_arch.features
    assert "Long Context" in phi_arch.features

    cohere_cfg = _cfg(
        {
            "model_type": "cohere2",
            "vocab_size": 256000,
            "hidden_size": 4096,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "sliding_window": 4096,
            "rope_theta": 8000000.0,
            "use_qk_norm": True,
        }
    )
    cohere_arch = ArchitectureAnalyzer().analyze(cohere_cfg)
    assert "Cohere" in cohere_arch.features
    assert "GQA" in cohere_arch.features
    assert "Sliding Window" in cohere_arch.features
    assert "QK-Norm" in cohere_arch.features


def test_bert_and_t5_analyzers_report_encoder_layouts() -> None:
    bert_cfg = _cfg(
        {
            "model_type": "roberta",
            "vocab_size": 50265,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
        }
    )
    bert_arch = ArchitectureAnalyzer().analyze(bert_cfg)
    assert bert_arch.arch_type == "encoder-only"
    assert "Bidirectional" in bert_arch.features
    assert "AbsPos" in bert_arch.features
    assert getattr(bert_arch, "encoder_layers", 0) == 24
    assert bert_arch.layers[0].name == "Token Embedding"

    t5_cfg = _cfg(
        {
            "model_type": "t5",
            "architectures": ["T5ForConditionalGeneration"],
            "vocab_size": 32128,
            "d_model": 1024,
            "d_ff": 2816,
            "d_kv": 64,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "feed_forward_proj": "gated-gelu",
            "is_encoder_decoder": True,
        }
    )
    t5_arch = ArchitectureAnalyzer().analyze(t5_cfg)
    assert t5_arch.arch_type == "encoder-decoder"
    assert "CrossAttn" in t5_arch.features
    assert "RelPos" in t5_arch.features
    assert "GeGLU" in t5_arch.features
    assert getattr(t5_arch, "encoder_layers", 0) == 24
    assert getattr(t5_arch, "decoder_layers", 0) == 24
    assert t5_arch.to_dict()["encoder_layers"] == 24
    assert t5_arch.to_dict()["decoder_layers"] == 24
    assert t5_arch.total_layers == 48


def test_falcon_and_opt_analyzers_surface_decoder_specific_features() -> None:
    falcon_cfg = _cfg(
        {
            "model_type": "falcon",
            "vocab_size": 65024,
            "hidden_size": 4544,
            "num_hidden_layers": 32,
            "num_attention_heads": 71,
            "intermediate_size": 18176,
            "max_position_embeddings": 2048,
            "alibi": True,
            "multi_query": True,
        }
    )
    falcon_arch = ArchitectureAnalyzer().analyze(falcon_cfg)
    assert falcon_arch.arch_type == "decoder-only"
    assert "Falcon" in falcon_arch.features
    assert "ALiBi" in falcon_arch.features
    assert "MQA" in falcon_arch.features
    assert falcon_arch.parameters["num_kv_heads"] == 1

    opt_cfg = _cfg(
        {
            "model_type": "opt",
            "vocab_size": 50272,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "ffn_dim": 10240,
            "max_position_embeddings": 2048,
        }
    )
    opt_arch = ArchitectureAnalyzer().analyze(opt_cfg)
    assert "OPT" in opt_arch.features
    assert "LearnedPos" in opt_arch.features
    assert "ReLU" in opt_arch.features
    assert opt_arch.parameters["intermediate_size"] == 10240


def test_baichuan_yi_and_internlm_analyzers_surface_family_features() -> None:
    baichuan_cfg = _cfg(
        {
            "model_type": "baichuan",
            "vocab_size": 125696,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
    )
    baichuan_arch = ArchitectureAnalyzer().analyze(baichuan_cfg)
    assert "Baichuan" in baichuan_arch.features
    assert "RMSNorm" in baichuan_arch.features
    assert "SwiGLU" in baichuan_arch.features
    assert baichuan_arch.total_layers == 32

    yi_cfg = _cfg(
        {
            "model_type": "yi",
            "vocab_size": 64000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "rope_scaling": {"type": "linear", "factor": 4.0},
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
    )
    yi_arch = ArchitectureAnalyzer().analyze(yi_cfg)
    assert "Yi" in yi_arch.features
    assert "Dynamic NTK" in yi_arch.features
    assert "GQA" in yi_arch.features

    internlm_cfg = _cfg(
        {
            "model_type": "internlm2",
            "vocab_size": 92544,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
    )
    internlm_arch = ArchitectureAnalyzer().analyze(internlm_cfg)
    assert "InternLM" in internlm_arch.features
    assert "RMSNorm" in internlm_arch.features
    assert "SwiGLU" in internlm_arch.features
    assert "GQA" in internlm_arch.features


def test_bloom_neox_and_starcoder_families_surface_decoder_features() -> None:
    bloom_cfg = _cfg(
        {
            "model_type": "bloom",
            "vocab_size": 250880,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "seq_length": 2048,
        }
    )
    bloom_arch = ArchitectureAnalyzer().analyze(bloom_cfg)
    assert "Bloom" in bloom_arch.features
    assert "ALiBi" in bloom_arch.features
    assert "MHA" in bloom_arch.features
    assert bloom_arch.total_layers == 24

    neox_cfg = _cfg(
        {
            "model_type": "gpt_neox",
            "vocab_size": 50432,
            "hidden_size": 6144,
            "num_hidden_layers": 44,
            "num_attention_heads": 64,
            "intermediate_size": 24576,
            "max_position_embeddings": 32768,
            "rotary_pct": 0.25,
            "use_parallel_residual": True,
        }
    )
    neox_arch = ArchitectureAnalyzer().analyze(neox_cfg)
    assert "GPT-NeoX" in neox_arch.features
    assert "RoPE" in neox_arch.features
    assert "Parallel Residual" in neox_arch.features
    assert "Long Context" in neox_arch.features

    starcoder_cfg = _cfg(
        {
            "model_type": "gpt_bigcode",
            "vocab_size": 49152,
            "hidden_size": 6144,
            "num_hidden_layers": 40,
            "num_attention_heads": 48,
            "n_inner": 24576,
            "max_position_embeddings": 8192,
            "multi_query": True,
            "tie_word_embeddings": True,
        }
    )
    starcoder_arch = ArchitectureAnalyzer().analyze(starcoder_cfg)
    assert "StarCoder" in starcoder_arch.features
    assert "LearnedPos" in starcoder_arch.features
    assert "MQA" in starcoder_arch.features
    assert starcoder_arch.parameters["num_kv_heads"] == 1

    starcoder2_cfg = _cfg(
        {
            "model_type": "starcoder2",
            "vocab_size": 49152,
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "intermediate_size": 12288,
            "max_position_embeddings": 65536,
            "rope_theta": 1000000.0,
            "sliding_window": 4096,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
        }
    )
    starcoder2_arch = ArchitectureAnalyzer().analyze(starcoder2_cfg)
    assert "StarCoder2" in starcoder2_arch.features
    assert "GQA" in starcoder2_arch.features
    assert "RoPE" in starcoder2_arch.features
    assert "Sliding Window" in starcoder2_arch.features
    assert starcoder2_arch.total_layers == 30
