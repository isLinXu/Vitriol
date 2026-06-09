"""Analyzer registry: maps model types / architectures to analyzers."""
from typing import Any

from ._helpers import (
    _architectures,
    _cfg_get,
)
from .base import ModelAnalyzer, TransformerAnalyzer
from .deepseek import DeepSeekAnalyzer, KimiAnalyzer
from .dense import (
    BaichuanAnalyzer,
    CohereAnalyzer,
    GemmaAnalyzer,
    InternLMAnalyzer,
    LlamaAnalyzer,
    MistralAnalyzer,
    PhiAnalyzer,
    Qwen2MoeAnalyzer,
    QwenAnalyzer,
    StableLMAnalyzer,
    YiAnalyzer,
)
from .encoder import BertAnalyzer, T5Analyzer
from .ernie import ErnieAnalyzer
from .glm import GLMAnalyzer
from .gpt import (
    BloomAnalyzer,
    FalconAnalyzer,
    GPT2Analyzer,
    GPTNeoXAnalyzer,
    OPTAnalyzer,
    StarCoder2Analyzer,
    StarCoderAnalyzer,
)
from .hy3 import Hy3Analyzer
from .intern_s1 import InternS1Analyzer
from .minimax import MiniMaxAnalyzer
from .qwen35 import Qwen35Analyzer
from .sequence_mixer import SequenceMixerAnalyzer


class AnalyzerRegistry:
    """Registry for architecture analyzers with key resolution."""
    _analyzers = {
        "default": TransformerAnalyzer(),
        "bert": BertAnalyzer(),
        "roberta": BertAnalyzer(),
        "t5": T5Analyzer(),
        "bart": T5Analyzer(),
        "bloom": BloomAnalyzer(),
        "gptneox": GPTNeoXAnalyzer(),
        "gpt_neox": GPTNeoXAnalyzer(),
        "gpt_bigcode": StarCoderAnalyzer(),
        "starcoder2": StarCoder2Analyzer(),
        "falcon": FalconAnalyzer(),
        "opt": OPTAnalyzer(),
        "qwen": QwenAnalyzer(),
        "qwen2": QwenAnalyzer(),
        "qwen2_moe": Qwen2MoeAnalyzer(),
        "deepseek": DeepSeekAnalyzer(),
        "deepseek_v3": DeepSeekAnalyzer(),
        "deepseek_v4": DeepSeekAnalyzer(),
        "llama": LlamaAnalyzer(),
        "mistral": MistralAnalyzer(),
        "mixtral": MistralAnalyzer(),
        "gemma": GemmaAnalyzer(),
        "gemma2": GemmaAnalyzer(),
        "gemma3": GemmaAnalyzer(),
        "gemma3_text": GemmaAnalyzer(),
        "gemma4": GemmaAnalyzer(),
        "gemma4_text": GemmaAnalyzer(),
        "phi": PhiAnalyzer(),
        "phi1": PhiAnalyzer(),
        "phi2": PhiAnalyzer(),
        "phi3": PhiAnalyzer(),
        "phi4": PhiAnalyzer(),
        "cohere": CohereAnalyzer(),
        "cohere2": CohereAnalyzer(),
        "stablelm": StableLMAnalyzer(),
        "stablelm_epoch": StableLMAnalyzer(),
        "stableplankton": StableLMAnalyzer(),
        "yi": YiAnalyzer(),
        "internlm": InternLMAnalyzer(),
        "internlm2": InternLMAnalyzer(),
        "internlm3": InternLMAnalyzer(),
        "baichuan": BaichuanAnalyzer(),
        "glm": GLMAnalyzer(),
        "glm4": GLMAnalyzer(),
        "glm5": GLMAnalyzer(),
        "glm_moe_dsa": GLMAnalyzer(),
        "chatglm": GLMAnalyzer(),
        "kimi_k25": KimiAnalyzer(),
        "ernie4_5_moe_vl": ErnieAnalyzer(),
        "gpt2": GPT2Analyzer(),
        "minimax_m2": MiniMaxAnalyzer(),
        "interns1_pro": InternS1Analyzer(),
        "qwen3_5_moe": Qwen35Analyzer(), # Add Qwen3.5 support
        "qwen3_5_moe_text": Qwen35Analyzer(),
        "hy_v3": Hy3Analyzer(),
        "mamba": SequenceMixerAnalyzer(),
        "mamba2": SequenceMixerAnalyzer(),
        "rwkv": SequenceMixerAnalyzer(),
        "retnet": SequenceMixerAnalyzer(),
        "hyena": SequenceMixerAnalyzer(),
    }

    _architecture_aliases = [
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("mixtral", "mixtral"),
        ("gemma", "gemma"),
        ("phi", "phi"),
        ("cohere", "cohere"),
        ("bloom", "bloom"),
        ("gptneox", "gpt_neox"),
        ("gpt_neox", "gpt_neox"),
        ("starcoder2", "starcoder2"),
        ("starcoder", "gpt_bigcode"),
        ("bigcode", "gpt_bigcode"),
        ("falcon", "falcon"),
        ("opt", "opt"),
        ("yi", "yi"),
        ("internlm", "internlm2"),
        ("baichuan", "baichuan"),
        ("bert", "bert"),
        ("roberta", "roberta"),
        ("t5", "t5"),
        ("bart", "bart"),
        ("glm", "glm"),
        ("chatglm", "chatglm"),
        ("qwen2moe", "qwen2_moe"),
        ("qwen3_5moe", "qwen3_5_moe"),
        ("qwen3_5", "qwen3_5_moe"),
        ("qwen", "qwen"),
        ("deepseek", "deepseek"),
        ("kimi", "kimi_k25"),
        ("ernie", "ernie4_5_moe_vl"),
        ("intern", "interns1_pro"),
        ("minimax", "minimax_m2"),
        ("hyv3", "hy_v3"),
        ("hy_v3", "hy_v3"),
        ("mamba", "mamba"),
        ("rwkv", "rwkv"),
        ("retnet", "retnet"),
        ("hyena", "hyena"),
    ]

    @classmethod
    def _resolve_key(cls, model_type: str) -> str:
        normalized = str(model_type or "").lower()
        if normalized in cls._analyzers:
            return normalized

        family_prefixes = [
            ("qwen3_5", "qwen3_5_moe"),
            ("qwen2_moe", "qwen2_moe"),
            ("qwen2", "qwen2"),
            ("qwen", "qwen"),
            ("gemma", "gemma"),
            ("phi", "phi"),
            ("cohere", "cohere"),
            ("bloom", "bloom"),
            ("gptneox", "gpt_neox"),
            ("gpt_neox", "gpt_neox"),
            ("starcoder2", "starcoder2"),
            ("starcoder", "gpt_bigcode"),
            ("stablelm", "stablelm"),
            ("yi", "yi"),
            ("internlm", "internlm2"),
            ("baichuan", "baichuan"),
            ("glm", "glm"),
            ("mamba2", "mamba2"),
            ("mamba", "mamba"),
            ("rwkv", "rwkv"),
            ("retnet", "retnet"),
            ("hyena", "hyena"),
        ]
        for prefix, key in family_prefixes:
            if normalized.startswith(prefix):
                return key

        family_contains = [
            ("llama", "llama"),
            ("mistral", "mistral"),
            ("mixtral", "mixtral"),
            ("deepseek", "deepseek"),
            ("kimi", "kimi_k25"),
            ("chatglm", "chatglm"),
            ("bloom", "bloom"),
            ("neox", "gpt_neox"),
            ("starcoder2", "starcoder2"),
            ("starcoder", "gpt_bigcode"),
            ("bigcode", "gpt_bigcode"),
            ("yi", "yi"),
            ("internlm", "internlm2"),
            ("baichuan", "baichuan"),
            ("minimax", "minimax_m2"),
            ("intern", "interns1_pro"),
            ("ernie", "ernie4_5_moe_vl"),
            ("hy3", "hy_v3"),
            ("hunyuan", "hy_v3"),
            ("mamba", "mamba"),
            ("rwkv", "rwkv"),
            ("retnet", "retnet"),
            ("hyena", "hyena"),
        ]
        for token, key in family_contains:
            if token in normalized:
                return key

        return "default"

    @classmethod
    def get(cls, model_type: str) -> ModelAnalyzer:
        return cls._analyzers[cls._resolve_key(model_type)]

    @classmethod
    def _resolve_architecture_key(cls, architecture_name: str) -> str:
        normalized = str(architecture_name or "").replace("-", "_").lower()
        for token, key in cls._architecture_aliases:
            if token in normalized:
                return key
        return "default"

    @classmethod
    def resolve(cls, config: Any) -> ModelAnalyzer:
        candidates = [
            _cfg_get(config, "model_type", ""),
            _cfg_get(getattr(config, "text_config", None), "model_type", ""),
        ]
        for candidate in candidates:
            key = cls._resolve_key(str(candidate or ""))
            if key != "default":
                return cls._analyzers[key]

        for architecture_name in _architectures(config):
            key = cls._resolve_architecture_key(architecture_name)
            if key != "default":
                return cls._analyzers[key]

        return cls._analyzers["default"]
