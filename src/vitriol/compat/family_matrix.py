from __future__ import annotations

from typing import Final

FAMILY_MATRIX: Final[list[dict[str, object]]] = [
    {
        "family": "llama",
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": "LlamaAdapter",
        "notes": "Baseline decoder-only family",
    },
    {
        "family": "mistral",
        "model_id": "hf-internal-testing/tiny-random-MistralForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Standard decoder-only family",
    },
    {
        "family": "gpt2",
        "model_id": "sshleifer/tiny-gpt2",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Canonical GPT-style decoder family",
    },
    {
        "family": "opt",
        "model_id": "hf-internal-testing/tiny-random-OPTForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Facebook OPT family",
    },
    {
        "family": "bloom",
        "model_id": "hf-internal-testing/tiny-random-BloomForCausalLM",
        "task_type": "causal_lm",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Bloom decoder family",
    },
    {
        "family": "t5",
        "model_id": "hf-internal-testing/tiny-random-T5ForConditionalGeneration",
        "task_type": "seq2seq",
        "target_tier": "tier1",
        "trust_remote_code": False,
        "expected_adapter": None,
        "notes": "Encoder-decoder baseline",
    },
]
