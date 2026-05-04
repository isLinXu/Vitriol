# Model Family Coverage

This table reports only model families that have executable validation coverage in this repository.
It does not claim universal compatibility for every model variant inside a family.

## Current Coverage

| Family | Representative Model | Tier | Task Type | `trust_remote_code` | Notes |
|---|---|---|---|---|---|
| `llama` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | `Tier 1` | `causal_lm` | `false` | Baseline decoder-only family |
| `mistral` | `hf-internal-testing/tiny-random-MistralForCausalLM` | `Tier 1` | `causal_lm` | `false` | Standard decoder-only family |
| `gpt2` | `sshleifer/tiny-gpt2` | `Tier 1` | `causal_lm` | `false` | Canonical GPT-style decoder family |
| `opt` | `hf-internal-testing/tiny-random-OPTForCausalLM` | `Tier 1` | `causal_lm` | `false` | Facebook OPT family |
| `bloom` | `hf-internal-testing/tiny-random-BloomForCausalLM` | `Tier 1` | `causal_lm` | `false` | Bloom decoder family |
| `t5` | `hf-internal-testing/tiny-random-T5ForConditionalGeneration` | `Tier 1` | `seq2seq` | `false` | Encoder-decoder baseline |

## Tier Definitions

- `Tier 1`: export + `transformers` read + model load + real forward or generation path pass
- `Tier 2`: export + read + load pass, but real inference is not yet stable
- `Tier 3`: family is only partially validated or still lacks stable evidence

## Evidence Boundary

- The source of truth for the table is the executable matrix in `src/vitriol/compat/family_matrix.py`.
- Broad family smoke validation is driven by `tests/test_hub_smoke_models.py`.
- Deterministic local regressions live in `tests/test_end_to_end_local_generate.py`.
- Family-specific support should only be claimed when corresponding tests actually pass.
