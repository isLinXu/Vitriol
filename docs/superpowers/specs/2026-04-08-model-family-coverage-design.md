# Model Family Coverage Design

## Goal

Build a stable, executable coverage matrix for Archon's exported model weights so the project can verify, with real evidence, which HuggingFace model families can be:

1. exported by Archon,
2. reloaded by `transformers`,
3. and executed for at least one real forward or generation path.

The primary goal is not to claim universal compatibility. The primary goal is to establish a trustworthy baseline for broad text-model-family coverage that can be extended over time.

## Why This Matters

Current validation is strong for several individual paths, but the framework does not yet provide a single, explicit answer to:

- which model families are covered,
- which ones are only partially covered,
- and which failures are known and intentionally deferred.

Without a family-level matrix, support claims are difficult to verify, regressions are easier to miss, and adapter or shrink-specific fixes tend to remain local instead of becoming reusable framework guarantees.

## Scope

This design covers:

- text-oriented HuggingFace model families first,
- export with `MinimalWeightGenerator`,
- reload via `transformers` auto classes,
- real validation through load and inference checks,
- adapter- and shrink-related compatibility fixes required to make the matrix meaningful.

This design does not attempt in the first phase to fully cover:

- all multimodal families,
- all custom remote-code repositories,
- every community variant on HuggingFace,
- or every non-generation task type.

Those remain allowed future expansions once the text-family baseline is stable.

## Success Criteria

The design is successful when Archon can produce an evidence-backed coverage table where each listed model family is assigned one of three tiers:

- `Tier 1`: export + `transformers` read + model load + real forward or generation all pass.
- `Tier 2`: export + `transformers` read + model load pass, but real inference still fails or is not yet stable.
- `Tier 3`: family is theoretically reachable or partially compatible, but has no stable validation evidence yet.

The first implementation pass should prioritize getting a credible `Tier 1` baseline for mainstream text families rather than maximizing raw family count.

## Initial Family Set

The first coverage wave should focus on representative mainstream text families:

- `Llama`
- `Mistral`
- `Qwen`
- `Qwen-MoE`
- `Qwen3.5-MoE`
- `DeepSeek`
- `GLM / glm_moe_dsa`
- `GPT2`
- `OPT`
- `Bloom`
- `T5`

This list is intentionally mixed:

- standard CausalLM families,
- adapter-dependent families,
- encoder-decoder coverage,
- MoE coverage,
- and MLA-style or custom attention structure coverage.

## Coverage Model

Coverage should be represented as executable test data rather than free-form documentation.

Each family entry should define:

- `family`: stable family label used in reports and docs.
- `model_id`: representative HuggingFace model.
- `task_type`: one of `causal_lm`, `seq2seq`, or `generic`.
- `target_tier`: expected validation tier.
- `trust_remote_code`: whether the representative path requires it.
- `expected_adapter`: adapter name when family support depends on adapter registration.
- `notes`: short human-readable limitation or rationale.

This structure becomes the source of truth for both tests and the published compatibility summary.

## Test Strategy

### 1. Family Smoke Matrix

Extend the existing hub smoke pattern into a family-aware matrix test.

Each matrix row should verify:

- `generate()` completes,
- `config.json`, `meta-config.json`, index, and manifest are produced,
- `meta-config.json` matches the upstream source config when applicable,
- a compatible `transformers` auto loader can read the output directory.

This layer is responsible for broad family reach and artifact integrity.

### 2. Tier 1 Inference Validation

For rows marked `Tier 1`, require a real execution path after load:

- `AutoTokenizer.from_pretrained(...)`
- `AutoConfig.from_pretrained(...)`
- `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` or `AutoModel`
- one real forward or `generate()` path

This layer is the actual support claim boundary.

### 3. Family-Specific Regression Tests

Families that require custom logic must also get dedicated, local, deterministic tests.

Examples:

- `qwen3_5_moe`: config flattening and auto-class registration
- `glm_moe_dsa`: shrink consistency for QK and V dimensions
- future encoder-decoder families: task-type-specific loading and inference behavior

These tests prevent broad support from silently regressing when the generator or adapter logic changes.

## Reuse Of Existing Framework Paths

The implementation should reuse current project entry points instead of adding a second validation system.

Primary reuse points:

- `tests/test_hub_smoke_models.py` for broad family smoke coverage
- `tests/test_end_to_end_local_generate.py` for deterministic local family regressions
- `src/archon/core/validator.py` for real load and inference verification
- `src/archon/adapters/*` for family-specific config and auto-class compatibility
- `src/archon/core/generator.py` for family-specific shrink and reconcile behavior

## Expected Implementation Changes

### Coverage Matrix Definition

Add a shared family-matrix definition that tests can consume directly.

This should avoid duplicating family metadata across multiple test files and should become the canonical mapping from family name to representative model and expected validation tier.

### Validator Generalization

The validator currently centers on `AutoModelForCausalLM` first, with generic fallback.

To support family-level claims cleanly, validation should become more task-aware:

- use `task_type` to prefer the right auto loader,
- keep generic fallback for compatibility checks,
- only mark `Tier 1` as passed when the expected inference path succeeds.

### Adapter Expansion

Current adapter coverage is concentrated in:

- `Llama`
- `DeepSeek`
- `Qwen / Qwen3.5-MoE`

The family matrix will likely expose gaps where:

- no adapter exists,
- a family works only accidentally through fallback loaders,
- or load succeeds but saved configs are not reliable enough for inference.

When that happens, fixes should go first into the adapter layer rather than ad-hoc test exceptions.

### Generator Family Exceptions

When a family needs export-time dimension or config handling, the rule should live in the generator as an explicit family-specific invariant, not as a hidden side effect of generic shrink logic.

The `glm_moe_dsa` head-dimension fix is the reference pattern:

- identify the family invariant,
- encode it explicitly,
- then lock it with a deterministic regression test.

## Reporting

The framework should eventually publish a concise compatibility summary derived from the matrix and test evidence.

That summary should state:

- tested family,
- representative model,
- achieved tier,
- whether `trust_remote_code` is required,
- and any known limitation.

The report must be evidence-driven and should not claim support beyond executed validation.

## Rollout Order

Recommended rollout order:

1. extract the current known-good families into a first matrix,
2. make the family smoke matrix executable,
3. add `Tier 1` inference checks for the core mainstream text families,
4. fix family-specific failures in adapters or generator logic,
5. add dedicated regressions for every family-specific fix,
6. publish the resulting tier table.

This order creates a baseline quickly while keeping the work aligned to real compatibility evidence.

## Risks And Constraints

- Hub-hosted representative models may change, so smoke selection should prefer stable or tiny reference models where possible.
- Some families may only be representable with `trust_remote_code=True`; those must be clearly labeled rather than silently mixed with standard families.
- Full multimodal coverage is out of scope for the first pass and should not block text-family completion.
- Broad matrix growth can become expensive; `Tier 1` should stay focused on high-value families, while deeper expansions can remain opt-in.

## Decision Summary

This design chooses:

- model-family coverage as the primary support unit,
- `export + load + inference` as the minimum standard for real support claims,
- a tiered matrix instead of binary supported/unsupported labels,
- and reusable adapter/generator fixes over one-off special cases.

The intended outcome is a framework that can say, with real test evidence, not just that it can export weights in isolated cases, but which model families it can cover end to end and where the remaining gaps still are.
