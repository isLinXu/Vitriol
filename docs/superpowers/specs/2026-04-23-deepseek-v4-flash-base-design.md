# DeepSeek-V4-Flash-Base Design

## Goal

Build a stable, evidence-backed support path for `deepseek-ai/DeepSeek-V4-Flash-Base` in Vitriol so the repository can:

1. export an `ultra` minimal-weight artifact,
2. validate that artifact with `validate --no-inference`,
3. and produce usable architecture and weight visualizations.

The goal is not to claim universal support for all DeepSeek-V4 variants. The goal is to establish one real, repeatable `DeepSeek-V4-Flash-Base` closed loop that can serve as the baseline for future V4-family expansion.

## Why This Matters

Vitriol already has practical DeepSeek-family support, especially around the existing `deepseek` analyzer and MLA-aware family handling. However, `DeepSeek-V4-Flash-Base` is a new large MoE base model and may expose gaps in:

- family recognition,
- config loading and fallback,
- ultra export assumptions,
- tokenizer metadata preservation,
- or visualization expectations.

Without a real end-to-end run on the concrete Hugging Face model, current support remains an inference rather than validated compatibility evidence.

## Scope

This design covers:

- `deepseek-ai/DeepSeek-V4-Flash-Base` only,
- `ultra` export through the existing `generate` CLI path,
- load validation through `validate --no-inference`,
- static architecture visualization via `arch-viz`,
- and weight visualization via `weight-viz`.

This design intentionally does not include in the first pass:

- real inference validation,
- generic support claims for all DeepSeek-V4 repositories,
- multimodal support,
- or a broad DeepSeek family refactor.

Those can be follow-up tasks once this single-model baseline is stable.

## Success Criteria

The work is successful when Vitriol can produce a concrete output directory for `DeepSeek-V4-Flash-Base` that contains:

- `config.json`
- `meta-config.json`
- `pytorch_model.bin.index.json`
- `vitriol-manifest.json`
- `architecture.html`
- `architecture.png`
- `architecture_detail.png`

And when the following evidence-backed checks pass:

- `generate --strategy ultra` completes successfully,
- `validate --no-inference` reports that the model and tokenizer are loadable,
- `arch-viz` completes and renders the expected files,
- `weight-viz` starts successfully against the exported directory,
- any family-specific compatibility fix is locked by a deterministic regression test.

## Working Assumption

The first implementation pass should assume that `DeepSeek-V4-Flash-Base` is close enough to the current `deepseek` family path that the repository should reuse existing support first.

That means the implementation should prefer:

- existing `DeepSeekAnalyzer`,
- existing DeepSeek family patch logic,
- existing generic config fallback behavior,
- and existing visualization entry points.

Only when the real run exposes a failure should the code add model-specific or family-specific compatibility logic.

## Primary Approach

The implementation should follow a "run the real path first, then patch the real failure" strategy.

Recommended sequence:

1. inspect the upstream config and attempt the existing `ultra` export path,
2. record the first concrete failure point if any,
3. apply the smallest compatible fix in the correct layer,
4. rerun export,
5. run `validate --no-inference`,
6. generate architecture artifacts,
7. launch or verify `weight-viz`,
8. add regression coverage for the exact compatibility fix.

This keeps the work grounded in real breakpoints instead of speculative adaptation.

## Compatibility Surfaces To Watch

The implementation should explicitly watch these surfaces during the first run:

### 1. Model Identification

Potential issues:

- new `model_type` not recognized by existing analyzer or adapter lookup,
- architecture aliases that resolve incorrectly,
- or family detection that falls back to a generic transformer path.

Expected response:

- prefer extending family resolution or aliases rather than introducing an isolated one-off branch unless the model truly diverges from current DeepSeek handling.

### 2. Config Loading

Potential issues:

- upstream config requiring `trust_remote_code`,
- a partially unknown config structure,
- nested text config fields,
- or raw fallback behavior returning an object the generator or parser does not expect.

Expected response:

- reuse `load_config_or_raw()` and related raw-config helpers,
- patch only the minimal field projection or object-shape assumption that blocks export or visualization.

### 3. Ultra Export

Potential issues:

- DeepSeek-V4-specific RoPE, MLA, or MoE fields not preserved correctly,
- generated manifest or index inconsistencies,
- unsupported non-persistent buffers,
- or dimension assumptions that do not match current DeepSeek family behavior.

Expected response:

- put export-time fixes into generator or family patch logic where the invariant belongs,
- avoid hidden, model-ID-only special casing unless no reusable family rule exists.

### 4. Validation

Potential issues:

- tokenizer metadata loss,
- auto-loader mismatch,
- config fields that load during export but fail during validation,
- or validation assuming a different model task shape.

Expected response:

- keep the first pass focused on `--no-inference`,
- treat loadability as the compatibility boundary for this phase,
- patch validation only where it misclassifies or mishandles a supported export artifact.

### 5. Visualization

Potential issues:

- analyzer misidentifying the model family,
- architecture parameter estimates being obviously wrong,
- HTML/block/detail renderers not surfacing meaningful DeepSeek structure,
- or `weight-viz` depending on config fields absent from the exported artifact.

Expected response:

- keep visualization fixes incremental,
- prefer family detection and metadata improvements before renderer-only hacks,
- only add renderer-specific work when the analyzer already has the right structure data.

## Expected Code Areas

The work will likely touch only a subset of these files, depending on the first real failure:

- `src/vitriol/utils/hf_loading.py`
- `src/vitriol/core/generator.py`
- `src/vitriol/patches/model_family_patches.py`
- `src/vitriol/arch_viz/analyzers.py`
- `src/vitriol/arch_viz/parser.py`
- `src/vitriol/tools/model_demo.py`
- `tests/test_hf_loading.py`
- `tests/test_more_regressions.py`
- `tests/test_smoke_vitriol.py`

If the model runs through existing support unchanged, the implementation should avoid broad edits and instead add only the evidence-producing tests and docs updates that are justified by the successful run.

## Test Strategy

### 1. Real Artifact Validation

The core proof should come from real CLI execution, not only synthetic unit tests.

The expected command sequence is:

- `generate <model> --strategy ultra`
- `validate <output> --no-inference`
- `arch-viz <output>`
- `weight-viz -m <output>`

This is the support claim boundary for this task.

### 2. Focused Regression Coverage

If any compatibility issue is fixed, the repository should add or update deterministic tests that cover the exact failure mode.

Examples:

- unknown or nested DeepSeek-V4 config fallback,
- DeepSeek-specific tokenizer metadata preservation,
- analyzer resolution for a new `model_type` or architecture alias,
- ultra export field cleanup or invariant handling.

The tests should be local and fast. They should not depend on the remote Hugging Face model.

### 3. Visualization Regression

If the analyzer or visual output changes, add HTML or architecture assertions that confirm the exported DeepSeek-V4 artifact renders through the intended family path instead of a broken generic fallback.

This should be lightweight and tied only to the new compatibility surface, not to unstable presentation details.

## Output Naming

The implementation should keep output paths explicit and model-specific to avoid mixing artifacts with earlier DeepSeek runs.

Recommended output directory:

- `output/deepseek_v4_flash_base_ultra`

This path makes later validation, re-rendering, and regression references easier.

## Risks And Constraints

- The upstream repository may require `trust_remote_code=True`.
- The upstream model may expose a new DeepSeek-family `model_type` even if the architecture is mostly compatible with current DeepSeek handling.
- The model is large, so the first run may surface performance or artifact-size constraints that smaller reference models do not.
- Some visualization quality issues may be acceptable in phase one if export and validation succeed and the visual path remains functional.

These are acceptable constraints as long as the final result is explicit about what is proven and what is not.

## Decision Summary

This task should use a conservative, evidence-first approach:

- reuse current DeepSeek support first,
- run the real `ultra` export path,
- patch only concrete failures,
- require `validate --no-inference`,
- require both architecture and weight visualization support,
- and lock any fix with focused regression coverage.

That approach provides a trustworthy `DeepSeek-V4-Flash-Base` baseline without prematurely broadening the DeepSeek family surface area.
