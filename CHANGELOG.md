# Changelog

All notable changes to the Vitriol project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed

- Restart failed or unexpectedly exited distributed coordinator and worker loops with bounded exponential backoff; propagate worker heartbeat failures to the supervisor and retain cancellation-aware shutdown.
- Roll back partial distributed dispatch failures, restore tasks at the FIFO head without charging task retries, and route scheduler errors through supervised exponential backoff.
- Preserve distributed pending-task FIFO order while workers are unavailable, and wake dispatch on queued work or newly available capacity instead of polling every second.
- Preserve direct Queue.put/put_nowait notifications and Python 3.9 event-loop initialization while keeping retried or reassigned work behind existing pending tasks.

- Measure distributed heartbeat and execution watchdogs with monotonic clocks while keeping public timestamps in Unix time; reset and release execution timing across retries, disconnects, completion, and restart.
- Respect short execution timeouts with a bounded watchdog polling interval, expire attempts at the deadline, and reject nonpositive or nonfinite watchdog configuration.

- Use monotonic elapsed time and remaining wait budgets for distributed task waits; make nonpositive timeouts immediate and reject NaN without cancelling task execution.
- Copy submitted shard lists and return detached task status snapshots, including nested results, so caller mutations cannot change queued work through those interfaces.

- Make distributed coordinator and worker startup idempotent, join owned coroutines on shutdown, and preserve unfinished tasks for restart.
- Retire disconnected or timed-out execution attempts before retry so late results cannot overwrite reassigned work or double-count completion; handle zero start timestamps in timeout checks.

- Initialize distributed task queues on their execution loop so Python 3.9 supports synchronous coordinator construction after asyncio.run; preserve pending tasks when a worker unregisters before the loop starts.
- Keep the Python CI matrix on Ubuntu 24.04 for Python 3.9 availability, and install declared build dependencies through isolated distribution builds.
- Allow machine-level rounding in the identical-architecture cosine similarity regression.

- Close inherited output-lock descriptors after Python fork without unlocking the parent; serialize descriptor registration with fork and preserve active registrations across module reloads.
- Prevent inherited context cleanup from closing reused child descriptors, and preserve the original generation error when explicit unlock also fails.

- Exclude concurrent generators from the same output directory with a nonblocking OS lock covering the full public generation call, including aliases and reentrant calls.
- Release directory ownership after generation errors and clean reserved orphan temporary artifacts only after acquiring the lock.

- Journal final shard renames before moving files, recover partially completed renames, and synchronize checkpoint/index/generator paths after publication.
- Replace stale final shard destinations, publish index JSON atomically, and stop generation when required checkpoint commits fail.
- Account sparse descriptors through their nbytes protocol while preserving compact tensor storage accounting across module reloads.

- Restore default RNG streams at committed shard boundaries; validate checkpoint digests, accounting, ordered progress and runtime compatibility before resuming.
- Read Safetensors checkpoint metadata without materializing weights, account SparseSpec output correctly, and preserve completed empty shards.
- Make Quantum scaling stable across processes without reseeding global RNG, and include standard PyTorch format metadata for Transformers loading.
- Regenerate learned strategies on restart when their network state is absent, avoiding mixed-generator artifacts.

- Preserve every generated tensor across shard flushes and validate persisted checkpoints before resuming; write shards and checkpoints atomically.
- Reject generation into the source model directory and non-local shard filenames.
- Reject missing, unexpected, or mismatched weights during validation instead of certifying automatically initialized replacements; reset reports between attempts.
- Preserve offline policy when resolving trusted custom model classes and restore active configuration after visualization errors.
- Restore lazy public exports used by the `bench` CLI after module extraction.
- Correct QJL attention dimensions and mixed precision, sliding-window proxy invalidation, masked/padded attention, dropout scaling, and adaptive single-token quantization.
- Make shape-preserving Hadamard rotations invertible for non-power-of-two dimensions, with float32 accumulation for half precision.
- Keep adaptive KV precision per batch across incremental appends, restore the original rotation basis, and reallocate after cache rebuilds; support GQA and distinct key/value dimensions.
- Compute adaptive precision from original Q/K coordinates in float32, avoiding half-precision overflow and zero-entropy NaNs.
- Bound sliding-window recent-token reserves by the cache budget and handle a zero allocation for older tokens correctly.
- Preserve SDPA and automatic attention selection when FlashAttention is unavailable, so compatibility initialization does not silently bypass KV runtime patches.
- Validate runtime preprocessing cache entries against live tensor identity, version and layout; bypass reuse for untracked inference tensors, autograd graphs and autocast contexts, and release entries on restore.
- Make runtime patch ownership explicit, reject conflicting active instances, preserve external wrappers, and support exception-safe nested contexts without repopulating caches after restoration.
- Forward positional attention arguments, custom scaling and GQA through compute-skip/sparse paths; preserve batched masks, dropout scaling, double precision and finite gradients for fully masked rows.
- Give each runtime configuration its own mutable adaptive codec.
- Support both mapping-based and legacy Transformers attention registries; bind backend reads to the actual returned K/V tensors instead of a thread-global cache handle.
- Retain complete native KV history for reliable backend-failure fallback, invalidate sidecar stores after cache mutations, seed warm caches from native history, and keep handles usable after hooks are removed.
- Make cache/attention hook ownership and combined installation rollback explicit, preserve external wrappers and native mask offsets, and expose fallback layers and native-cache retention in benchmark metadata.
- Score generated tokens using their causal logits, restore inference patches on failure, and report unmeasured KL/layer metrics as unavailable.
- Honor API defaults and nested security overrides, return HTTP 429 for rate limits, keep generation off the event loop, and fix job counts and log streaming after buffer overflow.
- Use supplied NAS evaluator scores instead of random quality scores.
- Support low-precision and degenerate model fingerprints; keep weight-statistics sampling bounded for noncontiguous and broadcast tensors.
- Install API test dependencies in CI and separate the integration suite from the full-suite coverage threshold.

### Changed

- Publish pytest failures as GitHub annotations and retain JUnit/coverage reports for seven days, including failed CI runs.

- Concurrent generation into an active output directory raises recoverable OutputDirectoryBusyError. The persistent .vitriol-generation.lock marker is retained; OS locking determines whether it is busy.

- New checkpoint content digests use a versioned, allocation-independent chunked SHA-256 format that skips sparse holes when supported; previous whole-file SHA-256 checkpoints remain readable.

- Generation checkpoints now use a versioned signature with RNG state and SHA-256 shard digests. Pre-RNG checkpoints regenerate; dense files and filesystems without extent support still require sequential reads.

- PPL reports now label self-generated continuations and estimated KV payload memory explicitly. Unknown KL values serialize as `null` rather than synthetic numbers.
- Preserve the existing configuration precedence: explicit/CLI values > YAML > environment > defaults.
- Non-power-of-two shape-preserving KV rotations now use consecutive power-of-two blocks (e.g. 80 = 64 + 16). Rebuild experimental rotated caches produced by older code; the signed, padded TurboQuant/TurboQuantum transforms retain their existing layouts.
- Generic cache hooks now always update native KV, including legacy `passthrough_update=False` requests (which log a warning). Store payload estimates exclude this native copy and must not be interpreted as total KV memory savings.

---

## [0.3.1] - 2026-06-18

### Added

- **`vitriol check`** — Structure-First golden path (analyze → arch-viz → generate → validate → fingerprint → HTML report)
- **`vitriol cis`** — CIS ranking (`rank`, `table`, `score`, `report`, **`compare`**)
- **`vitriol cis compare`** — Multi-strategy benchmark: generate → validate → empirical CIS
- **Composite GitHub Action** — `.github/actions/vitriol-check` for one-line CI integration
- **Case studies** — `docs/case-studies/01` through `04` (architecture compare, CI validation, CIS ranking, multi-strategy compare)
- **Golden-path integration tests** — `tests/integration/` with nightly CI workflow
- **Capability tiers** — Stable / Beta / Experimental matrix in README
- **`CompressionIntelligenceScorer.score_all_strategies()`** — API referenced by README now implemented
- **Generator modular split** — `config_loader`, `custom_code_sync`, `generator_persistence`

### Fixed

- **`ConfigManager.get_environment()`** — Restored method accidentally merged into `is_valid()`
- **`PluginManager.unload_plugin()`** — Implemented missing API
- **Validator security** — seq2seq loads propagate `trust_remote_code` via `hf_kwargs()`
- **`generator_persistence` viz** — Default `trust_remote_code=False` (was True)
- **`experimental` decorator** — Preserves Click Command objects for CLI compatibility
- **`_FALLBACK_CHAIN` import** — Restored in `generator.py` after modular split

### Changed

- CLI command count: **19** top-level commands (+ `check`, + `cis` group)
- Default recommended entry: `vitriol check`

---

## [0.3.0] - 2026-04-30

### :star: Major New Features

**ExoBrain — Ultra Shell External Brain Inference System**
- `exobrain infer` — End-to-end inference pipeline: Shell model + Teacher KV extraction → ExoBrain injection → generate()
- `exobrain distill` — Knowledge distillation: Teacher KV → Shell weight distillation (MSE/KL/Cosine loss → gradient update → save)
- ExoBrainBus: unified knowledge retrieval from VectorDB / API / LocalWeight sources
- ExoBrainAttentionPatcher: Attention interception with Prefill + Decode support
- 3 fusion modes: replace / residual / gated
- 3 knowledge sources: VectorDBSource / APIKnowledgeSource / LocalWeightSource

**ExoBrain v0.5 Optimizations**
- AdaptiveLayerSelector: attention-entropy-based adaptive layer selection (4 strategies)
- KVPrefetcher: pre-cache projection KV for zero-redundancy decode retrieval
- Contrastive Loss: InfoNCE-style semantic alignment (contrastive_weight + temperature)
- Per-Head Entropy Gating: independent gating per attention head (gate_mode="per_head_entropy")
- ExoBrainEvaluator: quantitative injection quality (Attention Entropy Shift, Logit Divergence, Top-1 Agreement)

**ExoBrain v0.6 Optimizations**
- MultiTeacherRouter: multi-teacher KV integration with dynamic routing (similarity/ensemble/round_robin/first_available)
- AdaptiveInjectionScheduler: PPL-based adaptive injection scheduling (threshold/relative/entropy/always/never)
- BrainKVCompressor: external brain KV compression (topk_spatial/quantize_8bit/mean_pool/svd_lowrank)
- ProgressiveDistiller: progressive knowledge solidification (5-stage α_brain decay 1.0→0.0)
- ExoBrainProfiler: full-chain performance profiler (context manager + bottleneck detection)

**Innovative KV Cache Modules**
- `cross_layer.py` — CrossLayerKV: cross-layer differential compression (I-frame/P-frame, SNR: 20.1 dB @ 3.0 bpv)
- `attention_gated.py` — AttentionGatedKV: attention-gated variable-precision (3-tier quantization, SNR: ~11.8 dB @ 3.83 bpv)
- `dict_kv.py` — DictKV: dictionary sparse coding (OMP + K-SVD, compression ratio scales with dim: d=1024→29.5×, d=4096→118×)
- SpectralKV: frequency-domain KV cache compression
- PredictiveKV: predictive temporal KV compression

**New CLI Commands**
- `infer` — Single-prompt inference with TurboQuant presets
- `trace` — Generate offline trace.json for replay
- `exobrain` — ExoBrain inference & knowledge distillation (with infer/distill sub-commands)
- 18 total commands now available

**New Weight Generation Strategy**
- `HybridUltra` — Hybrid strategy combining multiple approaches for optimal compression

### :hammer: Enhancements

**Security & Trust**
- `trust_remote_code` fully parameterized across CLI/API/WebUI/core (no hardcoded True)
- WebUI: 3 tabs with trust_remote_code Checkbox toggles (Compare/Simulator/Scorecard)
- API: process_generation_job always writes trust_remote_code/allow_network/local_files_only
- CLI: `--trust-remote-code/--no-trust-remote-code` and `--offline` flags

**API**
- `/models` endpoint now dynamically generated from DEFAULT_FAMILIES + AdapterRegistry + STRATEGY_REGISTRY
- New sub-endpoints: `/models/families` and `/models/adapters`
- Fixed missing os/json imports in server module

**Project Renaming: Archon → Vitriol**
- Complete global rename: package name, CLI entry point, class names, config paths, environment variables
- Environment variables: ARCHON_* → VITRIOL_*
- Class names: ArchonConfig → VitriolConfig, ArchonError → VitriolError
- Config path: ~/.config/archon/ → ~/.config/vitriol/
- CI variables: ARCHON_CI_TRUST_REMOTE_CODE → VITRIOL_CI_TRUST_REMOTE_CODE

**KV Cache System**
- 17 KV Cache modules with 17 strategy presets
- TurboQuantum: 4 quantization presets (turbo2/turbo3/turbo4)
- Layer-adaptive, temporal-pooling, hybrid-pipeline optimization modules
- Triton-accelerated kernels: FWHT, block quantization, bit packing

### :bug: Fixes
- Fixed trust_remote_code not propagated in API/WebUI paths
- Fixed /models endpoint serving static data instead of dynamic registry
- Fixed missing os/json imports in API server module
- Fixed PPL evaluation hook compatibility with the benchmark runner's `v_quantize_only_first_n_layers` signature
- Fixed experimental RL NAS search-space compatibility (`from_config`, `sample_random`, `validate_gene`, and CLI `--episodes` flow)

### :books: Documentation
- README rewrite with ExoBrain, KV innovations, and HybridUltra documentation
- Chinese README fully synchronized
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md updated for Vitriol naming
- Added NAS/PPL compatibility notes and release validation checklist

---

## [Unreleased]

### Security
- API authentication now accepts `X-API-Key` and `Authorization: Bearer` headers, while retaining the legacy `api_key` query parameter for compatibility.
- Sensitive API read endpoints (`/status`, `/jobs`, `/batch/{id}`, `/models`, `/strategies`, `/stream/logs`) now honor `security.api_key_required`; `/` and `/health` remain public.
- Dynamic remote-module patching is skipped unless `trust_remote_code=True` is explicitly enabled.
- Custom-code synchronization is restricted to whitelisted HuggingFace module names, `auto_map`-referenced Python files when metadata is available, safe output paths, configurable file counts, and configurable per-file byte limits.

### Changed
- Runtime dependency constraints now pin NumPy to `>=1.21,<2` and PyTorch to `>=2.0.0,<3.0.0` to reduce accidental major-version breakage.
- **Internal module decomposition** (no public API/import-path changes; behavior preserved):
  - `arch_viz`: split the 2.7k-line `_analyzers_legacy.py` into 14 focused analyzer modules under `arch_viz/analyzers/`.
  - `kv/exobrain`: split the two ExoBrain megaliths (`exobrain.py`, `exobrain_inference.py`) into the acyclic `kv/exobrain/` package; legacy import paths preserved via re-exports/shims.
  - `arch_viz/renderers`: split the 3.3k-line `HTMLRenderer` god-class into four presentation mixins (`_html_styles/_html_columns/_html_scripts/_html_sections`); rendered HTML is byte-for-byte identical.
  - `cli/commands/bench.py` (2088→757 lines): extracted result/markdown/summary formatting helpers into `bench_format.py`.
  - `bench/runner.py` (1736→~1330 lines): extracted the policy-planning leaf cluster into `bench/_planning.py`; all `vitriol.bench.runner.*` paths re-exported.
- `GenerationConfig` now validates `dtype`, `max_shard_size`, `rank`, and the numeric types of `n_bits`/`sparsity` (existing `ValueError` messages preserved); `build_generation_config` rejects unknown config keys with an actionable error instead of a cryptic `TypeError`.
- The REST API server now guards its `fastapi`/`uvicorn`/`pydantic` imports, raising an actionable install hint (`pip install 'vitriol[api]'`) when the `[api]` extra is missing.

### Added
- GitHub Pages deployment with automated CI/CD pipeline
- `CONTRIBUTING.md` contribution guidelines
- `CHANGELOG.md` version history tracking
- Issue and Pull Request templates
- `@experimental` decorator (`vitriol.utils.experimental`) marking functions/classes experimental at runtime via a one-shot `ExperimentalWarning` (silenceable with `VITRIOL_SILENCE_EXPERIMENTAL=1`); applied to the RL searcher and the REST API server.
- Optional-dependency helpers (`vitriol.utils.optional`): `require()`, `has()`, and `MissingDependencyStub`, plus a `MissingOptionalDependencyError` that subclasses `ImportError` and carries pip / `vitriol[extra]` install hints.
- JSON Schema contract for the generation config: `generation_config_schema()` (draft-07) and `validate_generation_dict()`, which performs full structural validation when the optional `jsonschema` package is installed.

---

## [0.2.0] - 2026-04-02

### :star: Major New Features

**Architecture Evolution System**
- `evolve tree` — Build interactive D3.js evolution trees (120+ nodes, 17 families)
- `evolve compare` — Generate architecture comparison reports (Markdown/JSON/HTML)
- `evolve simulate` — Performance simulator (params, VRAM, FLOPs, KV Cache by GPU type)
- `evolve families` — List known model families

**Model Hash Fingerprinting**
- Architecture Hash — structural fingerprint of model config
- Weight Distribution Hash — statistical fingerprint of weight tensors
- Behavioral DNA Hash — output-based behavioral signature
- Vitriol Signature — unified composite identifier
- Supports both Transformers and Diffusers models

**Quantized Inference & KV Cache Compression**
- TurboQuant: turbo2/turbo3/turbo4 KV cache quantization (up to 6.4x compression)
- Adaptive KV Codec: entropy-based adaptive bit-width + Walsh-Hadamard rotation
- Sparse V: attention-gated KV decoding for low-attention blocks
- Compute Skip Attention: block-level KV skipping
- KV Cache Policy presets: safe / balanced / aggressive

**Compression Intelligence Score (CIS)**
- Four-dimensional evaluation: Information Preservation × Storage Efficiency × Expressivity × Trainability
- Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train
- CriticalPointDetector: detects ~90% compression phase transition
- Strategy ranking: learned(0.84) > lowrank(0.71) > quantized(0.69) > random(0.65) > ultra(0.35)

**New CLI Commands**
- `hash` — Generate model hash fingerprints
- 16 total commands now available

**New Weight Generation Strategies**
- `learned` — Neural network-based weight generation (WeightGeneratorNetwork)
- `hybrid_learned` — Hybrid: learned for attention/embedding + compact for others
- 13 total strategies

**GitHub Pages & Documentation**
- Online 3D viewer: `https://islinxu.github.io/Vitriol/viewer.html`
- Remote HuggingFace model loading (`viewer.html#?hf=org/repo`)
- Model Zoo with demo configs (Qwen3.5-397B, Qwen3 Demo, DeepSeek V3 Demo)
- Interactive evolution tree visualization (D3.js)
- Innovation timeline (2019–2024)
- Modern dark-themed index page with responsive design

### :hammer: Enhancements

**Core Engine**
- `meta-config.json` support: preserves original HuggingFace config alongside shrunk config
- Nested `text_config` parsing for vision-language models
- Config load priority: meta-config.json > config_meta.json > config.json

**Visualization**
- 3D viewer enhancements: module info panel, data flow particle animation, right-click menu, keyboard shortcuts, search, hover tooltips, model comparison mode, export (PNG/JSON)

**NAS**
- Targeted NAS algorithm: constraint optimizer, multi-objective Pareto optimization, directed mutation
- Support for constraints: max_vram, max_params, max_layers, attention_type
- Support for objectives: minimize_params, minimize_vram, maximize_efficiency

**CI/CD**
- Three GitHub Actions workflows: CI, Hub-Smoke, Pages
- Matrix testing with trust_remote_code variants
- Auto-deploy docs/ to GitHub Pages on push to main

### :bug: Fixes
- Fixed Session Context race condition in storage layer (transactional save)
- Fixed validate command incompatibility with Ultra strategy
- Fixed nested text_config parsing failure in analyzers

### :books: Documentation
- Complete README rewrite (~1000 lines, 28 sections) in English + Chinese
- Quantified value analysis: storage savings, time savings, cost reduction
- Model Zoo, 3D Visualization, Demo & Screenshots sections added

---

## [0.1.0] - 2026-03-31

### :star: Initial Release

**Core Features**
- **Weight Generation**: 10 strategies (random, compact, ultra, sparse, structured_sparse, ternary, binary, quantized, lowrank, quantum)
- **Architecture Visualization**: 2D HTML diagrams + 3D Three.js WebGL viewer
- **Architecture Analysis**: 9 built-in analyzers (attention patterns, FFN structure, RoPE config, etc.)
- **Neural Architecture Search**: random and evolutionary algorithms
- **CLI Interface**: 13 commands (generate, analyze, batch, export, visualize, viz, arch-viz, nas, vocab-viz, weight-viz, evolve, webui, validate)
- **Model Adapters**: LLaMA, Qwen, DeepSeek + DefaultAdapter
- **WebUI**: Gradio-based interface
- **Experimental API**: FastAPI REST server
- **Patch System**: 10 model-specific patches including Qwen3.5
- **KV Cache**: backend, codec, cache_store, policy modules

**Architecture**
- Structure-data decoupling paradigm for zero-cost architecture exploration
- Ultra strategy stride=0 hack achieving ~4.5M:1 compression ratio
- Plugin system for extensibility
- Resilience and distributed computing modules

---

[Unreleased]: https://github.com/isLinXu/Vitriol/compare/v0.3.1...main
[0.3.1]: https://github.com/isLinXu/Vitriol/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/isLinXu/Vitriol/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/isLinXu/Vitriol/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/isLinXu/Vitriol/releases/tag/v0.1.0
