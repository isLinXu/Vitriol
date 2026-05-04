# Archived TurboQuant Gap Report

> Historical note retained for project archaeology. This report reflects a **pre-Apr-2026** snapshot and is **not** the current source of truth for public TurboQuant alignment claims.
>
> For the current externally-facing assessment, use [kv-turboquant-qwen35-0.8b-alignment.md](./kv-turboquant-qwen35-0.8b-alignment.md).

## Why This File Is Archived

This document originally captured real gaps in Vitriol's TurboQuant-related implementation, but several of its statements are now outdated because the codebase has since added:

- packed KV storage through the KV-store path
- residual QJL packed tensors
- proxy attention paths that score directly from packed residual-aware representations
- regression coverage for the previously no-op Qwen/runtime TurboQuant paths

As a result, this file should no longer be read as a current implementation verdict.

## What Is Still Historically Useful

The older report remains useful for understanding the project history:

- why the repository originally separated storage-oriented KV compression from paper-faithful TurboQuant claims
- which correctness failures existed in earlier runtime patch paths
- why later work focused on residual QJL, packed scoring, and long-context validation

## What Is No Longer Accurate As Written

The following older claims are no longer accurate in their original absolute form:

- "No residual QJL stage"
- "Packed KV path fully decodes before attention" as a universal statement
- "TurboQuant path is only a mock" as a repository-wide conclusion
- "Current implementation should not yet be described as TurboQuant-aligned" without qualification

Those statements described an earlier state of the codebase, not the current one.

## Current Public Boundary

The current public-facing boundary should be stated more narrowly:

- Vitriol matches the paper/reference direction in bit semantics, layer-selection intent, and residual-aware packed KV engineering.
- Vitriol has real long-context correctness evidence on `Qwen/Qwen3.5-0.8B`.
- Vitriol has measured KV-memory-estimate improvements on real runs.
- Vitriol does **not** yet have a complete public proof of paper-faithful end-to-end effect alignment, especially for ultra-long contexts and consistent peak-memory / throughput wins across hardware backends.

## Source Of Truth

For release notes, README claims, and open-source positioning, treat the following as authoritative instead of this file:

- [kv-turboquant-qwen35-0.8b-alignment.md](./kv-turboquant-qwen35-0.8b-alignment.md)
- the current `src/vitriol/kv/` implementation
- the active TurboQuant/QJL regression tests under `tests/`
