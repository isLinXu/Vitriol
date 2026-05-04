# TurboQuant Alignment Check for Qwen3.5-0.8B

This note checks whether Vitriol's current TurboQuant KV-cache behavior is aligned with the reference behavior described in the `mlx-vlm` TurboQuant KV Cache documentation and implementation approach.

## Scope

- Reference target:
  - `mlx-vlm` TurboQuant KV Cache behavior
  - `3.5-bit = 3-bit K + 4-bit V`
  - Quantize standard/full KV cache layers, keep already-memory-efficient cache types exact
  - Delayed quantization via `quantized_kv_start`
- Validation model:
  - `Qwen/Qwen3.5-0.8B`
- Runtime used here:
  - local ModelScope copy of `Qwen/Qwen3.5-0.8B`
  - device: `mps`
  - dtype: `torch.float16`

## What Matches the Reference

### 1. TurboQuant bit semantics match

Vitriol now interprets `3.5-bit` as:

- `K = 3-bit`
- `V = 4-bit`

This matches the reference TurboQuant interpretation used by `mlx-vlm`.

### 2. Layer targeting matches the intended strategy

On `Qwen3.5-0.8B`, the measured policy layout shows:

- `linear_attention`: kept exact
- `full_attention`: TurboQuant applied

Observed counts for `balanced(quantized_kv_start=0)`:

- `full_attention = 6`
- `linear_attention = 18`
- `turbo_k = 6`
- `turbo_v = 6`

This is consistent with the `mlx-vlm` design principle of applying TurboQuant only to standard KV cache layers and leaving already efficient cache types exact.

### 3. Long-context behavior is directionally aligned in correctness, but not yet in speed on MPS

Measured `safe` vs `balanced(quantized_kv_start=0)`:

- `128` tokens:
  - `delta_speedup = -0.053x`
  - `exact = True`
- `4096` tokens:
  - `safe_speedup = 1.058x`
  - `balanced_speedup = 0.541x`
  - `delta_speedup = -0.517x`
  - `exact = True`
- `8192` tokens:
  - `safe_speedup = 0.790x`
  - `balanced_speedup = 0.580x`
  - `delta_speedup = -0.210x`
  - `exact = True`

Interpretation:

- TurboQuant is not attractive for very short contexts
- TurboQuant remains accuracy-stable at long context
- On the current `Qwen3.5-0.8B + MPS` setup, long-context TurboQuant still does not beat `safe` on throughput

This is still partially aligned with the reference positioning of TurboQuant as a long-context optimization, but on this machine the realized gain is currently KV-memory reduction rather than end-to-end speedup.

## What Is Not Fully Verified Yet

### 1. Memory-side validation is now stronger, but still not complete

The `mlx-vlm` documentation emphasizes:

- KV memory reduction
- peak memory reduction
- stronger wins at very long contexts

We now have stable memory-side measurements for both `4096` and `8192` tokens:

- `4096` tokens
  - `safe`
    - `estimated_kv_megabytes = 48.188`
    - `peak_device_megabytes = 6350.984`
    - `peak_minus_estimated_mb = 6302.796`
  - `balanced(quantized_kv_start=0)`
    - `estimated_kv_megabytes = 33.599`
    - `peak_device_megabytes = 5360.000`
    - `peak_minus_estimated_mb = 5326.401`
- `8192` tokens
  - `safe`
    - `estimated_kv_megabytes = 96.188`
    - `peak_device_megabytes = 11475.000`
    - `peak_minus_estimated_mb = 11378.812`
  - `balanced(quantized_kv_start=0)`
    - `estimated_kv_megabytes = 67.068`
    - `peak_device_megabytes = 12260.000`
    - `peak_minus_estimated_mb = 12192.932`

Interpretation:

- Estimated KV memory dropped by about `30%` at both `4096` and `8192`
- At `4096`, device peak memory also dropped on this `MPS` run
- At `8192`, device peak memory increased despite the lower KV estimate

This is directionally consistent with TurboQuant's intended KV-cache savings, but still not a full match to the reference memory claims, because:

- the current device-level peak is still dominated by more than KV cache alone
- `peak_minus_estimated_mb` remains very large in both modes
- the sign of peak-memory improvement is not yet stable across long-context points
- we have not yet validated ultra-long-context memory behavior

### 1.5 Offline residual-signal validation is now available

Vitriol now has an offline `kv-analyze` path that measures quantization fidelity on a single prefill cache without waiting for a full decode benchmark.

Measured `balanced` vs `fast-balanced` with:

- model: local copy of `Qwen/Qwen3.5-0.8B`
- `quantized_kv_start = 0`
- `balanced`: residual sketch enabled
- `fast-balanced`: residual sketch disabled

At `128` prompt tokens:

- `balanced`
  - `avg_key_mse = 0.043164`
  - `avg_logits_mse = 0.487826`
  - `avg_output_mse = 0.000024`
  - `avg_residual_gain_k = 0.480090`
- `fast-balanced`
  - `avg_key_mse = 0.083032`
  - `avg_logits_mse = 1.047644`
  - `avg_output_mse = 0.000036`

At `512` prompt tokens:

- `balanced`
  - `avg_key_mse = 0.043782`
  - `avg_logits_mse = 0.496630`
  - `avg_output_mse = 0.000008`
  - `avg_residual_gain_k = 0.479975`
- `fast-balanced`
  - `avg_key_mse = 0.084190`
  - `avg_logits_mse = 1.058232`
  - `avg_output_mse = 0.000017`

Interpretation:

- The residual sketch consistently reduces `key_mse`
- The residual sketch also reduces proxy attention-logits drift by roughly half
- The output drift is already small in both cases, but `balanced` is still better

This does not prove full paper alignment, but it does show that Vitriol's residual stage is not dead weight: on real Qwen3.5-0.8B KV tensors, it improves fidelity in a stable direction.

### 2. We have not validated the ultra-long context regime highlighted by the reference

The reference material highlights very long context behavior. Our successful measurements currently cover:

- `128`
- `4096`
- `8192`

Attempting `16384` on the current Mac/MPS runtime failed with:

- `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX`

This looks like an Apple MPS backend limitation, not a TurboQuant policy bug. Still, it means we cannot yet claim alignment with the reference repo's ultra-long-context results.

## Where Vitriol Currently Goes Beyond the Reference

Vitriol adds extra strategy layers beyond a pure TurboQuant comparison:

- `aggressive`
- `ultra-long`
- `Sparse-V`
- `Compute-Skip`

These are useful, but they are not part of the narrowest `mlx-vlm` TurboQuant reference check. For strict alignment, the cleanest comparison is:

- `safe`
- `balanced + quantized_kv_start=0`

## Important Finding: More Aggressive Is Not Better on Qwen3.5-0.8B

Measured `8192` token comparison:

- `balanced(quantized_kv_start=0)` vs `aggressive(quantized_kv_start=0)`

Result:

- `base_exact = True`
- `compare_exact = True`
- `base_speedup = 1.1136x`
- `compare_speedup = 0.9738x`
- `delta_speedup = -0.1398x`

Policy counts:

- `balanced`
  - `turbo_k = 6`
  - `turbo_v = 2`
  - `sparse_v = 0`
- `aggressive`
  - `turbo_k = 6`
  - `turbo_v = 6`
  - `sparse_v = 4`

Interpretation:

- Moderate TurboQuant helps
- More `V` quantization plus `Sparse-V` does not help this model/runtime combination
- On `Qwen3.5-0.8B + MPS`, `balanced` is a better default than `aggressive`

## Current Verdict

Vitriol is currently:

- aligned in TurboQuant semantics
- aligned in layer-selection intent
- aligned in long-context correctness and KV-memory reduction trend
- supported by real offline evidence that the residual sketch improves KV fidelity

Vitriol is not yet fully proven aligned in:

- ultra-long-context memory behavior
- device-level peak-memory reduction under the reference-style long-context regime
- end-to-end speedup on the current `Qwen3.5-0.8B + MPS` runtime
- direct apples-to-apples comparison with the reference repo's reported long-context regime

So the most accurate statement is:

> Vitriol's TurboQuant behavior is methodologically aligned with `mlx-vlm`, the real `Qwen3.5-0.8B` path stays correct at `4096` and `8192` tokens, and KV-memory estimates improve by about `30%`, but full effect alignment is still not proven because ultra-long-context behavior is missing and the current `MPS` runtime does not yet show consistent peak-memory or throughput wins.

## Practical Recommendation

For `Qwen3.5-0.8B` on the current machine:

- Use `balanced` as the default TurboQuant preset
- Use smaller or zero `quantized_kv_start` only for long-context benchmarking
- Do not force TurboQuant for short prompts
- Do not prefer `aggressive` by default on MPS
- Treat `estimated_kv_mb` as KV-only and `peak_device_mb` / `peak_minus_estimated_mb` as the real device-memory context when judging runtime benefit

Recommended starting command:

```bash
PYTHONPATH=src python3 -m vitriol.cli.main bench kv-long \
  <local-model-path-or-model-id> \
  --preset balanced \
  --preset-param quantized_kv_start=0 \
  --prompt-tokens 4096 \
  --max-new-tokens 16 \
  --calib-new-tokens 8 \
  --search-max-n 2 \
  --format summary
```

## Next Steps for a Stricter Alignment Check

1. Run the same strict pair only:
   - `safe`
   - `balanced + quantized_kv_start=0`

2. Repeat on a CUDA machine:
   - to avoid the `MPS` `INT_MAX` limit at `16384+`
   - to check whether the missing throughput win is device-specific

3. Break down non-KV peak memory:
   - activations
   - temporary buffers
   - allocator/cache effects

4. Only after that, evaluate:
   - `aggressive`
   - `ultra-long`
   - `Sparse-V`
   - `Compute-Skip`
