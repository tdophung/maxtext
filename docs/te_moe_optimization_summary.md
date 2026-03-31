# TE MoE Ring-of-Experts Optimization Summary

## Goal

Make TransformerEngine's (TE) MoE implementation faster than MaxText's (MT)
native implementation in the ring-of-experts configuration.

## Test Setup

- 1 node, 4 GB100 GPUs
- 4 decoder layers, ring-of-experts sharding
- Profiling script: `profile_moe_comparison.sh`
- Training script: `test_permrouter.sh` (toggling `te_permutation_and_router_impl`)

---

## Completed Optimizations

### 1. Remove Redundant `all_gather` of `pre_bias_logits`

**Problem**: In ring-of-experts mode with expert parallelism, MaxText
all-gathered `pre_bias_logits` across expert shards. This tensor is only used
by MT's DeepSeek routing path — TE never reads it. The wasted collective added
latency.

**Fix**: Conditionally skip the all-gather when the TE path is active.

**File**: `src/maxtext/layers/moe.py` (line ~1628)

```python
# TE path does not use pre_bias_logits, so skip its allgather
# to avoid a wasted collective.
x, logits = tuple(
    jax.lax.all_gather(z, axis_name=..., tiled=True)
    for z in (x, logits)
)
if not use_te:
    pre_bias_logits = jax.lax.all_gather(pre_bias_logits, ...)
```

**Savings**: ~1 NCCL all-gather per step.

---

### 2. Mask `routing_map` to Local Experts Inside `_te_permute`

**Problem**: In ring-of-experts mode, each GPU only processes a subset of
experts. MT masks routing decisions to local experts before its permutation.
TE's path was receiving the full (global) `routing_map`, causing it to permute
tokens for non-local experts — wasting work in permutation, GEMMs, and
unpermutation.

**Fix**: Apply `local_expert_mask` to `routing_map` and `sparse_probs` inside
`_te_permute` before calling `te_token_dispatch`.

**File**: `src/maxtext/layers/moe.py` (line ~818)

```python
if num_experts_per_shard is not None:
    local_expert_mask = (jnp.arange(self.num_experts) < num_experts_per_shard)
    routing_map = routing_map * local_expert_mask[None, :]
    sparse_probs = sparse_probs * local_expert_mask[None, :].astype(sparse_probs.dtype)
```

**Impact**: Eliminates downstream masking ops that XLA previously inserted as
separate fusions. Also means TE's `row_id_map` only references valid (local)
positions, so the pre-unpermute zeroing mask can be skipped for TE.

---

### 3. Skip Pre-Unpermute Zeroing Mask for TE Path

**Problem**: Before unpermutation, MT zeros out positions beyond the actual
token count to prevent garbage from ragged_dot's unused rows from leaking
through. With optimization #2, TE's `row_id_map` already excludes non-local
experts, so this mask is unnecessary for the TE path — it was just wasted XLA
fusions.

**Fix**: Guard the mask behind `if not perm_state.use_te`.

**File**: `src/maxtext/layers/moe.py` (line ~1844)

```python
if self.config.use_ring_of_experts:
    # MT path needs zeroing; TE path's routing_map was already masked
    if not perm_state.use_te:
        mask = jnp.arange(intermediate_output.shape[0]) < jnp.sum(group_sizes)
        intermediate_output = jnp.where(mask[:, None], intermediate_output, 0)
```

**Savings**: Removes 1 XLA fusion (comparison + broadcast + select) per MoE
layer.

---

### 4. Disable Padding Alignment in Ring-of-Experts Mode

**Problem**: TE's `te_permutation_align_size` pads each expert's token count to
a multiple of `align_size`. In ring-of-experts mode, where tokens are already
masked to local experts, this padding inflates the buffer through all 3 GEMMs
and their transposes. XLA picks slower GEMM tilings for the padded shapes.

**Fix**: Set `align_size = None` when `roll_to_expert_id is not None`
(ring-of-experts mode).

**File**: `src/maxtext/layers/moe.py` (line ~826)

```python
# In ring-of-experts mode, disable padding alignment
if roll_to_expert_id is not None:
    align_size = None
else:
    align_size = self.config.te_permutation_align_size
    if align_size == 0:
        align_size = None
```

---

### Net Result of Completed Optimizations

After optimizations 1–4, GPU profiling shows TE's MoE region is **~798 µs
faster** than MT's in kernel execution time. However, TE is still **~6 ms
slower** in wall-clock step time due to host-side dispatch overhead from ~40
Triton custom calls per step (see planned optimizations below).

---

## Planned Optimizations

### 5. CUDA Graph Capture for Triton Custom Calls (Command Buffers)

**Problem**: TE's MoE permutation uses ~40 Triton `custom_call` invocations per
training step (across 4 decoder layers, fwd + bwd). Each call is dispatched
individually from the host because `triton_kernel_call` is registered as a
legacy custom call (not FFI), and XLA's command buffer conversion pass only
captures custom calls with an FFI handler marked `kCmdBufferCompatible`. The
host dispatch overhead (~150 µs/call) accounts for ~6 ms/step — negating the
GPU kernel speedup.

**Approach**: Register an FFI shim handler for `triton_kernel_call` with the
`kCmdBufferCompatible` trait so XLA's `CommandBufferConversionPass` includes
Triton calls in CUDA graph capture.

**Current status**: A pure-Python patch (`src/maxtext/triton_cmd_buffer_patch.py`)
has been written that uses `ctypes` to construct a shim FFI handler implementing
XLA's metadata query protocol and registers it via
`jaxlib._jax.register_custom_call_target`. This requires zero C++ recompilation.
Testing is in progress via `profile_moe_comparison.sh --cmd-buffer-patch`.

**Estimated savings**: ~6 ms/step (eliminates host dispatch gaps between Triton
kernel launches).

**Detailed plan**: `docs/triton_kernel_call_ffi_migration_plan.md`

**Risk**: Triton autotuning during warmup involves trial kernel launches that
are not CUDA-graph-safe. Post-warmup, `TritonKernelCall` only calls
`cuLaunchKernel` which is graph-safe. Command buffers are built from the
compiled HLO which reflects post-autotuning state, so this should be fine in
practice.

---

### 6. Fuse 3-Pass `row_id_map` Construction into a Single Kernel

**Problem**: TE's permutation builds a `row_id_map` tensor in 3 separate Triton
kernels + 1 JAX scatter:

| Pass | Grid | What it does |
|------|------|--------------|
| Pass 1 | `(num_experts, num_blocks)` | Block-local cumsum of `routing_map` |
| Pass 2 | `(num_experts, num_blocks)` | Global prefix sum across blocks |
| JAX scatter | — | Initialize columns `[num_experts:]` to -1 |
| Pass 3 | `(num_tokens,)` | Per-token argsort |

Passes 1 and 2 are separate because Triton cannot synchronize across thread
blocks within a single kernel launch. Each dispatch incurs ~150 µs host
overhead.

**Approach**: Fuse Passes 1 + 2 + scatter into a single kernel by assigning one
thread block per expert and looping over token blocks sequentially. The running
prefix sum becomes a local variable — no inter-block sync needed. The `-1`
initialization is handled by pre-allocating the output buffer with
`jnp.full(..., -1)` and using `input_output_aliases`.

**Result**: 4 dispatches → 2 dispatches per MoE layer.

**Estimated savings**: ~450 µs per MoE layer × 4 layers = ~1.8 ms/step.

**Detailed plan**: `docs/row_id_map_kernel_fusion_plan.md`

**Files to modify**:
- `transformer_engine/common/triton/permutation.py` — add fused kernel
- `transformer_engine/jax/triton_extensions/permutation.py` — add JAX primitive,
  update `make_row_id_map()`

---

### 7. Integrate TE Grouped GEMM

**Problem**: Currently the MoE path uses `jax.lax.ragged_dot` for the grouped
GEMM operations (w0, w1, wo). XLA lowers `ragged_dot_general` into a single
large `__cublas$gemm` by constructing a block-diagonal LHS — this involves a
Triton fusion (`loop_transpose_fusion`) that broadcasts the permuted tokens
across the expert dimension, masks each expert's valid range, and transposes
into `[num_tokens, num_experts*hidden]`. This input-construction overhead
exists regardless of whether TE or MT permutation is used, as it is inherent
to how XLA implements `ragged_dot_general`.

**Note on layouts**: HLO analysis confirms there is **no layout conversion**
(copy changing `{1,0}` to `{0,1}`) between TE permute output and `ragged_dot`.
Both use row-major layouts throughout. The `loop_transpose_fusion` between them
is `ragged_dot`'s own input setup, not a TE-specific overhead. (This corrects
an earlier hypothesis about layout mismatch at the TE/XLA boundary.)

**Approach**: Replace `ragged_dot` with TE's native grouped GEMM, which may
use a more efficient strategy than XLA's broadcast+mask+transpose approach for
constructing the GEMM input. An integration already exists on the upstream
branch `te/main` at `https://github.com/nvjax-svc-0/maxtext/commits/te/main/`.

**Status**: Waiting for the author to squash commits before rebasing onto the
current working branch to minimize merge conflicts.

**Expected benefit**: May eliminate the `ragged_dot` input-construction fusions
(`loop_transpose_fusion`) if TE's grouped GEMM handles expert dispatch
internally. May also unlock better kernel fusion within TE's GEMM
implementation.

---

## Performance Summary

| Metric | Before Optimizations | After 1–4 | After 5 (est.) | After 5+6 (est.) | After 5+6+7 (est.) |
|--------|---------------------|-----------|-----------------|-------------------|---------------------|
| GPU kernel time (TE vs MT) | TE slower | TE ~798 µs faster | same | ~2.6 ms faster | TBD |
| Wall-clock step time | TE ~6 ms slower | TE ~5 ms slower | TE ~1 ms faster | TE ~2.8 ms faster | TBD (no layout conversion savings; benefit depends on TE GEMM strategy) |
| Tokens/s/device | MT ~20,550 | MT ~20,550, TE ~20,300 | TE should exceed MT | — | — |

*Estimates assume 4 decoder layers, fwd-only `row_id_map` (bwd reuses saved map).*

---

## Files Modified

| File | Changes |
|------|---------|
| `src/maxtext/layers/moe.py` | Optimizations 1–4 |
| `src/maxtext/triton_cmd_buffer_patch.py` | New — FFI shim for CUDA graph capture (opt 5) |
| `profile_moe_comparison.sh` | Added `--cmd-buffer-patch` and `--dump-hlo` flags |
| `docs/row_id_map_kernel_fusion_plan.md` | Detailed plan for opt 6 |
| `docs/triton_kernel_call_ffi_migration_plan.md` | Detailed plan for opt 5 |

## Future Work (Blocked on Resources)

- **Larger model validation**: Current testing is on 4 layers / 1 node.
  Validation on 32+ layer models requires more GPU memory than 4×GB100.
- **Full FFI migration for `triton_kernel_call`**: The Python ctypes shim
  (opt 5) is a stopgap. A proper C++ FFI handler in `jaxlib` is the clean
  long-term solution, aligned with JAX's direction (issue #27988). See
  `docs/triton_kernel_call_ffi_migration_plan.md` Option A (FFI Shim) and
  Option C (Full Migration).
