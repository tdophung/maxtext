# Plan: Fuse 3-Pass `row_id_map` Construction into a Single Kernel

## Background

TransformerEngine's MoE permutation builds a `row_id_map` tensor that tells the
permute/unpermute Triton kernels where each token should be scattered to (or
gathered from) in the expert-sorted buffer. Today this is done in **3 separate
Triton kernels + 1 JAX scatter**, each requiring a host-side dispatch (~150 µs
each). Fusing them into a **single kernel** (keeping Pass 3 separate) saves 2
dispatches per MoE layer.

## Current 3-Pass Architecture

**Files:**
- Triton kernels: `TransformerEngine/transformer_engine/common/triton/permutation.py`
- JAX primitives + orchestrator: `TransformerEngine/transformer_engine/jax/triton_extensions/permutation.py`
  - Orchestrator function: `make_row_id_map()` (line ~1846)

**Data shapes:**
- `routing_map`: `[num_tokens, num_experts]` — binary mask (1 = token routed to expert)
- `row_id_map`: `[num_tokens, num_experts * 2 + 1]` — output lookup table
- `workspace`: `[num_experts, cdiv(num_tokens, BLOCK_SIZE)]` — inter-block communication

### Pass 1: Block-Local Cumsum
- **Grid**: `(num_experts, cdiv(num_tokens, BLOCK_SIZE))`
- **What it does**: For each (expert, token_block) pair, computes a block-local
  cumulative sum of `routing_map`. Stores per-block token counts to `workspace`.
- **Output**: `row_id_map[:, 0:num_experts]` has block-local positions;
  `workspace` has per-block counts.

### Pass 2: Global Prefix Sum
- **Grid**: `(num_experts, cdiv(num_tokens, BLOCK_SIZE))`
- **What it does**: For each (expert, token_block), loads ALL prior blocks'
  counts from `workspace`, computes their sum, and adds it to the block-local
  positions from Pass 1. Marks unrouted tokens as -1.
- **Output**: `row_id_map[:, 0:num_experts]` now has global destination rows.

### JAX Scatter
- `row_id_map = row_id_map_pass2.at[:, num_experts:].set(-1)`
- Initializes columns `[num_experts:]` to -1 for Pass 3.

### Pass 3: Per-Token Argsort
- **Grid**: `(num_tokens,)`
- **What it does**: For each token, loads its destination rows for all experts,
  counts valid entries (`n_routed`), runs a bitonic argsort, and packs:
  - Columns `[0, n_routed)`: sorted destination row indices
  - Columns `[num_experts, num_experts + n_routed)`: corresponding expert IDs
  - Column `[num_experts * 2]`: `n_routed` count

### Why 3 Passes?

Passes 1 and 2 are separated because Pass 1 produces per-block partial sums
that need global aggregation before Pass 2 can compute final positions. Triton
cannot synchronize across thread blocks within a single kernel launch.

## Proposed Fusion: Passes 1 + 2 + Scatter → Single Kernel

### Key Insight

Assign **one thread block per expert** and have it **loop** over all token
blocks sequentially. Since one program handles all blocks for its expert, the
running prefix sum is a local variable — no inter-block synchronization needed.

### Fused Kernel Algorithm

```python
@triton.jit
def _row_id_map_fused_pass12_kernel(
    routing_map_ptr,
    row_id_map_buf_ptr,    # pre-allocated buffer filled with -1 (input_output_alias)
    num_tokens,
    stride_routing_map_token,
    stride_routing_map_expert,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Pass 1 + Pass 2: compute global destination rows for one expert."""
    expert_id = tl.program_id(0)
    running_count = 0

    for block_start in range(0, num_tokens, BLOCK_SIZE):
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < num_tokens

        # Load routing decisions for this expert
        routed = tl.load(
            routing_map_ptr
            + expert_id * stride_routing_map_expert
            + offset * stride_routing_map_token,
            mask=mask,
            other=0,
        ).to(tl.int32)

        # Block-local cumsum (same as Pass 1)
        local_cumsum = tl.cumsum(routed) * routed

        # Convert to global position (replaces Pass 2)
        global_pos = tl.where(
            local_cumsum != 0,
            running_count + local_cumsum - 1,  # 0-indexed
            -1,                                 # unrouted
        )

        # Write destination row for this expert column
        tl.store(
            row_id_map_buf_ptr
            + expert_id * stride_row_id_map_expert
            + offset * stride_row_id_map_token,
            global_pos,
            mask=mask,
        )

        running_count += tl.sum(routed)
```

- **Grid**: `(num_experts,)` — one program per expert
- **No workspace tensor** — `running_count` replaces inter-block communication
- **No JAX scatter** — pre-allocate `row_id_map` filled with -1 via
  `input_output_aliases` to a `jnp.full((num_tokens, num_experts*2+1), -1, jnp.int32)`

### Updated `make_row_id_map()`

```python
def make_row_id_map(routing_map, num_tokens, num_experts):
    # Pre-allocate with -1 (columns [num_experts:] stay -1 for Pass 3)
    row_id_map_buf = jnp.full((num_tokens, num_experts * 2 + 1), -1, dtype=jnp.int32)

    # Fused Pass 1+2: compute global destination rows (writes to columns [0:num_experts])
    row_id_map = RowIdMapFusedPrimitive.outer_primitive.bind(
        routing_map,
        row_id_map_buf,    # aliased as output via input_output_aliases={1: 0}
        num_tokens=num_tokens,
        num_experts=num_experts,
        block_size=DEFAULT_BLOCK_SIZE,
    )

    # Pass 3: per-token argsort (unchanged)
    row_id_map = RowIdMapPass3Primitive.outer_primitive.bind(
        row_id_map,
        num_tokens=num_tokens,
        num_experts=num_experts,
    )

    return row_id_map
```

### What Changes

| Current (4 dispatches) | Proposed (2 dispatches) |
|------------------------|------------------------|
| Pass 1 Triton kernel   | Fused Pass 1+2 kernel  |
| Pass 2 Triton kernel   | *(eliminated)*         |
| JAX scatter `.at[].set(-1)` | *(eliminated — pre-initialized buffer)* |
| Pass 3 Triton kernel   | Pass 3 (unchanged)     |

### Estimated Savings

- **2 fewer Triton dispatches + 1 fewer JAX op** per MoE layer
- At ~150 µs host overhead per dispatch: ~450 µs saved per MoE layer
- With 4 decoder layers (fwd only, bwd reuses saved `row_id_map`): **~1.8 ms/step**

### Performance Considerations

1. **GPU occupancy**: The fused kernel launches `num_experts` programs (e.g., 8)
   vs. the current `num_experts × num_blocks` programs (e.g., 64). With only 8
   programs, SM utilization is low, but the kernel is memory-bound — 8 programs
   are sufficient to saturate memory bandwidth for this small workload.

2. **Large token counts**: For very large `num_tokens` (100K+), the sequential
   loop over blocks in a single program may become a bottleneck. A fallback to
   the 3-pass approach (or a hybrid with more programs per expert using atomics)
   could be added behind a threshold.

3. **Sharding**: The `partition()` method should shard on the token dimension
   (same as current Pass 1). Each shard gets its local `num_tokens` and the
   kernel processes them independently.

## Implementation Checklist

1. [ ] Write `_row_id_map_fused_pass12_kernel` in
       `transformer_engine/common/triton/permutation.py`
2. [ ] Create `RowIdMapFusedPrimitive` in
       `transformer_engine/jax/triton_extensions/permutation.py`
       (abstract, impl, lowering, infer_sharding, partition — follow the
       pattern of `RowIdMapPass1Primitive` but with `input_output_aliases={1: 0}`)
3. [ ] Update `make_row_id_map()` to use the fused primitive + Pass 3
4. [ ] Add unit test: verify `row_id_map` output matches the 3-pass version
       for various `(num_tokens, num_experts, topk)` configurations
5. [ ] Benchmark: compare host-side step time before/after fusion

## Files to Modify

- `transformer_engine/common/triton/permutation.py` — add fused kernel
- `transformer_engine/jax/triton_extensions/permutation.py` — add JAX primitive, update `make_row_id_map()`
- Test file (new or existing) — correctness test
