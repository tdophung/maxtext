# All-Gather Breakdown: TE vs MT (Mixtral-8x7b, 1 Layer, Ring-of-Experts)

## Profile Configuration

| Parameter | Value |
|-----------|-------|
| Model | Mixtral-8x7b |
| Decoder layers | 1 (`base_num_decoder_layers=1`) |
| Hidden dim | 4096 |
| Attention heads | 32 (QKV), 8 (KV) |
| Head dim | 128 |
| MLP dim | 14336 |
| Experts | 8, top-k=2 |
| Sharding | EP=2, FSDP=2 (4 GPUs) |
| Dtype | bf16 |
| TE profile | `xprof_profiles/te_impl_ring/` (2026-03-25) |
| MT profile | `xprof_profiles/mt_impl_ring/` (2026-03-24) |

## Summary

Both TE and MT have **19 all-gather operations per GPU per step** (10 forward + 9 backward).
The data transfer volumes are identical — all gathered tensors are model weights or
activations whose shapes are determined by the architecture, not the TE/MT implementation choice.

| Metric | TE | MT |
|--------|------|------|
| Forward all-gather count | 10 | 10 |
| Backward all-gather count | 9 | 9 |
| Forward all-gather time (GPU pid=1) | 3.55 ms | 3.88 ms |
| Backward all-gather time (GPU pid=1) | 3.14 ms | 4.26 ms |
| **Total all-gather time** | **6.70 ms** | **8.14 ms** |
| Total data transferred per step | ~1806 MB | ~1806 MB |

Time differences are due to compute-communication overlap, not data volume.

## Forward Pass All-Gathers (10)

| # | HLO Op | Source | Description | Input Shape(s) | Transfer (MB) | TE Time (us) | MT Time (us) |
|---|--------|--------|-------------|----------------|---------------|--------------|--------------|
| 1 | `all-gather-start.1` | linears.py:99 | Attention QKV projection weight | bf16[1024, 32, 128] | 24.00 | 91 | 84 |
| 2 | `all-gather-start` | embeddings.py:179 | Token embedder weight | bf16[32000, 1024] | 187.50 | 398 | 391 |
| 3 | `all-gather-start.17` | linears.py:99 | Attention KQV bundled (K weight + V weight + KV bias) | bf16[1024,8,128] + bf16[1024,8,128] + bf16[1024,8] | 12.05 | 69 | 75 |
| 4 | `all-gather-start.2` | linears.py:99 | Attention output projection weight | bf16[32, 128, 1024] | 24.00 | 317 | 85 |
| 5 | `all-gather-start.7` | moe.py:1631 | MoE gate logits (ring-of-experts EP gather) | bf16[2, 4096, 8] | 0.12 | 192 | 212 |
| 6 | `all-gather-start.12` | moe.py:2564 | MoE w1 expert weight (FSDP gather) | bf16[4, 2048, 14336] | 224.00 | 595 | 556 |
| 7 | `all-gather-start.6` | moe.py:1631 | MoE input activations x (ring-of-experts EP gather) | bf16[2, 4096, 4096] | 64.00 | 250 | 12 |
| 8 | `all-gather-start.11` | moe.py:2563 | MoE w0 expert weight (FSDP gather) | bf16[4, 2048, 14336] | 224.00 | 605 | 653 |
| 9 | `all-gather-start.13` | moe.py:2565 | MoE wo expert weight (FSDP gather) | bf16[4, 14336, 2048] | 224.00 | 653 | 646 |
| 10 | `all-gather-start.3` | linears.py:99 | Output head (logits_dense) weight | bf16[1024, 32000] | 187.50 | 380 | 1170 |

### Notes on Forward All-Gathers

- **Rows 1-4, 10**: Attention and embedding weights — identical between TE/MT, determined by model architecture.
- **Row 5** (`.7`): Gate logits all-gather for ring-of-experts. Tiny transfer (0.12 MB). Both TE and MT gather the same logits tensor; TE uses raw logits while MT uses post-score-func logits, but the shape is the same.
- **Row 7** (`.6`): Input activations for ring-of-experts. Both gather the same `x` tensor from the attention layer.
- **Rows 6, 8, 9** (`.12`, `.11`, `.13`): MoE expert weights gathered along the FSDP axis. These are the largest transfers (224 MB each).
- **Row 10** (`.3`): Output head weight — 790 us faster in TE, likely due to better overlap with prior MoE compute.

## Backward Pass All-Gathers (9)

| # | HLO Op | Source | Description | Input Shape(s) | Transfer (MB) | TE Time (us) | MT Time (us) |
|---|--------|--------|-------------|----------------|---------------|--------------|--------------|
| 11 | `all-gather-start.18` | decoders.py:1038 | Remat: attention KQV bundled weights | bf16[1024,8] + bf16[1024,8,128] + bf16[1024,8,128] | 12.05 | 88 | 108 |
| 12 | `all-gather-start.8` | moe.py:1631 | Remat: MoE gate logits (ring EP gather) | bf16[2, 4096, 8] | 0.12 | 8 | 20 |
| 13 | `all-gather-start.9` | moe.py:1631 | Remat: MoE input activations x (ring EP gather) | bf16[2, 4096, 4096] | 64.00 | 173 | 199 |
| 14 | `all-gather-start.10` | moe.py:1863 | Backward of psum_scatter (MoE output gradient) | bf16[2, 4096, 4096] | 64.00 | 209 | 199 |
| 15 | `all-gather-start.14` | decoders.py:1038 | Remat: MoE wo expert weight | bf16[4, 14336, 2048] | 224.00 | 708 | 663 |
| 16 | `all-gather-start.16` | decoders.py:1038 | Remat: MoE w0 expert weight | bf16[4, 2048, 14336] | 224.00 | 726 | 913 |
| 17 | `all-gather-start.15` | decoders.py:1038 | Remat: MoE w1 expert weight | bf16[4, 2048, 14336] | 224.00 | 1037 | 754 |
| 18 | `all-gather-start.4` | decoders.py:1038 | Backward: attention output projection weight | bf16[32, 128, 1024] | 24.00 | 98 | 1219 |
| 19 | `all-gather-start.5` | decoders.py:1038 | Backward: attention QKV projection weight | bf16[1024, 32, 128] | 24.00 | 97 | 185 |

### Notes on Backward All-Gathers

- **Rows 11-13, 15-17**: Rematerialization (remat) all-gathers. JAX's checkpointing recomputes the forward pass during the backward pass, requiring re-gathering of weights and activations.
- **Row 12** (`.8`): Remat of gate logits — negligible transfer (0.12 MB).
- **Row 14** (`.10`): The backward (gradient) of the `psum_scatter` at moe.py:1863, which is the reduce-scatter of MoE output in the ring-of-experts path.
- **Row 18** (`.4`): Attention output weight for backward — 1121 us faster in TE, largest single timing difference.

## All-Gather Categories

### By Communication Group

| Category | Replica Groups | Ops | Description |
|----------|---------------|-----|-------------|
| FSDP weight gather | `[1,4]<=[4]` (all 4 GPUs) | `.1`, `(unnumbered)`, `.17`, `.2`, `.3`, `.18`, `.4`, `.5` | Weights sharded across FSDP, gathered to full size |
| Expert weight gather | `[2,2]<=[2,2]T(1,0)` (FSDP within EP group) | `.11`, `.12`, `.13`, `.14`, `.15`, `.16` | MoE expert weights gathered along FSDP axis |
| Ring-of-experts EP gather | `{{0,1},{2,3}}` (EP groups of 2) | `.6`, `.7`, `.8`, `.9`, `.10` | Activations/logits duplicated across EP shards |

### By Data Volume

| Category | Per-op Transfer | Count | Total |
|----------|----------------|-------|-------|
| MoE expert weights (w0, w1, wo) | 224 MB each | 6 (3 fwd + 3 bwd) | 1344 MB |
| Embeddings + output head | 187.5 MB each | 2 | 375 MB |
| Ring-of-experts activations (x) | 64 MB each | 3 (1 fwd + 1 remat + 1 bwd psum_scatter) | 192 MB |  
| Attention weights (QKV, KQV, output) | 8-24 MB each | 6 (3 fwd-equivalent + 3 bwd-equivalent) | ~108 MB |
| Ring-of-experts gate logits | 0.12 MB each | 2 (1 fwd + 1 remat) | 0.25 MB |

## What About `pre_bias_logits`?

For **Mixtral-8x7b**, `pre_bias_logits` is `None` (only used by DeepSeek v3), so no all-gather
occurs for it in either path.

For **DeepSeek v3**, the TE path would save one additional all-gather of shape
`[ep_shards, batch*seq, num_experts]` because TE's fused router computes pre-bias logits
internally within its kernel, never materializing them as a separate JAX tensor. The MT path
requires a separate all-gather because `deepseek_routing()` needs `pre_bias_logits` as a
distinct input to compute routing weights (see `jnp.take_along_axis(pre_bias_logits, ...)`
at moe.py:644).

## Source Data

- TE HLO: `xprof_profiles/te_impl_ring/hlo_dump/module_5301.jit_train_step.sm_10.0a_gpu_after_optimizations.txt`
- MT HLO: Not available as text dump. Shapes inferred from identical model config. Binary proto at `xprof_profiles/mt_impl_ring/tensorboard/plugins/profile/2026_03_24_17_21_52/jit_train_step(5301).hlo_proto.pb`.
- Timing data: Extracted from `trace.json.gz` in each tensorboard profile directory, GPU pid=1.
