# MaxText TE MoE grouped-GEMM vs CuTeDSL comparison

The canonical four-GPU production-slice launcher in this workspace is:

```bash
cd /mnt/tdophung/ptyche-lustre-home/maxtext
bash scripts/run_dsv3_prod_slice_ep2_fsdp2.sh
```

It launches one JAX process per GPU with EP=2 and FSDP=2. The same launcher
selects either the normal Transformer Engine MXFP8 grouped GEMM path or the
cuDNN frontend CuTeDSL fused grouped-GEMM+SwiGLU path.

## Build Transformer Engine

The installed Transformer Engine must come from the sibling checkout so it
contains `transformer_engine/jax/cutedsl_extensions/moe.py` and the matching
functional MoE integration:

```bash
cd /mnt/tdophung/ptyche-lustre-home/TransformerEngine
python3 -m pip install \
  pybind11 ninja cmake pytest \
  nvidia-cudnn-frontend==1.25.0 \
  'nvidia-cutlass-dsl[cu13]==4.5.0'
NVTE_CMAKE_BUILD_DIR=build_jax \
NVTE_FRAMEWORK=jax \
NVTE_CUDA_ARCHS=100a \
pip3 install --no-build-isolation -e .
```

This is the JAX/SM100a build flow defined by `TransformerEngine/bashrc`.
The CUTLASS pin is required by cuDNN frontend 1.25.0. Newer CUTLASS DSL
versions can load the forward kernel but fail NVVM compilation of the
dSwiGLU backward kernel on SM100a.

This checkout also needs its normal MaxText runtime dependencies installed.
Do not install the generated `cuda12` dependency set over a container's custom
CUDA 13/JAX stack. In the development container used for this comparison, the
missing runtime imports were installed with:

```bash
python3 -m pip install \
  pathwaysutils tensorflow omegaconf ml-collections grain array-record \
  tensorflow-datasets seqio jinja2 jaxtyping aqtp qwix tokamax \
  transformers tiktoken sympy drjax
```

## Matched loss runs

Run the TE grouped-GEMM baseline first:

```bash
cd /mnt/tdophung/ptyche-lustre-home/maxtext
TE_MOE_CUTEDSL_FUSION=0 \
bash scripts/run_dsv3_prod_slice_ep2_fsdp2.sh
```

Then run CuTeDSL with every model and data setting unchanged:

```bash
cd /mnt/tdophung/ptyche-lustre-home/maxtext
TE_MOE_CUTEDSL_FUSION=1 \
bash scripts/run_dsv3_prod_slice_ep2_fsdp2.sh
```

Useful overrides include `STEPS`, `BATCH_PER_GPU`, `MAX_TARGET_LENGTH`,
`OUTPUT_ROOT`, `COORDINATOR_PORT`, and `EXTRA_MAXTEXT_ARGS`. For example:

```bash
STEPS=5 BATCH_PER_GPU=1 TE_MOE_CUTEDSL_FUSION=1 \
bash scripts/run_dsv3_prod_slice_ep2_fsdp2.sh
```

Use `DRY_RUN=1` to validate and print the fully resolved command without
launching JAX.

Each run writes a distinct directory below `maxtext/outputs/`, labeled either
`te_grouped_gemm` or `cutedsl`. Important files are:

- `slice-summary.txt`: resolved shape, implementation, and command
- `logs/proc_<rank>.log`: stdout/stderr for each one-GPU process
- `loss-curve.txt`: `step loss` extracted from rank 0
- `summary.txt`: exit status and elapsed time
- `jax_cache/`: run-local JAX compilation cache

Compare the two `loss-curve.txt` files step for step. PASS requires both runs
to finish with finite loss and the CuTeDSL curve to equal or only minimally
deviate from the baseline curve. Keep random seed, synthetic dataset, model
shape, step count, and all optimizer settings identical.

## Production-slice shape

The run uses DeepSeek V3 with two decoder layers (one dense and one routed
MoE layer) and the following routed-MoE settings:

- production reference: FSDP=16, EP=2, experts=256
- local run: FSDP=2, EP=2, experts=32
- experts per GPU: 8 in both production and local configurations
- hidden dimension: 7168
- MoE intermediate dimension: 2048
- experts per token: 8
- sequence length: 4096 by default
- batch per GPU: 4 by default
- quantization: `te_mxfp8` for dense and grouped GEMMs

The hidden and intermediate dimensions are both divisible by the 128-element
MXFP8 K alignment. CuTeDSL raises TE EP dispatch-slot alignment from 128 to
256 tokens, as required by the kernel.

## Required MoE configuration

Both arms use the full Transformer Engine MoEBlock:

```text
te_moe_block=true
te_use_gmm=false
sharding_tolerance=1.0
te_router_and_permutation_impl=false
te_gmm_quantization=te_mxfp8
quantization=te_mxfp8
```

The only comparison variable is:

```text
te_moe_cutedsl_fusion=false  # normal TE grouped GEMM
te_moe_cutedsl_fusion=true   # CuTeDSL grouped GEMM + SwiGLU fusion
```

MaxText maps the latter to
`NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1`. Do not switch to MaxText's
separate `te_use_gmm=true` ragged/permutation path; it does not test the same
TE MoEBlock integration.

## API path

The call chain is:

1. `maxtext/src/maxtext/layers/moe.py::_te_moe_block`
2. `transformer_engine.jax.moe.moe`
3. TE EP prepare/dispatch
4. TE `_ffn_fwd_per_shard` and `_ffn_bwd_per_shard`
5. normal `tex.grouped_gemm`, or the wrappers in
   `transformer_engine/jax/cutedsl_extensions/moe.py`
6. TE EP combine and the custom VJP

The CuTeDSL forward wrapper fuses MXFP8 FC1 grouped GEMM, SwiGLU, and FC2
input quantization. The backward wrapper fuses the FC2 dgrad GEMM, dSwiGLU,
and FC1 dgrad quantization. Remaining grouped GEMMs use the ordinary TE path.

Before a full MaxText run, the focused kernel tests are:

```bash
cd /mnt/tdophung/ptyche-lustre-home/TransformerEngine
CUDA_VISIBLE_DEVICES=0 pytest -q tests/jax/test_cutedsl_moe.py
```

## Validated result (2026-07-21)

The focused Transformer Engine suite passed all five tests on NVIDIA GB200.
A matched 10-step MaxText comparison also passed with batch size 4 per GPU,
sequence length 4096, EP=2, FSDP=2, and 32 routed experts:

| Step | Baseline loss | CuTeDSL loss | Baseline tokens/s/device | CuTeDSL tokens/s/device | CuTeDSL change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12.261 | 12.261 | 927 | 1,368 | compile step |
| 1 | 12.261 | 12.261 | 30,741 | 30,674 | -0.22% |
| 2 | 11.985 | 11.985 | 76,651 | 77,914 | +1.65% |
| 3 | 11.743 | 11.743 | 77,468 | 79,552 | +2.69% |
| 4 | 11.525 | 11.525 | 78,439 | 79,314 | +1.12% |
| 5 | 11.328 | 11.328 | 77,365 | 77,731 | +0.47% |
| 6 | 11.152 | 11.152 | 77,799 | 79,394 | +2.05% |
| 7 | 11.114 | 11.114 | 73,123 | 79,683 | +8.97% |
| 8 | 11.096 | 11.096 | 74,909 | 78,458 | +4.74% |
| 9 | 11.080 | 11.080 | 76,844 | 79,410 | +3.34% |

Across steady-state steps 2-9, CuTeDSL averaged 78,932 tokens/s/device
versus 76,575 for the baseline, a 3.08% throughput improvement. Mean step
time decreased from 0.2141 seconds to 0.2076 seconds (3.04%), and the median
paired throughput improvement was 2.37%. The maximum absolute loss difference
inferred from the more precise perplexity values was 0.000336.
