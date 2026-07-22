#!/bin/bash
# Run DeepSeek V3 on one 4-GPU node with EP=2/FSDP=2, but scale the
# model so the routed-MoE per-GPU shard matches a 32-GPU production config.
#
# Production shape:
#   total FSDP = 16
#   total EP   = 2
#   shard_exp_on_fsdp = true
#
# Local shape:
#   total FSDP = 2
#   total EP   = 2
#
# Routed expert target:
#   experts/GPU      = 256 / (2 * 16) = 8
#   hidden_dim/GPU   = 7168            (FSDP shards the expert axis)
#   moe_mlp_dim/GPU  = 2048

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAXTEXT_DIR="${MAXTEXT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_DIR="$(cd "${MAXTEXT_DIR}/.." && pwd)"
TRANSFORMER_ENGINE_DIR="${TRANSFORMER_ENGINE_DIR:-${WORKSPACE_DIR}/TransformerEngine}"

export CC="${CC:-ccache gcc}"
export CXX="${CXX:-ccache g++}"
export CCACHE_DIR="${CCACHE_DIR:-/mnt/ccache}"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda}"

export NVTE_JAX_ENFORCE_V2_GROUPED_GEMM="${NVTE_JAX_ENFORCE_V2_GROUPED_GEMM:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-16}"
export XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE="${XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE:-false}"
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB="${XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB:-180}"
export DECOUPLE_GCLOUD="${DECOUPLE_GCLOUD:-TRUE}"
export DECOUPLE_GLOUD="${DECOUPLE_GLOUD:-${DECOUPLE_GCLOUD}}"

# Import this MaxText checkout. An optional site_overrides directory can still
# be supplied by callers that need the Triton pre-import workaround.
export PYTHONPATH="${MAXTEXT_DIR}/src${SITE_OVERRIDES_DIR:+:${SITE_OVERRIDES_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

NCCL_EP_LIB_DIR="${TRANSFORMER_ENGINE_DIR}/3rdparty/nccl/build/lib"
if [[ -f "${NCCL_EP_LIB_DIR}/libnccl_ep.so.0" ]]; then
  export LD_LIBRARY_PATH="${NCCL_EP_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${MAXTEXT_DIR}/outputs}"

MODEL="${MODEL:-deepseek3-671b}"
EP="${EP:-2}"
FSDP="${FSDP:-2}"
STEPS="${STEPS:-20}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-4096}"
NUM_DECODER_LAYERS="${NUM_DECODER_LAYERS:-2}"
FIRST_NUM_DENSE_LAYERS="${FIRST_NUM_DENSE_LAYERS:-1}"
NUM_EXPERTS_PER_TOK="${NUM_EXPERTS_PER_TOK:-8}"
ATTENTION="${ATTENTION:-cudnn_flash_jax}"
REMAT_POLICY="${REMAT_POLICY:-custom}"
QUANTIZATION="${QUANTIZATION:-te_mxfp8}"
TE_GMM_QUANTIZATION="${TE_GMM_QUANTIZATION:-te_mxfp8}"
MEM_FRACTION="${XLA_MEM_FRAC:-0.92}"
DRY_RUN="${DRY_RUN:-0}"
COORDINATOR_PORT="${COORDINATOR_PORT:-21555}"
TE_MOE_CUTEDSL_FUSION="${TE_MOE_CUTEDSL_FUSION:-0}"

if [[ "${TE_MOE_CUTEDSL_FUSION}" != "0" && "${TE_MOE_CUTEDSL_FUSION}" != "1" ]]; then
  echo "TE_MOE_CUTEDSL_FUSION must be 0 (TE grouped GEMM baseline) or 1 (CuTeDSL), got ${TE_MOE_CUTEDSL_FUSION}."
  exit 1
fi
if [[ "${TE_MOE_CUTEDSL_FUSION}" == "1" ]]; then
  python3 - <<'PY'
import importlib.metadata

required = {
    "nvidia-cudnn-frontend": "1.25.0",
    "nvidia-cutlass-dsl": "4.5.0",
    "nvidia-cutlass-dsl-libs-cu13": "4.5.0",
}
errors = []
for package, expected in required.items():
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"{package} is not installed (required {expected})")
    else:
        if actual != expected:
            errors.append(f"{package}=={actual} is installed (required {expected})")
if errors:
    raise SystemExit("Invalid CuTeDSL environment:\n- " + "\n- ".join(errors))
PY
fi
export NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION="${TE_MOE_CUTEDSL_FUSION}"
CUTEDSL_CONFIG="false"
CUTEDSL_LABEL="te_grouped_gemm"
if [[ "${TE_MOE_CUTEDSL_FUSION}" == "1" ]]; then
  CUTEDSL_CONFIG="true"
  CUTEDSL_LABEL="cutedsl"
fi

PROD_FSDP=16
PROD_EP=2
PROD_NUM_EXPERTS=256
PROD_BASE_EMB_DIM=7168
PROD_BASE_MLP_DIM=18432
PROD_BASE_MOE_MLP_DIM=2048

TARGET_EXPERTS_PER_GPU=$((PROD_NUM_EXPERTS / (PROD_EP * PROD_FSDP)))
TARGET_MOE_HIDDEN_DIM=${PROD_BASE_EMB_DIM}

SLICE_NUM_EXPERTS=$((TARGET_EXPERTS_PER_GPU * EP * FSDP))
SLICE_BASE_EMB_DIM=${TARGET_MOE_HIDDEN_DIM}
MXFP8_ALIGNMENT=128
TOTAL_GPUS=$((EP * FSDP))
if (( TOTAL_GPUS != 4 )); then
  echo "This script is intended for one 4-GPU node. Got EP*FSDP=${TOTAL_GPUS}."
  exit 1
fi

if (( PROD_NUM_EXPERTS % PROD_EP != 0 )); then
  echo "Production num_experts=${PROD_NUM_EXPERTS} is not divisible by EP=${PROD_EP}."
  exit 1
fi

if (( PROD_BASE_EMB_DIM % PROD_FSDP != 0 )); then
  echo "Production base_emb_dim=${PROD_BASE_EMB_DIM} is not divisible by FSDP=${PROD_FSDP}."
  exit 1
fi

TOKENS_PER_LOCAL_GPU=$((BATCH_PER_GPU * MAX_TARGET_LENGTH))
ROUTED_ROWS_PER_LOCAL_GPU=$((TOKENS_PER_LOCAL_GPU * NUM_EXPERTS_PER_TOK))
LOCAL_MOE_HIDDEN=${SLICE_BASE_EMB_DIM}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TOTAL_GPUS - 1)))}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${OUTPUT_ROOT}/dsv3_prod_ep${PROD_EP}_fsdp${PROD_FSDP}_slice_ep${EP}_fsdp${FSDP}_${CUTEDSL_LABEL}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG_DIR="${RESULTS_DIR}/logs"
JAX_CACHE_DIR="${RESULTS_DIR}/jax_cache"
mkdir -p "${LOG_DIR}" "${JAX_CACHE_DIR}"

# Match the launcher's DSv3 XLA thresholds while retaining the baseline GPU flags
# used by test-maxtext.sh.
export BASE_XLA_FLAGS="${BASE_XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN \
--xla_gpu_all_reduce_combine_threshold_bytes=33554432 \
--xla_gpu_all_gather_combine_threshold_bytes=3355443200 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=3355443200 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false \
--xla_disable_hlo_passes=rematerialization}"

PROFILE_ARGS=""
if [[ "${PROFILE:-0}" == "1" ]]; then
  PROFILE_ARGS="profiler=xplane skip_first_n_steps_for_profiler=${PROFILE_SKIP_STEPS:-10} profiler_steps=${PROFILE_STEPS:-3}"
fi

ADDITIONAL_ARGS="\
jax_cache_dir=${JAX_CACHE_DIR} \
override_model_config=true \
max_target_length=${MAX_TARGET_LENGTH} \
base_emb_dim=${SLICE_BASE_EMB_DIM} \
base_mlp_dim=${PROD_BASE_MLP_DIM} \
base_moe_mlp_dim=${PROD_BASE_MOE_MLP_DIM} \
base_num_decoder_layers=${NUM_DECODER_LAYERS} \
first_num_dense_layers=${FIRST_NUM_DENSE_LAYERS} \
num_experts=${SLICE_NUM_EXPERTS} \
num_experts_per_tok=${NUM_EXPERTS_PER_TOK} \
sparse_matmul=true \
megablox=false \
capacity_factor=1.0 \
logits_dot_in_fp32=false \
use_iota_embed=false \
use_ring_of_experts=false \
ici_expert_parallelism=${EP} \
ici_fsdp_parallelism=${FSDP} \
ici_data_parallelism=1 \
dcn_data_parallelism=1 \
dcn_fsdp_parallelism=1 \
dcn_expert_parallelism=1 \
max_segments_per_seq=32 \
te_router_and_permutation_impl=false \
moe_permutation_group_align_size=128 \
te_moe_block=true \
te_moe_cutedsl_fusion=${CUTEDSL_CONFIG} \
te_use_gmm=false \
shard_exp_on_fsdp=true \
sharding_tolerance=1.0 \
te_gmm_quantization=${TE_GMM_QUANTIZATION} \
quantization=${QUANTIZATION} \
weight_dtype=bfloat16 \
mu_dtype=bfloat16 \
shardy=true \
scan_layers=true \
decoder_layer_input=device \
context=remat \
mlpwi=remat \
mlpwi_0=remat \
mlpwi_1=remat \
mlpwo=remat \
query_proj=remat \
key_proj=remat \
value_proj=remat \
out_proj=device \
abort_on_nan_loss=false \
${PROFILE_ARGS} \
${EXTRA_MAXTEXT_ARGS:-}"

cat <<EOF | tee "${RESULTS_DIR}/slice-summary.txt"
Production target:   FSDP=${PROD_FSDP}, EP=${PROD_EP}, experts=${PROD_NUM_EXPERTS}, emb=${PROD_BASE_EMB_DIM}
Local run:           FSDP=${FSDP}, EP=${EP}, experts=${SLICE_NUM_EXPERTS}, emb=${SLICE_BASE_EMB_DIM}

Matched routed-MoE slices:
  experts/GPU:   production ${TARGET_EXPERTS_PER_GPU}, local $((SLICE_NUM_EXPERTS / (EP * FSDP)))
  MoE hidden/GPU: production ${TARGET_MOE_HIDDEN_DIM}, local ${SLICE_BASE_EMB_DIM}
  moe_mlp_dim:   production ${PROD_BASE_MOE_MLP_DIM}, local ${PROD_BASE_MOE_MLP_DIM}

Quantization:
  quantization=${QUANTIZATION}
  te_gmm_quantization=${TE_GMM_QUANTIZATION}

MXFP8 K-dim check:
  local MoE hidden=${LOCAL_MOE_HIDDEN}, alignment=${MXFP8_ALIGNMENT}, remainder=$((LOCAL_MOE_HIDDEN % MXFP8_ALIGNMENT))
  MoE MLP dim=${PROD_BASE_MOE_MLP_DIM}, alignment=${MXFP8_ALIGNMENT}, remainder=$((PROD_BASE_MOE_MLP_DIM % MXFP8_ALIGNMENT))

MoE implementation:
  te_moe_block=true
  te_moe_cutedsl_fusion=${CUTEDSL_CONFIG}
  NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=${NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION}
  te_use_gmm=false
  one GPU per process=true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
Results dir: ${RESULTS_DIR}
EOF

VALIDATION_FAILED=0
if [[ "${TE_GMM_QUANTIZATION}" == "te_mxfp8" ]]; then
  if (( LOCAL_MOE_HIDDEN % MXFP8_ALIGNMENT != 0 )); then
    echo "Invalid MXFP8 local hidden dimension: ${LOCAL_MOE_HIDDEN} is not divisible by ${MXFP8_ALIGNMENT}." | tee -a "${RESULTS_DIR}/slice-summary.txt"
    VALIDATION_FAILED=1
  fi
  if (( PROD_BASE_MOE_MLP_DIM % MXFP8_ALIGNMENT != 0 )); then
    echo "Invalid MXFP8 MoE MLP dimension: ${PROD_BASE_MOE_MLP_DIM} is not divisible by ${MXFP8_ALIGNMENT}." | tee -a "${RESULTS_DIR}/slice-summary.txt"
    VALIDATION_FAILED=1
  fi
fi

if (( VALIDATION_FAILED )); then
  exit 1
fi

read -r -a ADDITIONAL_ARGV <<< "${ADDITIONAL_ARGS}"
CMD=(
  python3
  -m
  maxtext.trainers.pre_train.train
  "${MAXTEXT_DIR}/src/maxtext/configs/base.yml"
  run_name=run
  model_name="${MODEL}"
  steps="${STEPS}"
  per_device_batch_size="${BATCH_PER_GPU}"
  remat_policy="${REMAT_POLICY}"
  attention="${ATTENTION}"
  enable_checkpointing=false
  base_output_directory="${RESULTS_DIR}"
  dataset_path=local
  dataset_type=synthetic
  hardware=gpu_multiprocess
  enable_goodput_recording=false
  monitor_goodput=false
  enable_checkpoint_cloud_logger=false
  "${ADDITIONAL_ARGV[@]}"
)

printf "Command:" | tee -a "${RESULTS_DIR}/slice-summary.txt"
printf " %q" "${CMD[@]}" | tee -a "${RESULTS_DIR}/slice-summary.txt"
printf "\n" | tee -a "${RESULTS_DIR}/slice-summary.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

START_TS="$(date +%s)"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 $((TOTAL_GPUS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="${rank}"
    export OMPI_COMM_WORLD_SIZE="${TOTAL_GPUS}"
    export OMPI_COMM_WORLD_RANK="${rank}"
    export OMPI_COMM_WORLD_LOCAL_RANK="${rank}"
    export MAXTEXT_JAX_COORDINATOR_ADDRESS="127.0.0.1:${COORDINATOR_PORT}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
    if [[ "${PROFILE:-0}" == "1" ]]; then
      RANK_HLO_DIR="${RESULTS_DIR}/hlo/rank_${rank}"
      mkdir -p "${RANK_HLO_DIR}"
      export BASE_XLA_FLAGS="${BASE_XLA_FLAGS} --xla_dump_to=${RANK_HLO_DIR} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
    fi
    cd "${MAXTEXT_DIR}"
    "${CMD[@]}"
  ) >"${LOG_DIR}/proc_${rank}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for rank in "${!PIDS[@]}"; do
  if ! wait "${PIDS[${rank}]}"; then
    echo "rank=${rank} failed; see ${LOG_DIR}/proc_${rank}.log" | tee -a "${RESULTS_DIR}/summary.txt"
    FAILED=1
  fi
done

ELAPSED=$(($(date +%s) - START_TS))
echo "exit=${FAILED} elapsed=${ELAPSED}s" | tee -a "${RESULTS_DIR}/summary.txt"
echo "Output dir: ${RESULTS_DIR}" | tee -a "${RESULTS_DIR}/summary.txt"
sed -nE 's/.*completed step: ([0-9]+),.* loss: ([^, ]+), lm_loss:.*/\1 \2/p' \
  "${LOG_DIR}/proc_0.log" > "${RESULTS_DIR}/loss-curve.txt"
if (( FAILED != 0 )); then
  tail -n 80 "${LOG_DIR}/proc_0.log" || true
fi
exit "${FAILED}"
