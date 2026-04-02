#!/bin/bash
# =============================================================================
# profile_deepseek_te_1gpu.sh
#
# Single-GPU (EP=1, FSDP=1) profiling of TE router + permutation with
# sparse_matmul on a scaled-down DeepSeek V3 model (deepseek3-671b decoder
# block with reduced dimensions).
#
# Captures xplane profiling data + HLO dumps.
#
# For multi-GPU profiling with expert parallelism, see profile_moe_comparison.sh.
#
# Usage:
#   bash profile_deepseek_te_1gpu.sh                    # profile only
#   bash profile_deepseek_te_1gpu.sh --dump-hlo         # profile + HLO dump
#   bash profile_deepseek_te_1gpu.sh --launch-xprof     # profile + launch xprof
#   bash profile_deepseek_te_1gpu.sh --warmup 5         # more warmup steps
#   bash profile_deepseek_te_1gpu.sh --batch 4          # larger batch
# =============================================================================

set -euo pipefail

MAXTEXT_DIR="${MAXTEXT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROFILE_BASE_DIR="${MAXTEXT_DIR}/xprof_profiles_deepseek_1gpu"
EP=1
FSDP=1
WARMUP_STEPS=3
PROFILE_STEPS=1
BATCH_SIZE=2
SEQ_LEN=4096
MODEL_NAME="deepseek3-671b"

LAUNCH_XPROF=false
DUMP_HLO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup)         WARMUP_STEPS="$2"; shift 2 ;;
        --profile-steps)  PROFILE_STEPS="$2"; shift 2 ;;
        --batch)          BATCH_SIZE="$2"; shift 2 ;;
        --seq-len)        SEQ_LEN="$2"; shift 2 ;;
        --output-dir)     PROFILE_BASE_DIR="$2"; shift 2 ;;
        --dump-hlo)       DUMP_HLO=true; shift ;;
        --launch-xprof)   LAUNCH_XPROF=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --warmup N          warmup steps before profiling (default: 3)"
            echo "  --profile-steps N   steps to profile (default: 1)"
            echo "  --batch N           per-device batch size (default: 2)"
            echo "  --seq-len N         sequence length (default: 4096)"
            echo "  --output-dir DIR    output directory for profiles"
            echo "  --dump-hlo          dump HLO text alongside profile"
            echo "  --launch-xprof      launch xprof after profiling"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

STEPS=$((WARMUP_STEPS + PROFILE_STEPS))
TOTAL_GPUS=$((EP * FSDP))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1)))

CONFIGS_DIR="${MAXTEXT_DIR}/src/maxtext/configs"

echo "=============================================="
echo "DeepSeek V3 (mini) — TE Router + Permutation"
echo "  Single GPU | sparse_matmul | ring-of-experts"
echo "=============================================="
echo "MaxText dir:   ${MAXTEXT_DIR}"
echo "Profile dir:   ${PROFILE_BASE_DIR}"
echo "Model:         ${MODEL_NAME} (scaled down: emb=2048, experts=64, 1 layer)"
echo "Matmul:        sparse (ragged_dot)"
echo "Warmup steps:  ${WARMUP_STEPS}"
echo "Profile steps: ${PROFILE_STEPS}"
echo "Total steps:   ${STEPS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Seq len:       ${SEQ_LEN}"
echo "GPUs:          ${TOTAL_GPUS} (EP=${EP}, FSDP=${FSDP})"
echo "Dump HLO:      ${DUMP_HLO}"
echo "=============================================="

# Scaled-down DeepSeek V3: keeps the decoder_block=deepseek architecture
# (MLA attention, DeepSeek MoE routing with sigmoid + bias) but with
# dimensions small enough for single-GPU compilation and profiling.
COMMON_ARGS=(
    "${CONFIGS_DIR}/base.yml"
    model_name=${MODEL_NAME}
    hardware=gpu
    steps=${STEPS}
    per_device_batch_size=${BATCH_SIZE}
    max_target_length=${SEQ_LEN}
    attention=cudnn_flash_te
    remat_policy=minimal_with_context
    use_iota_embed=True
    dataset_type=synthetic
    reuse_example_batch=1
    enable_checkpointing=False
    scan_layers=False
    megablox=False
    sparse_matmul=true
    capacity_factor=1.0
    logits_dot_in_fp32=false
    ici_expert_parallelism=${EP}
    ici_data_parallelism=1
    ici_fsdp_parallelism=${FSDP}
    ici_tensor_parallelism=1
    max_segments_per_seq=32
    use_ring_of_experts=true
    sharding_tolerance=1.0
    profiler=xplane
    skip_first_n_steps_for_profiler=${WARMUP_STEPS}
    profiler_steps=${PROFILE_STEPS}
    profile_cleanly=True
    base_emb_dim=2048
    base_mlp_dim=2048
    base_moe_mlp_dim=2048
    base_num_decoder_layers=1
    first_num_dense_layers=0
    num_experts=64
    base_num_query_heads=128
    base_num_kv_heads=128
)

run_profile() {
    local run_name="$1"
    local te_impl="$2"
    local align_size="$3"
    local use_gmm="$4"
    local quant="$5"
    local output_dir="${PROFILE_BASE_DIR}/${run_name}"
    local log_file="${PROFILE_BASE_DIR}/${run_name}.log"
    local hlo_dump_dir="${output_dir}/hlo_dump"

    local gmm_label="ragged_dot"
    [[ "${use_gmm}" == "true" ]] && gmm_label="TE grouped GEMM"

    echo ""
    echo "======================================================"
    echo "  PROFILING: ${run_name}"
    echo "    te_router_and_permutation_impl=${te_impl}"
    echo "    te_permutation_align_size=${align_size}"
    echo "    te_use_gmm=${use_gmm}  (${gmm_label})"
    echo "    quantization=${quant:-none}"
    echo "    Output: ${output_dir}"
    if [[ "${DUMP_HLO}" == "true" ]]; then
        echo "    HLO dump: ${hlo_dump_dir}"
    fi
    echo "======================================================"

    rm -rf "${output_dir}"
    mkdir -p "${output_dir}"

    cd "${MAXTEXT_DIR}"

    local xla_flags_env=""
    if [[ "${DUMP_HLO}" == "true" ]]; then
        mkdir -p "${hlo_dump_dir}"
        xla_flags_env="--xla_dump_to=${hlo_dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_train_step"
    fi

    local quant_args=()
    if [[ -n "${quant}" ]]; then
        quant_args=(quantization="${quant}")
    fi

    XLA_FLAGS="${xla_flags_env}" \
    PYTHONPATH="${MAXTEXT_DIR}/src:${PYTHONPATH:-}" DECOUPLE_GCLOUD=TRUE \
        python3 -m maxtext.trainers.pre_train.train \
        "${COMMON_ARGS[@]}" \
        run_name="${run_name}" \
        base_output_directory="${PROFILE_BASE_DIR}" \
        te_router_and_permutation_impl=${te_impl} \
        te_permutation_align_size=${align_size} \
        te_use_gmm=${use_gmm} \
        "${quant_args[@]}" \
        2>&1 | tee "${log_file}"

    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo "  [OK] ${run_name} completed successfully."
        echo "  Profile data: ${output_dir}/tensorboard/"
    else
        echo ""
        echo "  [FAIL] ${run_name} failed with exit code ${exit_code}"
        echo "  See log: ${log_file}"
    fi
    echo ""
    return ${exit_code}
}

# =====================================================
# Run 1: TE router + TE permutation + TE grouped GEMM
# =====================================================
run_profile "dsv3_te_gmm" true 128 true te_fp8_currentscaling || true

# =====================================================
# Run 2: MT router + MT permutation + ragged_dot
# =====================================================
run_profile "dsv3_mt_impl" false 0 false "" || true

# =====================================================
# Run 3: TE router + TE permutation + ragged_dot
# =====================================================
run_profile "dsv3_te_ragged" true 0 false "" || true

# Summary
TE_GMM_DIR="${PROFILE_BASE_DIR}/dsv3_te_gmm/tensorboard/"
MT_DIR="${PROFILE_BASE_DIR}/dsv3_mt_impl/tensorboard/"
TE_RAGGED_DIR="${PROFILE_BASE_DIR}/dsv3_te_ragged/tensorboard/"

echo ""
echo "=============================================="
echo "Profiling Complete"
echo "=============================================="
echo ""
for d in "${TE_GMM_DIR}" "${MT_DIR}" "${TE_RAGGED_DIR}"; do
    label=$(basename "$(dirname "$(dirname "$d")")")
    [[ -d "$d" ]] && echo "  ${label}: ${d}"
done
if [[ "${DUMP_HLO}" == "true" ]]; then
    echo ""
    echo "HLO dumps:"
    [[ -d "${PROFILE_BASE_DIR}/dsv3_te_gmm/hlo_dump" ]]    && echo "  TE+GMM:     ${PROFILE_BASE_DIR}/dsv3_te_gmm/hlo_dump/"
    [[ -d "${PROFILE_BASE_DIR}/dsv3_mt_impl/hlo_dump" ]]    && echo "  MT:         ${PROFILE_BASE_DIR}/dsv3_mt_impl/hlo_dump/"
    [[ -d "${PROFILE_BASE_DIR}/dsv3_te_ragged/hlo_dump" ]]  && echo "  TE+ragged:  ${PROFILE_BASE_DIR}/dsv3_te_ragged/hlo_dump/"
fi
echo ""
echo "To view:"
echo "  xprof --port 8791 ${TE_GMM_DIR}      # Run 1: TE + TE GMM"
echo "  xprof --port 8792 ${MT_DIR}           # Run 2: MT + ragged_dot"
echo "  xprof --port 8793 ${TE_RAGGED_DIR}    # Run 3: TE + ragged_dot"
echo "  Then SSH tunnel: ssh -L 8791:localhost:8791 -L 8792:localhost:8792 -L 8793:localhost:8793 <host>"
echo "  Browser:"
echo "    http://localhost:8791  TE + TE GMM       → Tools > trace_viewer"
echo "    http://localhost:8792  MT + ragged_dot    → Tools > trace_viewer"
echo "    http://localhost:8793  TE + ragged_dot    → Tools > trace_viewer"
echo "=============================================="

if [[ "${LAUNCH_XPROF}" == "true" ]]; then
    echo ""
    echo "Launching xprof servers..."
    port=8791
    for d in "${TE_GMM_DIR}" "${MT_DIR}" "${TE_RAGGED_DIR}"; do
        label=$(basename "$(dirname "$(dirname "$d")")")
        if [[ -d "$d" ]]; then
            echo "  ${label} at http://localhost:${port}/"
            xprof --port ${port} "$d" &
        fi
        port=$((port + 1))
    done
    echo ""
    echo "Press Ctrl+C to stop xprof servers."
    wait
fi
