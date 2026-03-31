#!/bin/bash
# =============================================================================
# profile_moe_comparison.sh
#
# Profiles MaxText MoE training with xplane to compare:
#   Run 1: TE router + TE permutation implementation
#   Run 2: MT (MaxText) router + MT permutation implementation
#
# Uses mixtral-8x7b with 1 decoder layer (base_num_decoder_layers=1), EP=2, FSDP=2.
# Default: 3 warmup steps (JIT + Triton autotune), then 1 profiled step.
#
# IMPORTANT: Traces are saved to a SHARED filesystem (Lustre) so they persist
# after the srun/enroot session ends and are accessible from the login node.
#
# Workflow (multi-hop: local -> ptyche login -> compute node via srun):
#   1. On ptyche, srun into a compute node with GPUs
#   2. Inside the container, run this script
#   3. After completion, exit srun — traces remain on Lustre
#   4. On ptyche, launch xprof pointing to the Lustre trace dirs
#   5. From local machine, SSH-tunnel to ptyche to view in browser
#
# Usage:
#   bash profile_moe_comparison.sh               # runs both
#   bash profile_moe_comparison.sh --run te       # only TE impl
#   bash profile_moe_comparison.sh --run mt       # only MT impl
#   bash profile_moe_comparison.sh --launch-xprof # run both + launch xprof
# =============================================================================

set -euo pipefail

MAXTEXT_DIR="${MAXTEXT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Default to Lustre (shared filesystem) so traces survive srun exit
PROFILE_BASE_DIR="${MAXTEXT_DIR}/xprof_profiles"
EP=2
FSDP=2
WARMUP_STEPS=3
PROFILE_STEPS=1
STEPS=$((WARMUP_STEPS + PROFILE_STEPS))
BATCH_SIZE=2
SEQ_LEN=4096
MODEL_NAME="mixtral-8x7b"

RUN_SELECTION="both"
RING_SELECTION="both"
LAUNCH_XPROF=false
DUMP_HLO=false
CMD_BUFFER_PATCH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --run)       RUN_SELECTION="$2"; shift 2 ;;
        --ring)      RING_SELECTION="$2"; shift 2 ;;
        --launch-xprof) LAUNCH_XPROF=true; shift ;;
        --ep)        EP="$2"; shift 2 ;;
        --fsdp)      FSDP="$2"; shift 2 ;;
        --warmup)    WARMUP_STEPS="$2"; STEPS=$((WARMUP_STEPS + PROFILE_STEPS)); shift 2 ;;
        --profile-steps) PROFILE_STEPS="$2"; STEPS=$((WARMUP_STEPS + PROFILE_STEPS)); shift 2 ;;
        --batch)     BATCH_SIZE="$2"; shift 2 ;;
        --seq-len)   SEQ_LEN="$2"; shift 2 ;;
        --output-dir) PROFILE_BASE_DIR="$2"; shift 2 ;;
        --dump-hlo)  DUMP_HLO=true; shift ;;
        --cmd-buffer-patch) CMD_BUFFER_PATCH=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--run te|mt|both] [--ring on|off|both] [--launch-xprof] [--ep N] [--fsdp N]"
            echo "  --run te          : only TE impl runs"
            echo "  --run mt          : only MT impl runs"
            echo "  --run both        : both TE and MT (default)"
            echo "  --ring on         : only ring-of-experts runs"
            echo "  --ring off        : only no-ring runs"
            echo "  --ring both       : both ring and no-ring (default)"
            echo "  --warmup N        : warmup steps before profiling (default: 3)"
            echo "  --profile-steps N : steps to profile (default: 1)"
            echo "  --dump-hlo        : dump HLO text to {output_dir}/{run_name}/hlo_dump/"
            echo "  --cmd-buffer-patch: enable triton_kernel_call command buffer capture"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

STEPS=$((WARMUP_STEPS + PROFILE_STEPS))
TOTAL_GPUS=$((EP * FSDP))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1)))

CONFIGS_DIR="${MAXTEXT_DIR}/src/maxtext/configs"
TOKENIZER_PATH="${MAXTEXT_DIR}/src/maxtext/assets/tokenizers/tokenizer.mistral-v1"

# Ensure missing deps are available (local branch may need extras not in the container)
echo "Checking dependencies..."
for mod in drjax; do
    if ! python3 -c "import ${mod}" 2>/dev/null; then
        echo "  Installing missing dep: ${mod}"
        pip install -q "${mod}" 2>&1 | tail -1
    fi
done

echo "=============================================="
echo "MoE XProf Profiling Comparison"
echo "=============================================="
echo "MaxText dir:   ${MAXTEXT_DIR}"
echo "Profile dir:   ${PROFILE_BASE_DIR}"
echo "Model:         ${MODEL_NAME} (1 decoder layer)"
echo "Warmup steps:  ${WARMUP_STEPS} (JIT + Triton autotune)"
echo "Profile steps: ${PROFILE_STEPS} (steady-state only)"
echo "Total steps:   ${STEPS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Seq len:       ${SEQ_LEN}"
echo "EP:            ${EP}"
echo "FSDP:          ${FSDP}"
echo "Total GPUs:    ${TOTAL_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Run selection: ${RUN_SELECTION}"
echo "=============================================="

# Common training args shared by both runs
COMMON_ARGS=(
    "${CONFIGS_DIR}/base.yml"
    model_name=${MODEL_NAME}
    base_num_decoder_layers=1
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
    tokenizer_path=${TOKENIZER_PATH}
    sparse_matmul=true
    capacity_factor=1.0
    logits_dot_in_fp32=false
    ici_expert_parallelism=${EP}
    ici_data_parallelism=1
    ici_fsdp_parallelism=${FSDP}
    ici_tensor_parallelism=1
    max_segments_per_seq=32
    sharding_tolerance=1.0
    profiler=xplane
    skip_first_n_steps_for_profiler=${WARMUP_STEPS}
    profiler_steps=${PROFILE_STEPS}
    profile_cleanly=True
)

run_profile() {
    local run_name="$1"
    local te_perm="$2"
    local align_size="$3"
    local ring="$4"
    local output_dir="${PROFILE_BASE_DIR}/${run_name}"
    local log_file="${PROFILE_BASE_DIR}/${run_name}.log"
    local hlo_dump_dir="${output_dir}/hlo_dump"

    echo ""
    echo "======================================================"
    echo "  PROFILING: ${run_name}"
    echo "    te_permutation_impl=${te_perm}"
    echo "    te_permutation_align_size=${align_size}"
    echo "    use_ring_of_experts=${ring}"
    echo "    Warmup: ${WARMUP_STEPS} steps, Profile: ${PROFILE_STEPS} steps"
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

    local train_cmd="python3 -m maxtext.trainers.pre_train.train"
    if [[ "${CMD_BUFFER_PATCH}" == "true" ]]; then
        echo "    [CMD_BUFFER_PATCH] Enabling triton_kernel_call command buffer capture"
        train_cmd="python3 -c 'import maxtext.triton_cmd_buffer_patch; from absl import app; from maxtext.trainers.pre_train.train import main; app.run(main)'"
    fi

    XLA_FLAGS="${xla_flags_env}" \
    PYTHONPATH="${MAXTEXT_DIR}/src:${PYTHONPATH:-}" DECOUPLE_GCLOUD=TRUE \
        eval ${train_cmd} \
        "${COMMON_ARGS[@]}" \
        run_name="${run_name}" \
        base_output_directory="${PROFILE_BASE_DIR}" \
        te_permutation_impl=${te_perm} \
        te_permutation_align_size=${align_size} \
        use_ring_of_experts=${ring} \
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
# Run 1: TE router + TE permutation (no ring)
# =====================================================
if [[ ("${RUN_SELECTION}" == "both" || "${RUN_SELECTION}" == "te") && \
      ("${RING_SELECTION}" == "both" || "${RING_SELECTION}" == "off") ]]; then
    run_profile "te_impl_no_ring" true 128 false || true
fi

# =====================================================
# Run 2: MT router + MT permutation (no ring)
# =====================================================
if [[ ("${RUN_SELECTION}" == "both" || "${RUN_SELECTION}" == "mt") && \
      ("${RING_SELECTION}" == "both" || "${RING_SELECTION}" == "off") ]]; then
    run_profile "mt_impl_no_ring" false 0 false || true
fi

# =====================================================
# Run 3: TE router + TE permutation + ring of experts
# =====================================================
if [[ ("${RUN_SELECTION}" == "both" || "${RUN_SELECTION}" == "te") && \
      ("${RING_SELECTION}" == "both" || "${RING_SELECTION}" == "on") ]]; then
    run_profile "te_impl_ring" true 128 true || true
fi

# =====================================================
# Run 4: MT router + MT permutation + ring of experts
# =====================================================
if [[ ("${RUN_SELECTION}" == "both" || "${RUN_SELECTION}" == "mt") && \
      ("${RING_SELECTION}" == "both" || "${RING_SELECTION}" == "on") ]]; then
    run_profile "mt_impl_ring" false 0 true || true
fi

# =====================================================
# Summary and xprof instructions
# =====================================================
echo ""
echo "=============================================="
echo "Profiling Complete"
echo "=============================================="
echo ""
echo "Profile traces saved to:"

TE_NO_RING_DIR="${PROFILE_BASE_DIR}/te_impl_no_ring/tensorboard/"
MT_NO_RING_DIR="${PROFILE_BASE_DIR}/mt_impl_no_ring/tensorboard/"
TE_RING_DIR="${PROFILE_BASE_DIR}/te_impl_ring/tensorboard/"
MT_RING_DIR="${PROFILE_BASE_DIR}/mt_impl_ring/tensorboard/"

for d in "${TE_NO_RING_DIR}" "${MT_NO_RING_DIR}" "${TE_RING_DIR}" "${MT_RING_DIR}"; do
    label=$(basename "$(dirname "$(dirname "$d")")")
    [[ -d "$d" ]] && echo "  ${label}: ${d}"
done

echo ""
echo "=============================================="
echo "  HOW TO VIEW (multi-hop setup)"
echo "=============================================="
echo ""
echo "Traces are on the shared Lustre filesystem, so they"
echo "persist after the srun session ends."
echo ""
echo "NOTE: Only the steady-state step(s) are captured."
echo "  ${WARMUP_STEPS} warmup steps were run first to absorb JIT + Triton autotune overhead."
echo ""
echo "STEP 1: Exit srun (traces are safe on Lustre)."
echo ""
echo "STEP 2: On ptyche (login node), launch xprof (pick the runs to compare):"
echo "  xprof --port 8791 ${TE_NO_RING_DIR}   # TE, no ring"
echo "  xprof --port 8792 ${MT_NO_RING_DIR}   # MT, no ring"
echo "  xprof --port 8793 ${TE_RING_DIR}      # TE, ring"
echo "  xprof --port 8794 ${MT_RING_DIR}      # MT, ring"
echo ""
echo "  Or TensorBoard (all 4 runs in one UI):"
echo "  tensorboard --logdir=${PROFILE_BASE_DIR} --port 6006"
echo ""
echo "STEP 3: From your LOCAL machine, SSH-tunnel to ptyche:"
echo "  ssh -L 8791:localhost:8791 -L 8792:localhost:8792 \\"
echo "      -L 8793:localhost:8793 -L 8794:localhost:8794 ptyche"
echo ""
echo "STEP 4: Open in browser:"
echo "  http://localhost:8791  TE impl, no ring"
echo "  http://localhost:8792  MT impl, no ring"
echo "  http://localhost:8793  TE impl, ring of experts"
echo "  http://localhost:8794  MT impl, ring of experts"
echo "  Then: Tools > trace_viewer"
echo ""
echo "ALTERNATIVE: scp traces to local and run xprof locally:"
echo "  scp -r ptyche:${PROFILE_BASE_DIR} ~/xprof_profiles/"
echo ""
echo "In the trace viewer, look for these key operations:"
echo "  - MoE routing:      te_router (TE) vs TopKRouter (MT)"
echo "  - Local permute:    te_local_permute vs MT permute ops"
echo "  - Global permute:   te_global_permute vs ragged_all_to_all"
echo "  - Expert compute:   gmm / group matmul (same in both)"
echo "  - Local unpermute:  te_local_unpermute vs MT unpermute"
echo "  - Global unpermute: te_global_unpermute vs ragged_all_to_all rev"
echo "=============================================="

# Optionally launch xprof
if [[ "${LAUNCH_XPROF}" == "true" ]]; then
    echo ""
    echo "Launching xprof servers..."
    port=8791
    for d in "${TE_NO_RING_DIR}" "${MT_NO_RING_DIR}" "${TE_RING_DIR}" "${MT_RING_DIR}"; do
        label=$(basename "$(dirname "$(dirname "$d")")")
        if [[ -d "$d" ]]; then
            echo "  ${label} xprof at http://localhost:${port}/"
            xprof --port ${port} "$d" &
        fi
        port=$((port + 1))
    done
    echo ""
    echo "Press Ctrl+C to stop xprof servers."
    wait
fi

