#!/bin/bash 
#"after_te_permute_x", "after_mt_permute_x"
#"after_ragged_all_to_all_fwd_x"
#"after_te_local_permute_x", "after_mt_local_permute_x"
#"after_gmm_intermediate_output"
#"after_te_local_unpermute", "after_mt_local_unpermute"
#"after_te_ragged_all_to_all_rev", "after_mt_ragged_all_to_all_rev"
#"after_te_unpermute_output", "after_mt_unpermute_output"


EP=2
FSDP=2
STEPS=20

# Use dtype=bfloat16 (default). With dtype=float32 the run can hit reshape errors.
# Optional: set NVTE_ROUTER_DEBUG_DUMP=1 to dump router tensors via TE inspect (my_tensor_gpu{N}.bin).
MIXTRAL_ARGS="\
max_target_length=4096 \
sparse_matmul=true \
megablox=false \
capacity_factor=1.0 \
logits_dot_in_fp32=false \
use_ring_of_experts=true \
ici_expert_parallelism=$EP \
ici_data_parallelism=1 \
ici_fsdp_parallelism=$FSDP \
te_router_and_permutation_impl=true \
max_segments_per_seq=32 \
te_permutation_align_size=128 \
base_num_decoder_layers=4 \
te_use_gmm=true
"

DSv3_ARGS="\
max_target_length=4096 \
sparse_matmul=true \
megablox=false \
capacity_factor=1.0 \
logits_dot_in_fp32=false \
use_ring_of_experts=true \
ici_expert_parallelism=$EP \
ici_data_parallelism=1 \
ici_fsdp_parallelism=$FSDP \
max_segments_per_seq=32 \
base_mlp_dim=2048 \
base_emb_dim=2048 \
base_num_decoder_layers=1 \
first_num_dense_layers=0 \
num_experts=256 \
te_router_and_permutation_impl=true \
te_permutation_align_size=128 \
te_use_gmm=true \
quantization=te_fp8_currentscaling
"

#sharding_tolerance=1.0

TOTAL_GPUS=$((EP * FSDP))
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1)))

MAXTEXT_DIR=/mnt/tdophung/prenyx_lustre_home/maxtext test-maxtext.sh -b 2 --model-name=deepseek3-671b --attn-type=cudnn_flash_te --remat-policy=minimal_with_context --steps=$STEPS --data-parallel=1 --fsdp=$FSDP --tensor-parallel=1 --tensor-sequence-parallel=1 -a "$DSv3_ARGS"


