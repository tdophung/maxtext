# Copyright 2023-2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformerEngine Router Integration for MaxText MoE.

This module provides wrapper functions for TransformerEngine's fused MoE router
operations used in Mixture of Experts (MoE) models.

The TE router fuses score_function + top-k selection + expert bias + scaling
into a single CUDA kernel with proper automatic differentiation support.

The integration provides:
- te_fused_topk: Fused score_function(logits) → [bias] → top-k → [post-softmax] → scale.
  Returns (sparse_probs, routing_map) in the format expected by TE permutation.
- te_compute_aux_scores: Clean score_function(logits) → top-k (no bias/groups/scaling).
  Returns dense scores for aux loss computation.
- te_aux_loss: Fused MoE auxiliary load-balancing loss computation.

Key Design:
- TE router takes raw GEMM logits (before any score function or bias) and handles
  score_function (softmax/sigmoid), expert_bias, grouped top-k, use_pre_softmax,
  and scaling_factor internally via fused CUDA kernels.
- The output format (sparse_probs, routing_map) feeds directly into TE permutation's
  token_dispatch without needing index-to-mask or weight-to-dense-probs conversion.
"""

from typing import Optional, Tuple

import jax.numpy as jnp

try:
  from transformer_engine.jax.router import (
      fused_topk_with_score_function,
      fused_moe_aux_loss,
      ScoreFunction,
  )
  TE_ROUTER_AVAILABLE = True
except ImportError:
  TE_ROUTER_AVAILABLE = False
  fused_topk_with_score_function = None
  fused_moe_aux_loss = None
  ScoreFunction = None


def check_te_router_available():
  """Check if TransformerEngine router is available."""
  if not TE_ROUTER_AVAILABLE:
    raise ImportError(
        "TransformerEngine router is not available. "
        "Please install TransformerEngine with JAX support: "
        "pip install transformer-engine[jax]"
    )


def get_te_score_function(score_func_str: str):
  """Convert MaxText score function string to TE ScoreFunction enum.

  Args:
    score_func_str: Score function name from MaxText config
      ("sigmoid", "softmax", or "" which defaults to softmax).

  Returns:
    TE ScoreFunction enum value.
  """
  check_te_router_available()
  if score_func_str == "sigmoid":
    return ScoreFunction.SIGMOID
  return ScoreFunction.SOFTMAX


def te_fused_topk(
    logits: jnp.ndarray,
    topk: int,
    score_function: str = "",
    use_pre_softmax: bool = False,
    num_groups: int = -1,
    group_topk: int = -1,
    scaling_factor: float = 1.0,
    expert_bias: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Fused top-k routing using TransformerEngine's CUDA kernel.

  Pipeline: score_function(logits) → [optional bias] → top-k →
            [optional post-softmax] → scale.

  This is a plug-and-play alternative to MaxText's get_topk() method.
  It takes raw GEMM logits and produces (sparse_probs, routing_map)
  which can be fed directly to TE permutation's token_dispatch.

  Args:
    logits: Raw GEMM logits of shape [num_tokens, num_experts].
      These should be the direct output of the gating linear layer,
      before any score function (softmax/sigmoid) or bias is applied.
    topk: Number of top experts to select per token.
    score_function: Score function to apply to logits.
      "softmax", "sigmoid", or "" (defaults to softmax).
    use_pre_softmax: If True, apply softmax before top-k selection.
      If False (default), apply softmax after top-k (post-softmax).
      Only relevant when score_function is softmax.
    num_groups: Number of groups for grouped top-k routing (e.g., DeepSeek).
      <= 0 disables grouping (default).
    group_topk: Top-k at group level for grouped routing.
      <= 0 disables group-level selection (default).
    scaling_factor: Scaling factor applied to output probabilities.
    expert_bias: Expert bias of shape [num_experts].
      Added to scores before top-k selection. Only used with sigmoid
      score function (e.g., DeepSeek V3 loss-free load balancing).
      Pass None if unused.

  Returns:
    sparse_probs: Sparse probability tensor of shape [num_tokens, num_experts].
      Non-zero only at the top-k selected expert positions per token.
    routing_map: Boolean mask of shape [num_tokens, num_experts].
      True at selected expert positions.
  """
  check_te_router_available()

  te_score_func = get_te_score_function(score_function)

  sparse_probs, routing_map = fused_topk_with_score_function(
      logits,
      topk=topk,
      use_pre_softmax=use_pre_softmax,
      num_groups=num_groups,
      group_topk=group_topk,
      scaling_factor=scaling_factor,
      score_function=te_score_func,
      expert_bias=expert_bias,
  )

  return sparse_probs, routing_map


def te_compute_aux_scores(
    logits: jnp.ndarray,
    topk: int,
    score_function: str = "",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute clean dense scores for auxiliary loss using TE fused kernel.

  Runs score_function(logits) → top-k with no bias, groups, or scaling.
  Returns dense scores (all expert positions non-zero) suitable for
  computing the auxiliary load-balancing loss.

  Args:
    logits: Raw GEMM logits of shape [num_tokens, num_experts].
    topk: Number of top experts to select per token.
    score_function: Score function name ("softmax", "sigmoid", or "").

  Returns:
    aux_scores: Dense score tensor [num_tokens, num_experts].
      All expert positions contain scores (not sparse).
    routing_map: Boolean mask [num_tokens, num_experts].
      True at selected expert positions (clean selection, no bias).
  """
  check_te_router_available()
  te_score_func = get_te_score_function(score_function)

  aux_scores, routing_map = fused_topk_with_score_function(
      logits,
      topk=topk,
      compute_aux_scores=True,
      score_function=te_score_func,
  )

  return aux_scores, routing_map


def te_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    topk: int,
    coeff: float,
) -> jnp.ndarray:
  """Compute MoE auxiliary load-balancing loss using TE fused kernel.

  loss = (E * coeff / (k * T^2)) * sum_i(sum_t(probs[t,i]) * tokens_per_expert[i])

  where T = num_tokens, E = num_experts, k = topk.

  Args:
    probs: Score tensor from te_compute_aux_scores [num_tokens, num_experts].
    tokens_per_expert: Token counts per expert [num_experts]. Integer tensor.
    topk: Top-k value.
    coeff: Loss coefficient (e.g., load_balance_loss_weight).

  Returns:
    Scalar loss value.
  """
  check_te_router_available()
  return fused_moe_aux_loss(probs, tokens_per_expert, topk, coeff)
