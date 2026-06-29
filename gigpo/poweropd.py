# Copyright 2025 TGSA-GRPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PowerOPD core: bounded power-based rewards replacing log-ratio.

Reference: PowerOPD (arXiv:2606.17199) Section 3 & 4.

Key insight: log-ratio rewards log(π_T/π_θ) are unbounded (-∞ to +∞),
causing training instability when π→0. PowerOPD replaces with:

    Δ^α = π_T^α - π_θ^α,  where α > 0

Properties:
    * Bounded: Δ^α ∈ [-1, 1] naturally, no need for tanh/clamp
    * Sign-consistent: sign(Δ^α) = sign(π_T - π_θ) = sign(log π_T - log π_θ)
    * α controls focus: larger α emphasizes high-probability tokens only
    * Gradient-friendly: avoids extreme negative values when π→0

This module provides the building blocks for both TGSA (Case 2 singleton
signal, and margin mode) and pure OPD (KL distillation).
"""

from typing import Optional, Tuple
import torch


def logprob_to_prob(log_prob: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Convert log-probabilities to probabilities.

    Args:
        log_prob: Log-probabilities (can be unnormalized)
        normalize: If True, apply softmax normalization (for logits)

    Returns:
        Probabilities in (0, 1]
    """
    if normalize:
        return torch.softmax(log_prob, dim=-1)
    return torch.exp(log_prob)


def compute_power_diff(
    log_prob_teacher: torch.Tensor,
    log_prob_student: torch.Tensor,
    alpha: float = 5.0,
    length_normalize: bool = False,
    response_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute bounded power difference: π_T^α - π_θ^α.

    This is the core PowerOPD replacement for unbounded log-ratio.

    Args:
        log_prob_teacher: Teacher log-probs, shape (...,)
        log_prob_student: Student log-probs, same shape
        alpha: Power coefficient (default 5.0, higher = more selective)
        length_normalize: If True, average over response_mask first (for per-step)
        response_mask: Optional mask for length normalization
        eps: Numerical stability guard

    Returns:
        Power difference ∈ [-1, 1], same shape as inputs
    """
    # Convert to probabilities
    prob_T = torch.exp(log_prob_teacher)
    prob_S = torch.exp(log_prob_student)

    if length_normalize and response_mask is not None:
        # Average probs over the action span
        mask = response_mask.float()
        valid_len = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        prob_T = (prob_T * mask).sum(dim=-1) / (valid_len.squeeze(-1) + eps)
        prob_S = (prob_S * mask).sum(dim=-1) / (valid_len.squeeze(-1) + eps)

    # Power transformation
    power_T = torch.pow(prob_T.clamp(min=eps), alpha)
    power_S = torch.pow(prob_S.clamp(min=eps), alpha)

    # Bounded difference
    diff = power_T - power_S  # ∈ [-1, 1] naturally
    return diff


def compute_power_margin(
    log_prob_teacher: torch.Tensor,
    log_prob_runnerup: torch.Tensor,
    alpha: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute power-based margin: π_T^α - π_runnerup^α.

    For Case 1 group ranking with margin mode. Replaces unbounded log-margin.

    Args:
        log_prob_teacher: Teacher log-prob of selected action
        log_prob_runnerup: Teacher log-prob of best alternative (top-2)
        alpha: Power coefficient
        eps: Numerical stability guard

    Returns:
        Power margin ∈ [-1, 1]
    """
    prob_T = torch.exp(log_prob_teacher)
    prob_R = torch.exp(log_prob_runnerup)

    power_T = torch.pow(prob_T.clamp(min=eps), alpha)
    power_R = torch.pow(prob_R.clamp(min=eps), alpha)

    margin = power_T - power_R  # ∈ [-1, 1]
    return margin


def compute_power_kl_token(
    log_prob_student: torch.Tensor,
    log_prob_teacher: torch.Tensor,
    alpha: float = 5.0,
    use_logprob_gradient: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PowerOPD-style KL: use bounded power diff as token-level reward.

    Standard reverse KL: KL = E[log π_θ - log π_T] (unbounded)
    PowerOPD variant: reward = π_T^α - π_θ^α (bounded), stop-grad on reward

    Args:
        log_prob_student: Student log-probs (bs, seq_len)
        log_prob_teacher: Teacher log-probs (bs, seq_len)
        alpha: Power coefficient
        use_logprob_gradient: If True, multiply by log π_θ gradient (policy grad style)
        eps: Numerical stability guard

    Returns:
        Per-token contribution (bs, seq_len). When use_logprob_gradient=False,
        this is just the power diff (stop-gradient ready).
    """
    # Bounded power difference (stop-gradient candidate)
    with torch.no_grad():
        power_diff = compute_power_diff(log_prob_teacher, log_prob_student, alpha, eps=eps)

    if use_logprob_gradient:
        # Policy gradient style: reward * log π_θ
        # This makes the loss: -power_diff * log_prob_student
        return -power_diff * log_prob_student
    else:
        # Raw bounded reward (caller applies gradient manually)
        return power_diff


def compute_power_advantage_singleton(
    log_prob_teacher: torch.Tensor,
    log_prob_student: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 5.0,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute PowerOPD-based teacher preference signal for singleton states.

    Replaces: tanh(gamma * (log π_T - log π_θ - μ)/σ)
    With: gamma * (π_T^α - π_θ^α)  [already bounded, no tanh needed]

    Args:
        log_prob_teacher: (bs, seq_len)
        log_prob_student: (bs, seq_len)
        response_mask: (bs, seq_len)
        alpha: Power coefficient
        gamma: Scaling factor (optional, default 1.0)
        eps: Numerical stability guard

    Returns:
        (bs,) per-step preference signal ∈ [-gamma, gamma]
    """
    # Length-normalized power difference per step
    diff = compute_power_diff(
        log_prob_teacher,
        log_prob_student,
        alpha=alpha,
        length_normalize=True,
        response_mask=response_mask,
        eps=eps,
    )  # (bs,)

    # Scale (no tanh needed since already bounded)
    return gamma * diff


def compute_power_opd_advantage(
    log_prob_teacher: torch.Tensor,
    log_prob_student: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 5.0,
    kl_coef: float = 1.0,
    use_policy_gradient: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PowerOPD advantage: -coef * (bounded power KL approximation).

    Standard OPD: advantage = -coef * KL(log π_θ || log π_T) [unbounded log-ratio]
    PowerOPD: advantage = -coef * mean(power_diff) [bounded, stable]

    Args:
        log_prob_teacher: (bs, seq_len)
        log_prob_student: (bs, seq_len)
        response_mask: (bs, seq_len)
        alpha: Power coefficient
        kl_coef: Distillation strength
        use_policy_gradient: If True, returns PG-style advantage; else raw reward
        eps: Numerical stability guard

    Returns:
        (advantages, returns): both (bs, seq_len), masked
    """
    # Per-token power difference
    power_diff = compute_power_diff(log_prob_teacher, log_prob_student, alpha, eps=eps)
    # (bs, seq_len)

    # Mask and aggregate
    mask = response_mask.float()

    if use_policy_gradient:
        # Policy gradient: advantage = -coef * power_diff
        # (gradient flows through student's policy)
        adv = -kl_coef * power_diff
    else:
        # Raw bounded KL proxy (for compatibility with existing code)
        adv = -kl_coef * power_diff

    # Apply mask
    adv = adv * mask

    return adv, adv  # returns = advantages for pure OPD
