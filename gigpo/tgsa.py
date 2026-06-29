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

"""TGSA-GRPO core math: teacher-guided step-level advantage modulation.

This module is deployment-agnostic. It operates purely on already-aligned per-row /
per-token tensors that the trainer places on the DataProto batch:

    teacher_log_prob : (bs, response_length)  per-token teacher log pi_T(w_k | s_t, w_{<k})
    old_log_probs    : (bs, response_length)  per-token student log pi_theta (behavior policy)
    response_mask    : (bs, response_length)  action-token span of each step (one row == one env turn)
    step_group_uids  : (bs,)  object array    anchor-state cluster id per row (from build_step_group)
    group_size_per_row : (bs,)                |G_t| per row (== step group size)
    episode_group_std  : (bs,)                sigma^R_group per row (episode/prompt group return std)

The advantage tensor returned by ``compute_tgsa_advantage`` has the SAME shape and
contract as GiGPO's ``scores`` (bs, response_length), so it drops straight into
``compute_gigpo_outcome_advantage`` -> ``batch['advantages']`` with no loss-side change.

Reference: idea.md (TGSA-GRPO).
"""

from typing import Optional

import numpy as np
import torch

# verl ships these; keep the import local-fallbackable so the unit test can run
# without the full verl package if needed.
try:
    from verl.utils.torch_functional import masked_mean
except Exception:  # pragma: no cover - fallback for standalone unit tests
    def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
        return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


# --------------------------------------------------------------------------- #
# 1. Length normalization of a per-token log-prob to a per-step scalar.       #
#    log pi(a_t | s_t) = (1 / |a_t|) * sum_k log pi(w_k | s_t, w_{1:k-1})      #
# --------------------------------------------------------------------------- #
def length_normalize_logprob(log_prob: torch.Tensor,
                             response_mask: torch.Tensor,
                             eps: float = 1e-6) -> torch.Tensor:
    """Average a per-token log-prob over the action-token span of each step.

    Args:
        log_prob: (bs, response_length) per-token log-prob (teacher or student).
        response_mask: (bs, response_length) {0,1} mask of the action tokens.
        eps: numerical guard for empty/zero-length spans.

    Returns:
        (bs,) length-normalized log-prob per step.
    """
    mask = response_mask.float()
    valid_len = mask.sum(dim=-1).clamp(min=1.0)
    return (log_prob * mask).sum(dim=-1) / (valid_len + eps)


# --------------------------------------------------------------------------- #
# 2. Teacher preference signal L_T_tilde in [-1, 1].                          #
#    Case 1 (|G_t| >= 2): group-internal ranking of TEACHER log-prob.         #
#    Case 2 (|G_t| == 1): teacher-student difference, tanh-compressed.         #
# --------------------------------------------------------------------------- #
def _group_stats_per_row(values_per_row: torch.Tensor,
                         group_uids: np.ndarray) -> tuple:
    """Return per-row (mean, std, min, max) of ``values_per_row`` within its group.

    Group key = step_group_uid. Singletons get std=1.0 (so z-score is 0 / well-defined),
    matching core_gigpo's convention.

    Returns four (bs,) tensors aligned to the input row order.
    """
    bs = values_per_row.shape[0]
    device = values_per_row.device
    uids = np.asarray(group_uids, dtype=object)

    id2idx: dict = {}
    for i in range(bs):
        id2idx.setdefault(uids[i], []).append(i)

    mean = torch.zeros(bs, device=device)
    std = torch.ones(bs, device=device)
    mn = torch.zeros(bs, device=device)
    mx = torch.zeros(bs, device=device)
    for _uid, idxs in id2idx.items():
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        g = values_per_row[idx]
        if len(idxs) == 1:
            # singleton: std sentinel = 1.0, mean = value (so z-score -> 0)
            mean[idx] = g
            std[idx] = torch.tensor(1.0, device=device)
            mn[idx] = g
            mx[idx] = g
        else:
            mean[idx] = g.mean()
            std[idx] = g.std(unbiased=False).clamp(min=1e-8)
            mn[idx] = g.min()
            mx[idx] = g.max()
    return mean, std, mn, mx


def _fill_signal_stats(out_stats: dict,
                       out_hists: Optional[dict],
                       *,
                       l_tilde: torch.Tensor,
                       delta: torch.Tensor,
                       mu_delta: torch.Tensor,
                       std_delta: torch.Tensor,
                       delta_norm_pre: torch.Tensor,
                       delta_norm_clip: float,
                       case1: torch.Tensor,
                       case2: torch.Tensor,
                       is_group: torch.Tensor,
                       singleton_mask: torch.Tensor,
                       teacher_lp: torch.Tensor,
                       step_group_uids: np.ndarray,
                       eps: float) -> None:
    """Populate ``tgsa_signal/*`` scalars + histograms in place (pure side-channel).

    No effect on the returned ``l_tilde``. All reductions are tensor->python-float
    via ``.item()``. Histograms (``out_hists``) carry the raw per-row arrays so the
    full distribution -- not just its mean -- is visible in tensorboard.
    """
    has_group = bool(is_group.any())
    has_sing = bool(singleton_mask.any())

    out_stats["tgsa_signal/l_tilde_mean"] = float(l_tilde.mean().item())
    out_stats["tgsa_signal/l_tilde_frac_pos"] = float((l_tilde > 0).to(torch.float32).mean().item())
    out_stats["tgsa_signal/l_tilde_abs_mean"] = float(l_tilde.abs().mean().item())
    # raw delta over all rows: should be systematically negative (= -D_KL in expectation)
    out_stats["tgsa_signal/delta_mean"] = float(delta.mean().item())
    out_stats["tgsa_signal/std_delta"] = float(std_delta.item()) if has_sing else 0.0

    # delta_norm / saturation stats over SINGLETON rows only (where Case-2 acts)
    if has_sing:
        dn = delta_norm_pre[singleton_mask]
        out_stats["tgsa_signal/delta_norm_mean"] = float(dn.mean().item())
        out_stats["tgsa_signal/delta_norm_max"] = float(dn.abs().max().item())
        out_stats["tgsa_signal/delta_norm_frac_gt3"] = float((dn.abs() > 3.0).to(torch.float32).mean().item())
        if delta_norm_clip and delta_norm_clip > 0.0:
            out_stats["tgsa_signal/clip_active_frac"] = float((dn.abs() > delta_norm_clip).to(torch.float32).mean().item())
        else:
            out_stats["tgsa_signal/clip_active_frac"] = 0.0
        if out_hists is not None:
            out_hists["tgsa_signal/delta_norm_hist"] = dn.detach().cpu().to(torch.float32)
    else:
        out_stats["tgsa_signal/delta_norm_mean"] = 0.0
        out_stats["tgsa_signal/delta_norm_max"] = 0.0
        out_stats["tgsa_signal/delta_norm_frac_gt3"] = 0.0
        out_stats["tgsa_signal/clip_active_frac"] = 0.0

    out_stats["tgsa_signal/case1_mean"] = float(case1[is_group].mean().item()) if has_group else 0.0
    out_stats["tgsa_signal/case2_mean"] = float(case2[singleton_mask].mean().item()) if has_sing else 0.0

    # Case1 group teacher spread: per-group (max-min) of teacher_lp, averaged over
    # group rows. Near-zero spread => teacher is indifferent WITHIN the anchor group,
    # so the minmax ranking is noise -- flags "Case1-eligible but signal useless".
    if has_group:
        _, _, mn_lp, mx_lp = _group_stats_per_row(teacher_lp, step_group_uids)
        spread = (mx_lp - mn_lp)
        spread_group = spread[is_group]
        out_stats["tgsa_signal/case1_teacher_spread_mean"] = float(spread_group.mean().item())
        # Fraction of Case-1 rows whose group spread < 0.1 nats: minmax denom is tiny
        # there, so case1 saturates to noise. High value => Case-1 signal unreliable.
        out_stats["tgsa_signal/case1_spread_frac_lt01"] = float(
            (spread_group < 0.1).to(torch.float32).mean().item()
        )
        if out_hists is not None:
            # Raw teacher_lp distribution for Case-1 rows: lets you see whether
            # log-prob values are tightly clustered (small dynamic range) or spread
            # across multiple nats (healthy signal).
            out_hists["tgsa_signal/case1_teacher_lp_hist"] = (
                teacher_lp[is_group].detach().cpu().to(torch.float32)
            )
            # Per-group spread (max-min) distribution: the denominator of minmax.
            # A heavy left tail near 0 confirms the saturation risk you flagged.
            out_hists["tgsa_signal/case1_spread_hist"] = (
                spread_group.detach().cpu().to(torch.float32)
            )
            # case1 value distribution (Case-1 rows only, pre-clamp).
            # Should be roughly uniform on [-1, 1] if minmax is healthy;
            # a bimodal spike at ±1 confirms saturation from near-zero spread.
            out_hists["tgsa_signal/case1_value_hist"] = (
                case1[is_group].detach().cpu().to(torch.float32)
            )
    else:
        out_stats["tgsa_signal/case1_teacher_spread_mean"] = 0.0
        out_stats["tgsa_signal/case1_spread_frac_lt01"] = 0.0

    if out_hists is not None:
        out_hists["tgsa_signal/l_tilde_hist"] = l_tilde.detach().cpu().to(torch.float32)


def _group_rank_per_row(values_per_row: torch.Tensor,
                        group_uids: np.ndarray) -> torch.Tensor:
    """Return per-row rank-normalized score in [-1, 1] within its step group.

    For a group of size N, ranks are 0..N-1 (ascending by value), then mapped:
        score = 2 * rank / (N - 1) - 1   for N >= 2   -> [-1, 1]
        score = 0.0                        for N == 1   (singleton sentinel)

    Properties that make this preferable to minmax when log-probs are extreme:
      * Immune to outliers: one extremely low log-prob can't compress all other
        rows into [-1, -0.9] as happens with minmax (situation C).
      * Immune to near-zero spread: even if max-min ≈ 0.001 nats, rank still
        discriminates the best from the worst (situation B is handled cleanly).
      * Strictly bounded in [-1, 1] without clamp.
      * Preserves ordinal preference signal while discarding magnitude noise.

    Args:
        values_per_row: (bs,) e.g. teacher_lp, already length-normalized.
        group_uids: (bs,) object array of step-group cluster ids.

    Returns:
        (bs,) float tensor of rank-normalized scores in [-1, 1].
    """
    bs = values_per_row.shape[0]
    device = values_per_row.device
    uids = np.asarray(group_uids, dtype=object)

    out = torch.zeros(bs, device=device, dtype=values_per_row.dtype)
    id2idx: dict = {}
    for i in range(bs):
        id2idx.setdefault(uids[i], []).append(i)

    for _uid, idxs in id2idx.items():
        n = len(idxs)
        if n == 1:
            out[idxs[0]] = 0.0          # singleton sentinel: neutral signal
            continue
        idx = torch.tensor(idxs, dtype=torch.long, device=device)
        g = values_per_row[idx]         # (n,)
        # argsort twice gives rank (0 = smallest)
        order = g.argsort()
        rank = torch.zeros_like(g)
        rank[order] = torch.arange(n, dtype=g.dtype, device=device)
        # map rank in [0, n-1] to score in [-1, 1]
        score = 2.0 * rank / (n - 1) - 1.0
        out[idx] = score
    return out                          # (bs,) in [-1, 1]


def compute_teacher_preference_signal(
    teacher_log_prob: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    group_size_per_row: torch.Tensor,
    normalization_mode: str = "minmax",
    gamma: float = 1.0,
    eps: float = 1e-6,
    teacher_top2_logprob: Optional[torch.Tensor] = None,
    use_margin: bool = False,
    delta_norm_clip: float = 0.0,
    out_stats: Optional[dict] = None,
    out_hists: Optional[dict] = None,
) -> torch.Tensor:
    """Compute the unified teacher preference signal L_T_tilde in [-1, 1].

    Case 1 (|G_t| >= 2): group-internal ranking of length-normalized teacher log-prob.
        - ``zscore``: (lp - mu_group) / (sigma_group + eps)         [over TEACHER lp]
        - ``minmax``: 2 * (lp - min) / (max - min + eps) - 1        in [-1, 1]
        - if ``use_margin`` and ``teacher_top2_logprob`` given: replace the ranking
          value with the teacher margin m_t = lp - max_{a'!=a} lp, then minmax.
    Case 2 (|G_t| == 1): tanh(gamma * Delta_norm), where Delta_norm is the
        batch-level z-score of (log pi_T - log pi_theta) computed OVER SINGLETON
        ROWS ONLY.  Raw Delta = log pi_T - log pi_theta is systematically negative
        (= -D_KL(pi_theta||pi_T) <= 0 in expectation), so normalisation is required
        to restore the correct positive/negative polarity for four-quadrant semantics.

    Args:
        teacher_log_prob: (bs, response_length) per-token teacher log-prob.
        old_log_probs: (bs, response_length) per-token student log-prob (behavior).
        response_mask: (bs, response_length).
        step_group_uids: (bs,) object array of anchor cluster ids.
        group_size_per_row: (bs,) |G_t| per row.
        normalization_mode: "minmax" (default) | "zscore" | "rank".
            ``rank``: ordinal rank mapped to [-1, 1]; immune to extreme log-prob
            values and near-zero spread -- recommended when log-probs are unstable.
        gamma: tanh scale for the Case-2 difference signal.
        eps: numerical guard.
        teacher_top2_logprob: (bs,) optional. For margin: the teacher log-prob of the
            best *other* token (the runner-up) at the step's last position. Only used
            when use_margin=True. Caller obtains it via sglang top_logprobs_num=2.
        use_margin: enable the margin variant of Case 1 (idea optional improvement).
        delta_norm_clip: stability guard for Case-2. The batch z-score makes the
            singleton signal DEPEND ON BATCH COMPOSITION: when a batch's singleton
            Deltas are tightly clustered, sigma_Delta is tiny and delta_norm blows up
            -> tanh saturates (all +-1, signal loses resolution); when they are
            spread out, the signal is compressed. If >0, clamp delta_norm to
            [-delta_norm_clip, +delta_norm_clip] BEFORE tanh, bounding the
            amplification. 0.0 = off (pure z-score). Recommended ~3.0.

    Returns:
        L_T_tilde: (bs,) in [-1, 1]. >0 teacher relatively approves, <0 disapproves.
    """
    bs = teacher_log_prob.shape[0]
    device = teacher_log_prob.device

    teacher_lp = length_normalize_logprob(teacher_log_prob, response_mask, eps)   # (bs,)
    student_lp = length_normalize_logprob(old_log_probs, response_mask, eps)      # (bs,)

    gsize = group_size_per_row.to(device=device, dtype=torch.long)
    is_group = (gsize >= 2)   # (bs,) bool, Case 1 mask

    # ---- Case 1: group-internal ranking over TEACHER log-prob ----
    ranking_value = teacher_lp
    if use_margin:
        assert teacher_top2_logprob is not None, \
            "use_margin=True requires teacher_top2_logprob (teacher runner-up log-prob)."
        # m_t = log pi_T(a_t) - max_{a'!=a_t} log pi_T(a'); length-normalized a_t,
        # runner-up is a per-step scalar (caller-chosen aggregation, e.g. last token).
        runner = teacher_top2_logprob.to(device=device, dtype=teacher_lp.dtype)
        ranking_value = teacher_lp - runner

    if normalization_mode == "rank":
        # Rank-based normalization: immune to outliers and near-zero spread.
        # Ordinal rank (ascending) mapped to [-1, 1]; singletons get 0.0.
        # Does NOT call _group_stats_per_row (ranks need the raw values).
        case1 = _group_rank_per_row(ranking_value, step_group_uids)
        # mean/std/mn/mx still needed for stats logging below
        mean, std, mn, mx = _group_stats_per_row(ranking_value, step_group_uids)
    else:
        mean, std, mn, mx = _group_stats_per_row(ranking_value, step_group_uids)
        if normalization_mode == "zscore":
            case1 = (ranking_value - mean) / (std + eps)
        elif normalization_mode == "minmax":
            denom = (mx - mn).clamp(min=eps)
            case1 = 2.0 * (ranking_value - mn) / denom - 1.0
        else:
            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    # ---- Case 2: singleton, teacher-student difference ----
    # Raw delta is E_{a~pi_theta}[log pi_T - log pi_theta] = -D_KL(pi_theta||pi_T) <= 0,
    # so delta is systematically negative (Jensen's inequality).  Applying tanh directly
    # would produce a signal that is almost always negative, breaking four-quadrant
    # semantics for singleton rows.
    #
    # Fix: batch-level z-score normalisation of delta OVER SINGLETON ROWS ONLY before
    # tanh compression.  After normalisation, ~50% of singleton rows have delta > mu_delta
    # (positive) and ~50% below (negative), restoring the correct signal polarity.
    # The normalised signal still means "teacher prefers this action MORE than average
    # for this batch" (positive) / "LESS than average" (negative), which is semantically
    # consistent with the group-ranking Case-1 signal.
    #
    # Reference: idea.md §"两类教师信号的分布校准".
    delta = teacher_lp - student_lp              # (bs,) >0 teacher prefers more (raw)

    # compute mean/std over singleton rows only
    singleton_mask = ~is_group                   # (bs,) bool
    mu_delta = delta.new_zeros(())
    std_delta = delta.new_ones(())   # sentinel; unused when there are no singletons
    if singleton_mask.any():
        singleton_deltas = delta[singleton_mask]
        mu_delta = singleton_deltas.mean()
        std_delta = singleton_deltas.std(unbiased=False).clamp(min=eps)
        delta_norm = (delta - mu_delta) / (std_delta + eps)
    else:
        # all rows belong to valid groups; Case-2 path will not be used
        delta_norm = delta  # unused; torch.where mask will select case1 for all rows

    # Stability guard (idea optional): batch z-score makes the singleton signal
    # depend on batch composition -- a tightly-clustered batch yields tiny
    # sigma_Delta and a blown-up delta_norm (tanh saturates, resolution lost).
    # Clamp before tanh to bound the amplification. Off when delta_norm_clip<=0.
    delta_norm_pre = delta_norm  # pre-clip form, kept for stats/logging
    if delta_norm_clip and delta_norm_clip > 0.0:
        delta_norm = delta_norm.clamp(min=-delta_norm_clip, max=delta_norm_clip)

    case2 = torch.tanh(gamma * delta_norm)

    # ---- unify ----
    l_tilde = torch.where(is_group, case1, case2)
    # keep within [-1, 1] (z-score can overshoot; minmax/case2 already bounded)
    l_tilde = l_tilde.clamp(min=-1.0, max=1.0)

    # ---- stats (side-channel; no-op effect on the returned tensor) ----
    if out_stats is not None:
        _fill_signal_stats(
            out_stats, out_hists,
            l_tilde=l_tilde, delta=delta, mu_delta=mu_delta, std_delta=std_delta,
            delta_norm_pre=delta_norm_pre, delta_norm_clip=delta_norm_clip,
            case1=case1, case2=case2, is_group=is_group, singleton_mask=singleton_mask,
            teacher_lp=teacher_lp, step_group_uids=step_group_uids, eps=eps,
        )
    return l_tilde


# --------------------------------------------------------------------------- #
# 3. Total advantage A_total.                                                 #
#    A_total = A_E + lambda * L_T_tilde * |A_E| + mu * 1_deg * L_T_tilde      #
# --------------------------------------------------------------------------- #
def compute_tgsa_advantage(
    episode_advantages_row: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    group_size_per_row: torch.Tensor,
    episode_group_std: torch.Tensor,
    response_length: int,
    lambda_: float = 0.3,
    mu: float = 0.1,
    gamma: float = 1.0,
    eps_deg: float = 0.01,
    normalization_mode: str = "minmax",
    replace_step_advantage: bool = True,
    step_advantages_row: Optional[torch.Tensor] = None,
    step_advantage_w: float = 0.0,
    bounded_env_scaling: str = "none",
    teacher_top2_logprob: Optional[torch.Tensor] = None,
    use_margin: bool = False,
    delta_norm_clip: float = 0.0,
    out_stats: Optional[dict] = None,
    out_hists: Optional[dict] = None,
):
    """Compose the TGSA total advantage tensor (bs, response_length).

    Args:
        episode_advantages_row: (bs,) A_E per row (z-score-normalized episode return).
        teacher_log_prob / old_log_probs / response_mask: see module docstring.
        step_group_uids / group_size_per_row: anchor grouping.
        episode_group_std: (bs,) sigma^R_group per row (episode/prompt group return std),
            used for the degeneration indicator 1_deg.
        response_length: L, for broadcasting the per-row scalar to (bs, L).
        lambda_: normal-group teacher modulation strength (idea 0.2-0.5).
        mu: degenerate-group fallback strength (idea ~0.1).
        gamma: tanh scale for Case-2 difference signal.
        eps_deg: degeneration threshold on sigma^R_group (idea ~0.01).
        normalization_mode: "minmax" | "zscore" for Case-1 ranking.
        replace_step_advantage: if True, drop GiGPO's A_S term (idea A_total has no A_S);
            if False, ADD step_advantage_w * A_S (for ablation).
        step_advantages_row: (bs,) optional A_S per row, used only if not replace.
        step_advantage_w: weight for the kept A_S term.
        bounded_env_scaling: "none" | "tanh" | "clip" applied to |A_E| in the lambda
            term (idea optional improvement: bounded env scaling).
        teacher_top2_logprob / use_margin: see compute_teacher_preference_signal.
        delta_norm_clip: Case-2 stability guard (see compute_teacher_preference_signal).
            Clamp delta_norm to [-c, c] before tanh to bound batch-composition
            amplification. 0.0 = off. Recommended ~3.0.

    Returns:
        A_total: (bs, response_length), same shape/contract as GiGPO ``scores``.
    """
    device = episode_advantages_row.device

    a_e = episode_advantages_row.to(device=device)                       # (bs,)
    l_tilde = compute_teacher_preference_signal(
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        step_group_uids=step_group_uids,
        group_size_per_row=group_size_per_row,
        normalization_mode=normalization_mode,
        gamma=gamma,
        teacher_top2_logprob=teacher_top2_logprob,
        use_margin=use_margin,
        delta_norm_clip=delta_norm_clip,
        out_stats=out_stats,
        out_hists=out_hists,
    )                                                                     # (bs,) in [-1,1]

    # degeneration indicator: 1_deg = 1[sigma^R_group <= eps_deg].
    # NOTE: distinct from the singleton case. 1_deg fires on MULTI-member groups
    # whose returns collapsed (all-success / all-fail -> A_E ~ 0). Singletons
    # (|G_t|==1) are already handled by Case 2 and must NOT be OR-ed into 1_deg,
    # because their std sentinel is 1.0 (> eps_deg) by construction.
    sigma_r = episode_group_std.to(device=device, dtype=a_e.dtype)
    gsize = group_size_per_row.to(device=device, dtype=torch.long)
    one_deg = ((sigma_r <= eps_deg) & (gsize >= 2)).to(a_e.dtype)         # (bs,)

    # bounded env scaling on |A_E| (optional)
    abs_ae = a_e.abs()
    if bounded_env_scaling == "tanh":
        scale = torch.tanh(abs_ae)
    elif bounded_env_scaling == "clip":
        scale = abs_ae.clamp(max=1.0)
    elif bounded_env_scaling == "none":
        scale = abs_ae
    else:
        raise ValueError(f"Unknown bounded_env_scaling: {bounded_env_scaling}")

    # A_total per row. The |A_E| (not A_E) in the lambda term is load-bearing for
    # the four-quadrant semantics (verified in test_tgsa.py).
    a_total_row = a_e + lambda_ * l_tilde * scale + mu * one_deg * l_tilde

    if not replace_step_advantage:
        assert step_advantages_row is not None, \
            "replace_step_advantage=False requires step_advantages_row."
        a_total_row = a_total_row + step_advantage_w * step_advantages_row.to(device=device)

    # broadcast (bs,) -> (bs, response_length), masked
    mask = response_mask.to(device=device, dtype=a_total_row.dtype)
    a_total = a_total_row.unsqueeze(-1).tile([1, response_length]) * mask

    # ---- stats (side-channel; no effect on the returned tensor) ----
    if out_stats is not None:
        _fill_advantage_stats(
            out_stats,
            a_e=a_e, abs_ae=abs_ae, scale=scale, l_tilde=l_tilde,
            one_deg=one_deg, gsize=gsize, a_total_row=a_total_row,
            lambda_=lambda_, mu=mu,
        )
    return a_total


def _fill_advantage_stats(out_stats: dict,
                          *,
                          a_e: torch.Tensor,
                          abs_ae: torch.Tensor,
                          scale: torch.Tensor,
                          l_tilde: torch.Tensor,
                          one_deg: torch.Tensor,
                          gsize: torch.Tensor,
                          a_total_row: torch.Tensor,
                          lambda_: float,
                          mu: float) -> None:
    """Populate ``tgsa_advantage/*`` and ``tgsa_coverage/*`` scalars in place.

    Pure side-channel: tensor->python-float reductions only.
    """
    n = a_e.numel()
    is_group = (gsize >= 2)
    # normal-group rows = multi-member AND non-degenerate (where the lambda
    # modulation is the live mechanism)
    normal_mask = is_group & (one_deg == 0)

    lambda_term = lambda_ * l_tilde * scale
    mu_term = mu * one_deg * l_tilde

    out_stats["tgsa_advantage/a_e_mean"] = float(a_e.mean().item())
    out_stats["tgsa_advantage/a_e_frac_pos"] = float((a_e > 0).to(torch.float32).mean().item())
    out_stats["tgsa_advantage/abs_ae_mean"] = float(abs_ae.mean().item())
    out_stats["tgsa_advantage/lambda_term_mean"] = float(lambda_term.mean().item())
    out_stats["tgsa_advantage/mu_term_mean"] = float(mu_term.mean().item())
    out_stats["tgsa_advantage/a_total_mean"] = float(a_total_row.mean().item())
    out_stats["tgsa_advantage/a_total_frac_pos"] = float((a_total_row > 0).to(torch.float32).mean().item())

    # load-bearing invariant: sign(A_total) == sign(A^E) on rows where A^E is
    # well-defined (|A^E|>eps). Non-zero here flags the four-quadrant |A^E| bug
    # or an oversized lambda. (idea four-quadrant semantics, TGSA_IMPLEMENTATION §3.2)
    signful = a_e.abs() > 1e-6
    if bool(signful.any()):
        viol = (torch.sign(a_total_row[signful]) != torch.sign(a_e[signful])).to(torch.float32).mean()
        out_stats["tgsa_advantage/sign_violation_frac"] = float(viol.item())
    else:
        out_stats["tgsa_advantage/sign_violation_frac"] = 0.0

    # four-quadrant populations (A^E sign x L_T sign)
    ae_pos = (a_e > 0)
    lt_pos = (l_tilde > 0)
    out_stats["tgsa_advantage/frac_q1_succ_approve"] = float(((ae_pos) & (lt_pos)).to(torch.float32).sum().item() / n)
    out_stats["tgsa_advantage/frac_q2_succ_disapprove"] = float(((ae_pos) & (~lt_pos)).to(torch.float32).sum().item() / n)
    out_stats["tgsa_advantage/frac_q3_fail_approve"] = float(((~ae_pos) & (lt_pos)).to(torch.float32).sum().item() / n)
    out_stats["tgsa_advantage/frac_q4_fail_disapprove"] = float(((~ae_pos) & (~lt_pos)).to(torch.float32).sum().item() / n)

    # per-row coverage (cross-check vs tgsa_group's per-group fractions: large
    # groups dilute these per-row numbers, so the two views together show the
    # true group structure)
    out_stats["tgsa_coverage/frac_singleton"] = float((~is_group).to(torch.float32).mean().item())
    out_stats["tgsa_coverage/frac_normal_group"] = float(normal_mask.to(torch.float32).mean().item())
    out_stats["tgsa_coverage/frac_degenerate"] = float(one_deg.mean().item())


# --------------------------------------------------------------------------- #
# 4. Reverse-KL per-token estimator (single-token Monte-Carlo, no top-k).     #
#    D_KL(pi_theta || pi_T) ~ log pi_theta(w) - log pi_T(w) at w ~ pi_theta.  #
#    Matches ROLL compute_approx_kl and verl kl_penalty 'kl'/'k3'/'mse'/'abs'. #
#    Used by the optional env-gated teacher-KL regularizer (idea L245-280).   #
# --------------------------------------------------------------------------- #
def compute_reverse_kl_token(old_log_probs: torch.Tensor,
                             teacher_log_prob: torch.Tensor,
                             response_mask: torch.Tensor,
                             kl_penalty: str = "k3",
                             eps: float = 1e-6) -> torch.Tensor:
    """Per-token reverse-KL contribution D_KL(pi_theta || pi_T).

    Single-sample Monte-Carlo estimator at the student-sampled tokens (Schulman k1/k3).
    No top-k, no vocab summation (the expectation is under pi_theta, which is exactly
    where the on-policy samples come from).

    Args:
        old_log_probs: (bs, response_length) log pi_theta (student behavior).
        teacher_log_prob: (bs, response_length) log pi_T at the same tokens.
        response_mask: (bs, response_length).
        kl_penalty: "kl" (k1, unbiased high-var) | "k3" (low-var, default, matches
            ROLL actor) | "mse" (k2) | "abs".
        eps: unused for k1/k3, kept for API symmetry.

    Returns:
        (bs, response_length) per-token KL contribution (NOT yet masked-meaned).
        Caller masks + aggregates as needed.
    """
    # log_ratio = log pi_theta - log pi_T  (= -Delta)
    log_ratio = old_log_probs - teacher_log_prob
    if kl_penalty == "kl":
        kl = log_ratio
    elif kl_penalty == "abs":
        kl = log_ratio.abs()
    elif kl_penalty == "mse":
        kl = 0.5 * log_ratio.square()
    elif kl_penalty == "k3":
        # Schulman k3: (exp(-log_ratio) ... ) form using teacher-student direction.
        # k3 = exp(log pi_T - log pi_theta) - (log pi_T - log pi_theta) - 1
        diff = teacher_log_prob - old_log_probs          # = -log_ratio
        kl = (torch.exp(diff) - diff - 1.0).clamp(min=-10.0, max=10.0)
    else:
        raise ValueError(f"Unknown kl_penalty: {kl_penalty}")
    return kl


def reverse_kl_scalar(old_log_probs: torch.Tensor,
                      teacher_log_prob: torch.Tensor,
                      response_mask: torch.Tensor,
                      kl_penalty: str = "k3") -> torch.Tensor:
    """Batch-level reverse-KL scalar = masked_mean of the per-token contribution.

    For the optional adaptive lambda_t = lambda_max * min(1, D_KL / c) (idea L137).
    """
    kl_tok = compute_reverse_kl_token(old_log_probs, teacher_log_prob, response_mask, kl_penalty)
    return masked_mean(kl_tok, response_mask.float(), axis=-1)            # (bs,)
