# Copyright 2025 TGSA-GRPO contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Unit tests for gigpo/tgsa.py — TGSA-GRPO algorithm semantics.

These tests pin the four-quadrant guarantee (idea.md "语义验证"), the degenerate-group
fallback, the singleton-vs-group case split, length normalization, and the reverse-KL
single-token estimator. They are pure-Python/torch and do NOT require Ray/sglang/GPU.
"""

import math
import os
import sys

import numpy as np
import torch

# Allow running from repo root without installing the package.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gigpo.tgsa import (  # noqa: E402
    compute_reverse_kl_token,
    compute_teacher_preference_signal,
    compute_tgsa_advantage,
    length_normalize_logprob,
    reverse_kl_scalar,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _mask_row(L, start, end):
    """A (L,) mask with ones in [start, end)."""
    m = torch.zeros(L)
    m[start:end] = 1.0
    return m


def _make_batch(teacher_lp_rows, student_lp_rows, masks, uids):
    """Stack per-row tensors into (bs, L) batch tensors.

    teacher_lp_rows / student_lp_rows: list of (L,) tensors (per-token log-probs).
    masks: list of (L,) tensors.
    uids: list of group ids (one per row).
    """
    T = torch.stack(teacher_lp_rows, dim=0)
    S = torch.stack(student_lp_rows, dim=0)
    M = torch.stack(masks, dim=0)
    uids_arr = np.array(uids, dtype=object)
    gsize = torch.tensor([uids.count(u) for u in uids], dtype=torch.long)
    return T, S, M, uids_arr, gsize


# --------------------------------------------------------------------------- #
# 1. length normalization                                                      #
# --------------------------------------------------------------------------- #
def test_length_normalize_logprob():
    # two rows, different action spans
    lp = torch.tensor([[-1.0, -2.0, 0.0, 0.0],   # span [0,2) -> mean -1.5
                       [0.0, 0.0, -3.0, -5.0]])  # span [2,4) -> mean -4.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 1.0]])
    out = length_normalize_logprob(lp, mask)
    assert torch.allclose(out, torch.tensor([-1.5, -4.0]), atol=1e-6)
    assert out.shape == (2,)


# --------------------------------------------------------------------------- #
# 2. Four-quadrant semantics (idea "语义验证")                                #
#    Use Case 1 (|G_t|>=2) so L_T_tilde is set by group ranking of teacher lp.#
#    Build a 2-row group where row0 has higher teacher lp -> L_T>0 (approve),  #
#    row1 lower -> L_T<0 (disapprove). Then set A_E signs to exercise all 4.   #
# --------------------------------------------------------------------------- #
def _two_row_group(teacher_lps, student_lps):
    """Two rows sharing a group uid; teacher_lps/student_lps are length-norm scalars.
    Returns L_T_tilde (2,) with row0>0, row1<0 by construction (minmax)."""
    L = 4
    # encode each scalar as a constant over a 2-token span (length-norm preserves it)
    t_rows = [torch.full((L,), float(t)) for t in teacher_lps]
    s_rows = [torch.full((L,), float(s)) for s in student_lps]
    masks = [_mask_row(L, 0, 2), _mask_row(L, 0, 2)]
    uids = ["g", "g"]
    T, S, M, uids_arr, gsize = _make_batch(t_rows, s_rows, masks, uids)
    l_tilde = compute_teacher_preference_signal(
        teacher_log_prob=T, old_log_probs=S, response_mask=M,
        step_group_uids=uids_arr, group_size_per_row=gsize,
        normalization_mode="minmax", gamma=1.0,
    )
    return l_tilde, T, S, M, uids_arr, gsize


def test_case1_ranking_signs():
    # row0 teacher prefers more (higher lp) -> L_T > 0; row1 -> L_T < 0
    l_tilde, *_ = _two_row_group(teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    # minmax: row0 -> +1, row1 -> -1
    assert l_tilde[0].item() > 0
    assert l_tilde[1].item() < 0
    assert torch.allclose(l_tilde, torch.tensor([1.0, -1.0]), atol=1e-6)


def test_four_quadrant_success_approve_strong_reward():
    # A_E > 0, L_T > 0  -> A_total > A_E (strong reward)
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([+1.0, +1.0])   # success for both rows (only row0 tested)
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # row0: A_E=1, L_T=1, |A_E|=1 -> 1 + 0.3*1*1 = 1.3 broadcast over 2 tokens
    per_row0 = a_total[0, :2].mean()
    assert per_row0.item() == pytest_approx(1.3, abs=1e-6)
    assert per_row0.item() > 1.0  # stronger than A_E


def test_four_quadrant_success_disapprove_weak_reward():
    # A_E > 0, L_T < 0  -> A_total < A_E (weak reward)
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([+1.0, +1.0])
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    per_row1 = a_total[1, :2].mean()  # row1: A_E=1, L_T=-1 -> 1 - 0.3 = 0.7
    assert per_row1.item() == pytest_approx(0.7, abs=1e-6)
    assert per_row1.item() < 1.0  # weaker than A_E


def test_four_quadrant_fail_approve_weak_penalty():
    # A_E < 0, L_T > 0  -> A_total closer to 0 than A_E (weak penalty)
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([-1.0, -1.0])
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # row0: A_E=-1, L_T=+1, |A_E|=1 -> -1 + 0.3*1*1 = -0.7  (less negative -> weak penalty)
    per_row0 = a_total[0, :2].mean()
    assert per_row0.item() == pytest_approx(-0.7, abs=1e-6)
    assert per_row0.item() > -1.0  # weaker penalty than A_E


def test_four_quadrant_fail_disapprove_strong_penalty():
    # A_E < 0, L_T < 0  -> A_total more negative than A_E (strong penalty)
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([-1.0, -1.0])
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # row1: A_E=-1, L_T=-1, |A_E|=1 -> -1 + 0.3*(-1)*1 = -1.3 (more negative)
    per_row1 = a_total[1, :2].mean()
    assert per_row1.item() == pytest_approx(-1.3, abs=1e-6)
    assert per_row1.item() < -1.0  # stronger penalty than A_E


def test_sign_preservation_normal_group():
    """In a normal (non-degenerate) group, A_total keeps A_E's sign (env veto)."""
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([+0.5, -0.5])
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    r0 = a_total[0, :2].mean().item()
    r1 = a_total[1, :2].mean().item()
    assert r0 > 0, "success row must stay positive"
    assert r1 < 0, "fail row must stay negative"


# --------------------------------------------------------------------------- #
# 3. Degenerate-group fallback (idea "退化组行为")                            #
#    sigma^R_group <= eps AND |G_t|>=2  -> 1_deg=1, with A_E~0 the lambda term #
#    vanishes and A_total ~ mu * L_T_tilde (weak, correctly-directed).         #
# --------------------------------------------------------------------------- #
def test_degenerate_group_fallback():
    # 2-row group, all-success (equal returns -> sigma_R ~ 0), A_E ~ 0
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    # l_tilde = [+1, -1] (row0 approve, row1 disapprove)
    a_e = torch.tensor([0.0, 0.0])           # A_E ~ 0 (degenerate)
    sigma_r = torch.tensor([0.0, 0.0])       # group return std ~ 0 -> degenerate
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # A_total = 0 + 0.3*L_T*0 + 0.1*1*L_T = 0.1*L_T
    r0 = a_total[0, :2].mean().item()
    r1 = a_total[1, :2].mean().item()
    assert r0 == pytest_approx(0.1, abs=1e-6)    # +mu (approve -> positive)
    assert r1 == pytest_approx(-0.1, abs=1e-6)   # -mu (disapprove -> negative)


def test_nondegenerate_group_mu_term_off():
    """Normal group (sigma_R > eps): the mu fallback must NOT fire (one_deg=0)."""
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([0.0, 0.0])
    sigma_r = torch.tensor([5.0, 5.0])       # large std -> NOT degenerate
    L = T.shape[1]
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # A_E=0, |A_E|=0, one_deg=0 -> A_total = 0 (no fallback when not degenerate)
    assert torch.allclose(a_total[:, :2].mean(dim=-1), torch.zeros(2), atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. Singleton (|G_t|==1) uses Case 2: tanh(gamma * batch-normalised Delta).   #
#    Raw Delta = log pi_T - log pi_theta is systematically NEGATIVE under      #
#    a ~ pi_theta (E[Delta] = -D_KL(pi_theta||pi_T) <= 0, Jensen), so the raw  #
#    signal would be almost always negative and break four-quadrant semantics. #
#    Fix (idea "两类教师信号的分布校准"): batch-level z-score of Delta over    #
#    SINGLETON rows only, then tanh. These tests pin the bias correction.     #
# --------------------------------------------------------------------------- #
def test_singleton_batch_norm_centers_systematic_bias():
    """All singleton rows have NEGATIVE raw Delta (systematic teacher bias), but
    after batch z-score ~half become positive. This is the whole point of the
    normalisation: it restores correct polarity instead of 'teacher disapproves
    of everything'."""
    L = 4
    # student lp constant -1; teacher lp varies -> raw Delta all in [-2, -0.5] (<0)
    T = torch.tensor([[-3.0], [-2.5], [-2.0], [-1.5]]).expand(4, L).clone()
    S = torch.full((4, L), -1.0)
    M = torch.ones(4, L)
    uids = np.array(["a", "b", "c", "d"], dtype=object)  # 4 distinct singletons
    gsize = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    l_tilde = compute_teacher_preference_signal(
        teacher_log_prob=T, old_log_probs=S, response_mask=M,
        step_group_uids=uids, group_size_per_row=gsize,
        normalization_mode="minmax", gamma=1.0,
    )
    # raw Delta = [-2, -1.5, -1, -0.5]; mu=-1.25, std~0.559
    # delta_norm ~ [-1.342, -0.447, 0.447, 1.342]; tanh -> [-0.87, -0.42, 0.42, 0.87]
    assert l_tilde[0].item() < 0   # most-negative raw Delta -> still negative
    assert l_tilde[3].item() > 0   # least-negative raw Delta -> flipped to positive
    n_pos = int((l_tilde > 0).sum().item())
    assert n_pos == 2, f"expected ~half positive after centering, got {n_pos}/4"
    # symmetric: the two middle rows are mirror images
    assert l_tilde[1].item() == pytest_approx(-l_tilde[2].item(), abs=1e-5)
    # all within [-1, 1]
    assert l_tilde.abs().max().item() <= 1.0 + 1e-6


def test_singleton_batch_norm_ordering_preserved():
    """Higher teacher lp -> higher (less negative) Delta -> higher L_T_tilde.
    The ranking among singletons is preserved by the monotonic tanh(z-score)."""
    L = 4
    T = torch.tensor([[-4.0], [-3.0], [-2.0], [-1.0]]).expand(4, L).clone()
    S = torch.full((4, L), -1.5)
    M = torch.ones(4, L)
    uids = np.array(["a", "b", "c", "d"], dtype=object)
    gsize = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    l_tilde = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0)
    vals = l_tilde.tolist()
    assert vals == sorted(vals), "L_T_tilde must be monotonic in teacher lp"


def test_singleton_single_row_zero_fallback():
    """Edge case: a batch with only ONE singleton row. batch std=0 -> delta_norm=0
    -> tanh(0)=0 -> the teacher modulation term vanishes and the row falls back to
    pure A^E (env veto preserved). This is a safe degeneration, NOT a bug: with no
    other singletons to compare against, a relative signal is undefined."""
    L = 4
    T = torch.full((1, L), -1.0)
    S = torch.full((1, L), -2.0)
    M = _mask_row(L, 0, 2).unsqueeze(0)
    uids = np.array(["solo"], dtype=object)
    gsize = torch.tensor([1], dtype=torch.long)
    l_tilde = compute_teacher_preference_signal(
        teacher_log_prob=T, old_log_probs=S, response_mask=M,
        step_group_uids=uids, group_size_per_row=gsize,
        normalization_mode="minmax", gamma=1.0,
    )
    assert l_tilde[0].item() == pytest_approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# 4b. delta_norm_clip: stability guard for Case-2 batch z-score.              #
#    A tight singleton cluster -> tiny sigma_Delta -> blown-up delta_norm for  #
#    an outlier -> tanh saturates near +-1 (resolution lost). Clamping before  #
#    tanh bounds the amplification. Off by default (delta_norm_clip=0).       #
# --------------------------------------------------------------------------- #
def _tight_cluster_with_outlier(n_cluster=8, L=4):
    """8 tightly-clustered singletons (delta=-0.5) + 1 distant outlier (delta=-5).
    The cluster dominates sigma_Delta, so the outlier's z-score blows up >3."""
    T = torch.tensor([[-2.5]] * n_cluster + [[-7.0]]).expand(n_cluster + 1, L).clone()
    S = torch.full((n_cluster + 1, L), -2.0)  # delta = [-0.5]*8 + [-5.0]
    M = torch.ones(n_cluster + 1, L)
    uids = np.array([str(i) for i in range(n_cluster + 1)], dtype=object)
    gsize = torch.ones(n_cluster + 1, dtype=torch.long)
    return T, S, M, uids, gsize


def test_delta_norm_clip_off_saturates_on_outlier():
    """Without clipping, the outlier's L_T_tilde saturates near -1."""
    T, S, M, uids, gsize = _tight_cluster_with_outlier()
    lt = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0,
                                           delta_norm_clip=0.0)
    # outlier (last row) z-score > 3 -> tanh ~ -0.99 (saturated)
    assert lt[-1].item() < -0.95, f"expected saturation, got {lt[-1].item()}"


def test_delta_norm_clip_bounds_signal():
    """With clip=1.0, the outlier is clamped to tanh(-1)=-0.7616 (resolution kept)."""
    T, S, M, uids, gsize = _tight_cluster_with_outlier()
    lt = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0,
                                           delta_norm_clip=1.0)
    bound = float(torch.tanh(torch.tensor(1.0)))
    assert lt[-1].item() == pytest_approx(-bound, abs=1e-5)
    # all values stay within [-tanh(clip), tanh(clip)] for the singleton rows
    assert lt.abs().max().item() <= bound + 1e-6


def test_delta_norm_clip_does_not_flip_polarity():
    """Clamping only bounds magnitude; the sign (outlier is teacher-disapproved) stays."""
    T, S, M, uids, gsize = _tight_cluster_with_outlier()
    lt_off = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0, delta_norm_clip=0.0)
    lt_on = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0, delta_norm_clip=1.0)
    # every row keeps its sign
    for i in range(len(lt_off)):
        assert (lt_off[i].item() > 0) == (lt_on[i].item() > 0), f"polarity flipped at row {i}"


def test_delta_norm_clip_zero_is_noop():
    """delta_norm_clip=0 (default) must equal not passing it at all."""
    T, S, M, uids, gsize = _tight_cluster_with_outlier()
    lt_default = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0)
    lt_zero = compute_teacher_preference_signal(T, S, M, uids, gsize, "minmax", 1.0,
                                                delta_norm_clip=0.0)
    assert torch.allclose(lt_default, lt_zero, atol=1e-7)


def test_delta_norm_clip_only_affects_singletons():
    """Clip must not change Case-1 (group) rows, only Case-2 (singleton) rows."""
    L = 4
    # 2-row group (Case1) + a cluster of singletons with an outlier (Case2)
    T_g = torch.tensor([[-1.0], [-3.0]]).expand(2, L).clone()
    S_g = torch.full((2, L), -2.0)
    M_g = torch.ones(2, L)
    uids_g = np.array(["g", "g"], dtype=object)
    gsize_g = torch.tensor([2, 2], dtype=torch.long)
    lt_g_off = compute_teacher_preference_signal(T_g, S_g, M_g, uids_g, gsize_g, "minmax", 1.0)
    lt_g_on = compute_teacher_preference_signal(T_g, S_g, M_g, uids_g, gsize_g, "minmax", 1.0,
                                                delta_norm_clip=1.0)
    # Case-1 rows are identical with or without clip (clip only touches delta_norm)
    assert torch.allclose(lt_g_off, lt_g_on, atol=1e-7)
    """A singleton has std sentinel 1.0 (>eps); 1_deg must be False even if A_E~0,
    and Case 2 supplies the signal. The mu term must NOT fire."""
    L = 4
    T = torch.full((1, L), -1.0)
    S = torch.full((1, L), -2.0)
    M = _mask_row(L, 0, 2).unsqueeze(0)
    uids = np.array(["solo"], dtype=object)
    gsize = torch.tensor([1], dtype=torch.long)
    a_e = torch.tensor([0.0])
    sigma_r = torch.tensor([1.0])  # singleton sentinel
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # A_E=0, |A_E|=0 -> lambda term 0; one_deg = (1.0<=0.01)&(1>=2) = False -> mu 0.
    # So A_total must be 0 (Case-2 signal only affects the lambda/mu terms, which are
    # both gated off here). This confirms singletons are NOT bailed out by mu.
    assert torch.allclose(a_total[0, :2].mean(), torch.tensor(0.0), atol=1e-6)


def test_singleton_not_treated_as_degenerate():
    """A singleton has std sentinel 1.0 (>eps); 1_deg must be False even if A_E~0,
    and Case 2 supplies the signal. The mu term must NOT fire."""
    L = 4
    T = torch.full((1, L), -1.0)
    S = torch.full((1, L), -2.0)
    M = _mask_row(L, 0, 2).unsqueeze(0)
    uids = np.array(["solo"], dtype=object)
    gsize = torch.tensor([1], dtype=torch.long)
    a_e = torch.tensor([0.0])
    sigma_r = torch.tensor([1.0])  # singleton sentinel
    a_total = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    # A_E=0, |A_E|=0 -> lambda term 0; one_deg = (1.0<=0.01)&(1>=2) = False -> mu 0.
    # So A_total must be 0 (Case-2 signal only affects the lambda/mu terms, which are
    # both gated off here). This confirms singletons are NOT bailed out by mu.
    assert torch.allclose(a_total[0, :2].mean(), torch.tensor(0.0), atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. replace_step_advantage: keeping A_S adds it (ablation path).             #
# --------------------------------------------------------------------------- #
def test_keep_step_advantage_adds_term():
    l_tilde, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    a_e = torch.tensor([1.0, 1.0])
    L = T.shape[1]
    a_s = torch.tensor([0.5, -0.5])
    base = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=True,
    )
    with_as = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=T, old_log_probs=S,
        response_mask=M, step_group_uids=uids_arr, group_size_per_row=gsize,
        episode_group_std=torch.tensor([5.0, 5.0]), response_length=L,
        lambda_=0.3, mu=0.1, eps_deg=0.01, replace_step_advantage=False,
        step_advantages_row=a_s, step_advantage_w=0.5,
    )
    diff = (with_as - base)[:, :2].mean(dim=-1)
    assert torch.allclose(diff, torch.tensor([0.25, -0.25]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. Reverse-KL single-token estimator.                                       #
# --------------------------------------------------------------------------- #
def test_reverse_kl_k1_sign():
    # log pi_theta - log pi_T > 0 -> student assigns higher prob than teacher -> kl>0
    L = 4
    T = torch.full((1, L), -2.0)
    S = torch.full((1, L), -1.0)   # student prefers more
    M = _mask_row(L, 0, 2).unsqueeze(0)
    kl = compute_reverse_kl_token(S, T, M, kl_penalty="kl")
    # log_ratio = -1 - (-2) = +1 everywhere (function does NOT apply mask by design;
    # the caller masks via reverse_kl_scalar / masked_mean).
    assert kl[0, 0].item() == pytest_approx(1.0, abs=1e-6)
    assert kl[0, 2].item() == pytest_approx(1.0, abs=1e-6)  # unmasked by design
    # the SCALAR form applies the mask -> only the 2 in-mask tokens count
    scalar = reverse_kl_scalar(S, T, M, kl_penalty="kl")
    assert scalar[0].item() == pytest_approx(1.0, abs=1e-6)


def test_reverse_kl_k3_nonneg_for_close_distributions():
    # k3 estimator is non-negative when distributions are close (it's a bound).
    L = 4
    T = torch.full((1, L), -1.5)
    S = torch.full((1, L), -1.4)
    M = _mask_row(L, 0, 2).unsqueeze(0)
    kl = compute_reverse_kl_token(S, T, M, kl_penalty="k3")
    assert kl[0, 0].item() >= -1e-6  # ~0 and non-negative for tiny gap
    # scalar
    scalar = reverse_kl_scalar(S, T, M, kl_penalty="k3")
    assert scalar[0].item() >= -1e-6


def test_reverse_kl_no_topk_no_vocab_sum():
    """Smoke: KL only needs the single sampled-token log-probs; output shape is (bs,L),
    no vocab dimension, no top-k aggregation."""
    bs, L = 3, 5
    T = torch.randn(bs, L)
    S = torch.randn(bs, L)
    M = torch.ones(bs, L)
    kl = compute_reverse_kl_token(S, T, M, kl_penalty="kl")
    assert kl.shape == (bs, L)  # per-token, no vocab axis


# --------------------------------------------------------------------------- #
# 7. margin variant (optional)                                                #
# --------------------------------------------------------------------------- #
def test_margin_variant_changes_case1():
    """With use_margin, Case-1 ranking value = teacher_lp - runner_up."""
    l_tilde_plain, T, S, M, uids_arr, gsize = _two_row_group(
        teacher_lps=[-1.0, -3.0], student_lps=[-2.0, -2.0])
    # runner-up teacher log-prob per row (e.g. -1.5 for both)
    top2 = torch.tensor([-1.5, -1.5])
    l_tilde_margin = compute_teacher_preference_signal(
        teacher_log_prob=T, old_log_probs=S, response_mask=M,
        step_group_uids=uids_arr, group_size_per_row=gsize,
        normalization_mode="minmax", gamma=1.0,
        teacher_top2_logprob=top2, use_margin=True,
    )
    # margin values: row0 = -1 - (-1.5) = 0.5 ; row1 = -3 - (-1.5) = -1.5
    # minmax -> row0=+1, row1=-1 (same signs as plain ranking, different magnitudes pre-norm)
    assert l_tilde_margin[0].item() > 0
    assert l_tilde_margin[1].item() < 0


# --------------------------------------------------------------------------- #
# small pytest-free approx helper so the file is runnable standalone           #
# --------------------------------------------------------------------------- #
def pytest_approx(expected, abs=1e-6):
    # NOTE: param is named `abs` only to match call sites; inside __eq__ we use
    # math.isclose (NOT builtins.abs) because a param named `abs` would shadow it
    # inside the closure and break the comparison (caught as TypeError -> False).
    class _A:
        def __init__(self, e, tol):
            self.e, self.tol = e, tol

        def __eq__(self, other):
            try:
                return math.isclose(float(other), float(self.e), abs_tol=self.tol)
            except Exception:
                return False

        def __repr__(self):
            return f"approx({self.e}, abs={self.tol})"
    return _A(expected, abs)


if __name__ == "__main__":
    # Run all test_* functions; print PASS/FAIL.
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npassed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            npassed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{npassed}/{len(fns)} passed")
    sys.exit(0 if npassed == len(fns) else 1)
