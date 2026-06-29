# Copyright 2025 TGSA-GRPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the TGSA side-channel tensorboard stats (gigpo/tgsa_stats.py) and
the out_stats/out_hists plumbing in gigpo/tgsa.py + gigpo/core_gigpo.py.

Focus:
  * the group-level statistics are PER-GROUP (de-duplicated), NOT per-row --
    a large group must not dilute the singleton/degenerate fractions;
  * the all-success vs all-fail split of degenerate episode groups (idea mu+/mu-);
  * the logger is disabled-by-default (no-op, no filesystem) and enable+flush
    writes scalars AND histograms on the global-step axis;
  * compute_tgsa_advantage populates the signal/advantage/coverage stat keys.
"""

import numpy as np
import pytest
import torch

import os
import sys
import types

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# core_gigpo does `from verl import DataProto` at import time, but only uses
# DataProto as a type hint. In a standalone test env verl's full dep chain
# (pandas, ...) may be absent, so stub verl.DataProto before importing.
if "verl" not in sys.modules:
    try:
        import verl  # noqa: F401
    except Exception:
        _stub = types.ModuleType("verl")
        _stub.DataProto = object
        sys.modules["verl"] = _stub

from gigpo.tgsa_stats import TGSAStatsLogger, compute_tgsa_group_stats
from gigpo.tgsa import compute_tgsa_advantage
from gigpo import core_gigpo


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_logger():
    """A fresh, DISABLED singleton -- isolated per test."""
    logger = TGSAStatsLogger._reset_for_test()
    yield logger
    logger.disable()
    TGSAStatsLogger._reset_for_test()


class _FakeWriter:
    """Captures add_scalar / add_histogram calls instead of writing files."""

    def __init__(self):
        self.scalars = []   # list of (tag, value, step)
        self.hists = []     # list of (tag, values, step)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def add_histogram(self, tag, values, step):
        self.hists.append((tag, values, int(step)))

    def flush(self):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# group stats: per-group vs per-row (the core correctness property)            #
# --------------------------------------------------------------------------- #
def _make_group_case_all_success_all_fail():
    """3 episode groups x 3 trajectories, 1 row per traj (L=1):

    group A: returns [1,1,1]  -> sigma=0, degenerate, ALL SUCCESS
    group B: returns [0,0,0]  -> sigma=0, degenerate, ALL FAIL
    group C: returns [1,0,.5] -> sigma>0, NOT degenerate
    """
    # returns per row
    rets = [1.0, 1.0, 1.0,   0.0, 0.0, 0.0,   1.0, 0.0, 0.5]
    bs = len(rets)
    rewards = torch.tensor(rets, dtype=torch.float32).unsqueeze(-1)   # (bs, 1)
    index = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    traj_index = [f't{i}' for i in range(bs)]
    # sigma_r per row, shared within group; degenerate groups -> 0, C -> >0
    sigma_r = torch.tensor(
        [0.0, 0.0, 0.0,   0.0, 0.0, 0.0,   0.47, 0.47, 0.47],
        dtype=torch.float32)
    return rewards, index, traj_index, sigma_r


def test_group_stats_per_group_not_per_row():
    """1 multi-member step group (5 rows) + 95 singletons = 100 rows, 96 GROUPS.

    Per-GROUP singleton fraction must be 95/96 (~0.99), NOT the per-row 95/100
    (0.95). The whole point of the per-group view is that it is NOT diluted by
    large groups.
    """
    rewards, index, traj_index, sigma_r = _make_group_case_all_success_all_fail()
    # overwrite step groups: 5 rows share 'g0', rest unique
    step_uids = ['g0'] * 5 + [f's{i}' for i in range(95)]
    # pad the episode case up to 100 rows
    bs = 100
    rewards = torch.zeros(bs, 1)
    index = ['A'] * bs
    traj_index = [f't{i}' for i in range(bs)]
    sigma_r = torch.ones(bs)  # non-degenerate; irrelevant to step-group stats

    gsize = torch.ones(bs, dtype=torch.long)
    gsize[:5] = 5  # the 5 shared-anchor rows

    scalars, hists = compute_tgsa_group_stats(
        rewards, index, traj_index, step_uids,
        eps_deg=0.01, success_thresh=0.0, sigma_r=sigma_r, gsize=gsize)

    assert scalars["gigpo_group/n_step_groups"] == 96.0
    assert scalars["gigpo_group/step_frac_singleton"] == pytest.approx(95 / 96)
    assert scalars["gigpo_group/step_frac_multimember"] == pytest.approx(1 / 96)
    assert "gigpo_group/step_group_size_hist" in hists
    assert hists["gigpo_group/step_group_size_hist"].shape[0] == 96


def test_group_stats_all_success_all_fail_split():
    """Degenerate episode groups split into all-success vs all-fail (idea mu+/mu-).

    Groups A (all 1.0) and B (all 0.0) are both degenerate (sigma=0); A is
    all-success, B is all-fail. Their fractions must each be 1/2, and the
    degenerate-return mean must be 0.5 (the bimodal midpoint), exposing both
    modes rather than collapsing them.
    """
    rewards, index, traj_index, sigma_r = _make_group_case_all_success_all_fail()
    step_uids = [f's{i}' for i in range(rewards.shape[0])]  # all singleton steps
    gsize = torch.ones(rewards.shape[0], dtype=torch.long)

    scalars, hists = compute_tgsa_group_stats(
        rewards, index, traj_index, step_uids,
        eps_deg=0.01, success_thresh=0.0, sigma_r=sigma_r, gsize=gsize)

    assert scalars["gigpo_group/n_episode_groups"] == 3.0
    assert scalars["gigpo_group/episode_group_size_mean"] == pytest.approx(3.0)
    # A and B are degenerate (sigma<=eps & size>=2); C is not
    assert scalars["gigpo_group/frac_degenerate"] == pytest.approx(2 / 3)
    # of the 2 degenerate groups, 1 all-success, 1 all-fail
    assert scalars["gigpo_group/frac_degenerate_all_success"] == pytest.approx(0.5)
    assert scalars["gigpo_group/frac_degenerate_all_fail"] == pytest.approx(0.5)
    assert scalars["gigpo_group/degenerate_return_mean"] == pytest.approx(0.5)
    assert "gigpo_group/degenerate_return_hist" in hists
    # the bimodal histogram: one group at 1.0, one at 0.0
    deg = hists["gigpo_group/degenerate_return_hist"]
    assert set(np.round(deg, 4).tolist()) == {0.0, 1.0}


def test_group_stats_singleton_episode_groups_not_degenerate():
    """A singleton episode group (1 trajectory) gets sigma sentinel 1.0 (>eps),
    so it must NOT be counted as degenerate even though its 'std' is trivially 0.
    Mirrors the gsize>=2 conjunct in tgsa.py one_deg."""
    rewards = torch.tensor([[1.0], [0.0]])  # 2 rows, 2 distinct singleton groups
    index = ['A', 'B']
    traj_index = ['t0', 't1']
    sigma_r = torch.tensor([1.0, 1.0])  # sentinel for singleton episode groups
    step_uids = ['s0', 's1']
    gsize = torch.ones(2, dtype=torch.long)

    scalars, _ = compute_tgsa_group_stats(
        rewards, index, traj_index, step_uids,
        eps_deg=0.01, success_thresh=0.0, sigma_r=sigma_r, gsize=gsize)

    assert scalars["gigpo_group/n_episode_groups"] == 2.0
    assert scalars["gigpo_group/frac_degenerate"] == 0.0  # neither is degenerate


# --------------------------------------------------------------------------- #
# logger lifecycle: disabled no-op, enable+flush mechanics                    #
# --------------------------------------------------------------------------- #
def test_logger_disabled_by_default_is_noop(fresh_logger):
    assert fresh_logger.enabled is False
    # record + flush must not raise and must not touch the filesystem
    fresh_logger.record_scalar("gigpo_group/x", 1.0)
    fresh_logger.record_histogram("tgsa_signal/h", np.array([1.0, 2.0]))
    fresh_logger.flush(step=0)
    assert fresh_logger.enabled is False


def test_logger_enable_flush_writes_scalars_and_histograms(fresh_logger):
    fake = _FakeWriter()
    fresh_logger.enable(writer=fake)   # inject; no tensorboard needed
    assert fresh_logger.enabled is True

    fresh_logger.record_scalars({
        "gigpo_group/n_episode_groups": 3.0,
        "tgsa_advantage/a_total_mean": 0.42,
    })
    fresh_logger.record_histogram("tgsa_signal/delta_norm_hist",
                                  np.array([0.1, -0.2, 3.5], dtype=np.float32))
    fresh_logger.flush(step=7)

    tags = {t for (t, _, _) in fake.scalars}
    assert "gigpo_group/n_episode_groups" in tags
    assert "tgsa_advantage/a_total_mean" in tags
    # all scalars stamped with the global step
    assert all(s == 7 for (_, _, s) in fake.scalars)
    # histogram recorded too
    assert len(fake.hists) == 1
    assert fake.hists[0][0] == "tgsa_signal/delta_norm_hist"
    assert fake.hists[0][2] == 7


def test_logger_flush_empty_is_noop(fresh_logger):
    """flush with nothing buffered (e.g. non-GiGPO estimator step) writes nothing."""
    fake = _FakeWriter()
    fresh_logger.enable(writer=fake)
    fresh_logger.flush(step=5)
    assert fake.scalars == []
    assert fake.hists == []


def test_logger_record_accepts_tensor_and_list(fresh_logger):
    """record_histogram must coerce torch tensors and python lists alike."""
    fake = _FakeWriter()
    fresh_logger.enable(writer=fake)
    fresh_logger.record_histogram("h1", torch.tensor([1.0, 2.0, 3.0]))
    fresh_logger.record_histogram("h2", [0.5, 1.5])
    fresh_logger.flush(step=1)
    assert len(fake.hists) == 2


def test_logger_enable_degrades_gracefully_without_tensorboard(fresh_logger, monkeypatch):
    """If tensorboard is unavailable, enable() must NOT raise -- it warns and
    stays disabled so a wandb-only run never crashes (production safety)."""
    import builtins
    real_import = builtins.__import__

    def _fail_tensorboard(name, *args, **kwargs):
        if name == "torch.utils.tensorboard" or name == "tensorboard":
            raise ModuleNotFoundError("No module named 'tensorboard'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_tensorboard)
    fresh_logger.enable("/tmp/tgsa_stats_noprod")  # no writer injected
    assert fresh_logger.enabled is False
    # record + flush must be silent no-ops
    fresh_logger.record_scalar("gigpo_group/x", 1.0)
    fresh_logger.flush(step=0)


# --------------------------------------------------------------------------- #
# compute_tgsa_advantage populates out_stats / out_hists                       #
# --------------------------------------------------------------------------- #
def _make_tgsa_inputs(n=8, L=4, seed=0):
    torch.manual_seed(seed)
    response_mask = torch.ones(n, L)
    teacher_log_prob = torch.randn(n, L) * 0.5 - 1.0      # teacher logprobs
    old_log_probs = torch.randn(n, L) * 0.5 - 0.8         # student (behavior)
    episode_advantages_row = torch.randn(n) * 0.3          # A^E
    # all-singleton step groups -> Case2 path active
    step_group_uids = np.array([f's{i}' for i in range(n)], dtype=object)
    group_size_per_row = torch.ones(n, dtype=torch.long)
    episode_group_std = torch.ones(n) * 0.5                # non-degenerate
    return (teacher_log_prob, old_log_probs, response_mask, step_group_uids,
            group_size_per_row, episode_advantages_row, episode_group_std)


def test_compute_tgsa_advantage_populates_signal_and_advantage_stats():
    (tlp, olp, mask, uids, gsize, a_e, sigma_r) = _make_tgsa_inputs()
    out_stats = {}
    out_hists = {}
    compute_tgsa_advantage(
        episode_advantages_row=a_e,
        teacher_log_prob=tlp,
        old_log_probs=olp,
        response_mask=mask,
        step_group_uids=uids,
        group_size_per_row=gsize,
        episode_group_std=sigma_r,
        response_length=4,
        out_stats=out_stats,
        out_hists=out_hists,
    )
    # signal group
    for k in ["tgsa_signal/l_tilde_mean", "tgsa_signal/delta_mean",
              "tgsa_signal/std_delta", "tgsa_signal/delta_norm_mean",
              "tgsa_signal/delta_norm_max", "tgsa_signal/delta_norm_frac_gt3",
              "tgsa_signal/clip_active_frac", "tgsa_signal/case2_mean",
              "tgsa_signal/case1_teacher_spread_mean"]:
        assert k in out_stats, f"missing {k}"
    # advantage + coverage groups
    for k in ["tgsa_advantage/a_e_mean", "tgsa_advantage/abs_ae_mean",
              "tgsa_advantage/lambda_term_mean", "tgsa_advantage/mu_term_mean",
              "tgsa_advantage/a_total_mean", "tgsa_advantage/sign_violation_frac",
              "tgsa_advantage/frac_q1_succ_approve",
              "tgsa_advantage/frac_q2_succ_disapprove",
              "tgsa_advantage/frac_q3_fail_approve",
              "tgsa_advantage/frac_q4_fail_disapprove",
              "tgsa_coverage/frac_singleton", "tgsa_coverage/frac_normal_group",
              "tgsa_coverage/frac_degenerate"]:
        assert k in out_stats, f"missing {k}"
    # histograms
    assert "tgsa_signal/delta_norm_hist" in out_hists
    assert "tgsa_signal/l_tilde_hist" in out_hists
    # four quadrants sum to ~1
    qsum = (out_stats["tgsa_advantage/frac_q1_succ_approve"]
            + out_stats["tgsa_advantage/frac_q2_succ_disapprove"]
            + out_stats["tgsa_advantage/frac_q3_fail_approve"]
            + out_stats["tgsa_advantage/frac_q4_fail_disapprove"])
    assert qsum == pytest.approx(1.0)
    # all-singleton batch -> frac_singleton == 1
    assert out_stats["tgsa_coverage/frac_singleton"] == pytest.approx(1.0)


def test_compute_tgsa_advantage_out_stats_default_none_is_backward_compat():
    """out_stats=None must keep the original behavior (no stats, same tensor)."""
    (tlp, olp, mask, uids, gsize, a_e, sigma_r) = _make_tgsa_inputs()
    a_no_stats = compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=tlp, old_log_probs=olp,
        response_mask=mask, step_group_uids=uids, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=4)
    assert a_no_stats.shape == (8, 4)


def test_sign_violation_frac_is_zero_for_default_lambda():
    """With the default lambda (0.3) and |A^E|, sign(A_total)==sign(A^E) on
    well-defined rows -> the invariant violation fraction must be 0. (The
    load-bearing |A^E| four-quadrant property, TGSA_IMPLEMENTATION §3.2.)"""
    (tlp, olp, mask, uids, gsize, a_e, sigma_r) = _make_tgsa_inputs()
    out_stats = {}
    compute_tgsa_advantage(
        episode_advantages_row=a_e, teacher_log_prob=tlp, old_log_probs=olp,
        response_mask=mask, step_group_uids=uids, group_size_per_row=gsize,
        episode_group_std=sigma_r, response_length=4, out_stats=out_stats)
    assert out_stats["tgsa_advantage/sign_violation_frac"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# end-to-end through core_gigpo: group stats flow to the (disabled) logger    #
# --------------------------------------------------------------------------- #
def test_core_gigpo_tgsa_records_group_stats_when_logger_enabled(monkeypatch, fresh_logger):
    """With the logger enabled, compute_gigpo_outcome_advantage(tgsa on) must
    buffer the per-GROUP stats (not crash, not depend on verl)."""
    fresh_logger.enable(writer=_FakeWriter())  # inject; no tensorboard needed
    fake = fresh_logger._writer

    n, L = 9, 1
    # 3 episode groups x 3 trajectories; A/B degenerate (all-same return),
    # C varied. Step groups: first 3 share an anchor, rest singleton.
    rewards = torch.tensor([[1.0], [1.0], [1.0],
                            [0.0], [0.0], [0.0],
                            [1.0], [0.0], [0.5]], dtype=torch.float32)
    step_rewards = rewards.clone()
    response_mask = torch.ones(n, L)
    anchor_obs = np.array([f'a{i}' for i in range(n)], dtype=object)
    index = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
    traj_index = np.array([f't{i}' for i in range(n)])
    step_uids_proxy = None  # built internally by build_step_group; not asserted here

    tgsa_config = {"enabled": True, "lambda": 0.3, "mu": 0.1, "eps_deg": 0.01,
                   "normalization_mode": "minmax", "replace_step_advantage": True,
                   "bounded_env_scaling": "none", "delta_norm_clip": 0.0,
                   "success_thresh": 0.0}
    teacher_log_prob = torch.randn(n, L) * 0.5 - 1.0
    old_log_probs = torch.randn(n, L) * 0.5 - 0.8

    # build_step_group may need a real anchor grouping; force all-singleton by
    # making anchor_obs all-distinct so the TGSA path runs without similarity.
    a_total, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=rewards, step_rewards=step_rewards,
        response_mask=response_mask, anchor_obs=anchor_obs,
        index=index, traj_index=traj_index,
        log_group_stats=True,
        teacher_log_prob=teacher_log_prob, old_log_probs=old_log_probs,
        tgsa_config=tgsa_config)

    fresh_logger.flush(step=0)
    tags = {t for (t, _, _) in fake.scalars}
    # group structure was recorded
    assert "gigpo_group/n_episode_groups" in tags
    assert "gigpo_group/frac_degenerate" in tags
    # signal + advantage + cfg echo also flowed through
    assert any(t.startswith("tgsa_signal/") for t in tags)
    assert any(t.startswith("tgsa_advantage/") for t in tags)
    assert "tgsa_cfg/lambda" in tags
    # the returned advantage tensor still has the right shape/contract
    assert a_total.shape == (n, L)


# --------------------------------------------------------------------------- #
# Decoupling: plain GiGPO (no TGSA) also records group stats                  #
# --------------------------------------------------------------------------- #
def test_sigma_r_none_degenerate_detection_matches_explicit():
    """Plain GiGPO passes sigma_r=None; the group std is then computed
    internally from id2returns. The degenerate detection must be NUMERICALLY
    identical to passing the explicit sigma_r (same returns => same std)."""
    rewards, index, traj_index, sigma_r = _make_group_case_all_success_all_fail()
    step_uids = [f's{i}' for i in range(rewards.shape[0])]
    gsize = torch.ones(rewards.shape[0], dtype=torch.long)

    s_explicit, _ = compute_tgsa_group_stats(
        rewards, index, traj_index, step_uids,
        eps_deg=0.01, success_thresh=0.0, sigma_r=sigma_r, gsize=gsize)
    s_internal, _ = compute_tgsa_group_stats(
        rewards, index, traj_index, step_uids,
        eps_deg=0.01, success_thresh=0.0, sigma_r=None, gsize=gsize)

    # same degenerate count, same all-success/all-fail split, same return mean
    assert s_explicit["gigpo_group/frac_degenerate"] == s_internal["gigpo_group/frac_degenerate"]
    assert s_explicit["gigpo_group/frac_degenerate_all_success"] == \
        s_internal["gigpo_group/frac_degenerate_all_success"]
    assert s_explicit["gigpo_group/degenerate_return_mean"] == \
        pytest.approx(s_internal["gigpo_group/degenerate_return_mean"])


def test_plain_gigpo_records_group_stats_without_tgsa(fresh_logger):
    """No tgsa_config (TGSA off), no teacher -- plain GiGPO with
    log_group_stats=True must still buffer the gigpo_group/ panel. This is the
    core decoupling: group structure is diagnostic for GiGPO itself (A^S signal
    depends on anchor overlap), not just for TGSA."""
    fresh_logger.enable(writer=_FakeWriter())
    fake = fresh_logger._writer

    n, L = 9, 1
    rewards = torch.tensor([[1.0], [1.0], [1.0],
                            [0.0], [0.0], [0.0],
                            [1.0], [0.0], [0.5]], dtype=torch.float32)
    step_rewards = rewards.clone()
    response_mask = torch.ones(n, L)
    anchor_obs = np.array([f'a{i}' for i in range(n)], dtype=object)
    index = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
    traj_index = np.array([f't{i}' for i in range(n)])

    scores, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=rewards, step_rewards=step_rewards,
        response_mask=response_mask, anchor_obs=anchor_obs,
        index=index, traj_index=traj_index,
        log_group_stats=True)
    fresh_logger.flush(step=0)

    tags = {t for (t, _, _) in fake.scalars}
    # the group panel IS recorded (plain GiGPO, no teacher)
    assert "gigpo_group/n_episode_groups" in tags
    assert "gigpo_group/frac_degenerate" in tags
    assert "gigpo_group/frac_degenerate_all_success" in tags
    assert "gigpo_group/n_step_groups" in tags
    # but NO teacher-signal / advantage-decomposition / tgsa_cfg keys (those need
    # TGSA + a teacher)
    assert not any(t.startswith("tgsa_signal/") for t in tags)
    assert not any(t.startswith("tgsa_advantage/") for t in tags)
    assert not any(t.startswith("tgsa_cfg/") for t in tags)
    # advantage tensor still returned (shape is GiGPO's own broadcast behavior,
    # not our concern here -- we only assert it is a tensor and non-empty)
    assert torch.is_tensor(scores) and scores.numel() > 0


def test_log_group_stats_false_records_nothing(fresh_logger):
    """log_group_stats=False (and no TGSA) -> no group stats buffered. This is
    the backward-compat / opt-out path: a run that wants the original GiGPO
    behavior sees no side-channel stats."""
    fresh_logger.enable(writer=_FakeWriter())
    fake = fresh_logger._writer

    n, L = 6, 1
    rewards = torch.ones(n, L)
    step_rewards = rewards.clone()
    response_mask = torch.ones(n, L)
    anchor_obs = np.array([f'a{i}' for i in range(n)], dtype=object)
    index = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    traj_index = np.array([f't{i}' for i in range(n)])

    core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=rewards, step_rewards=step_rewards,
        response_mask=response_mask, anchor_obs=anchor_obs,
        index=index, traj_index=traj_index,
        log_group_stats=False)
    fresh_logger.flush(step=0)

    assert fake.scalars == []
    assert fake.hists == []


def test_compute_group_stats_backward_compat_alias():
    """compute_tgsa_group_stats is kept as an alias of compute_group_stats so
    any external reference still resolves (decoupling did not break the name)."""
    from gigpo.tgsa_stats import compute_group_stats, compute_tgsa_group_stats
    assert compute_tgsa_group_stats is compute_group_stats

