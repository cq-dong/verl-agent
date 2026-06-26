# Copyright 2025 TGSA-GRPO contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Integration test: gigpo/core_gigpo.py TGSA wiring.

Verifies:
  (1) Backward compatibility: calling compute_gigpo_outcome_advantage WITHOUT
      teacher/tgsa args reproduces the original GiGPO joint advantage
      (A^E + step_advantage_w * A^S).
  (2) Opt-in: with tgsa_config.enabled=True the joint combination is REPLACED
      by compute_tgsa_advantage, and the result equals a direct call with the
      same |G_t| / sigma^R_group derived inline.
  (3) |G_t| and sigma^R_group helpers produce the expected per-row values.

Pure-Python/torch, no Ray/sglang/GPU needed.
"""

import math
import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# core_gigpo does `from verl import DataProto` at import time, but only uses
# DataProto as a type hint (never at runtime in the functions exercised here).
# In a standalone test env verl's full dep chain (pandas, ...) may be absent,
# so stub verl.DataProto before importing core_gigpo.
if "verl" not in sys.modules:
    try:
        import verl  # noqa: F401
    except Exception:
        import types
        _stub = types.ModuleType("verl")
        _stub.DataProto = object
        sys.modules["verl"] = _stub

from gigpo import core_gigpo  # noqa: E402
from gigpo.tgsa import compute_tgsa_advantage  # noqa: E402


def _approx(expected, tol=1e-6):
    class _A:
        def __init__(self, e, t):
            self.e, self.t = e, t

        def __eq__(self, other):
            try:
                return math.isclose(float(other), float(self.e), abs_tol=self.t)
            except Exception:
                return False

        def __repr__(self):
            return f"approx({self.e})"
    return _A(expected, tol)


def _toy_batch():
    """A 2-row batch forming ONE episode group (same index) and ONE step group
    (same anchor obs), so GiGPO's A^E and A^S are both well-defined.
    Row0 success (reward +1), row1 fail (reward 0) -> non-degenerate episode group.

    Returns the raw pieces needed by compute_gigpo_outcome_advantage plus
    teacher/student per-token log-probs aligned to the response_mask.
    """
    bs, L = 2, 6
    response_mask = torch.zeros(bs, L)
    response_mask[0, 1:3] = 1.0   # row0 action span
    response_mask[1, 1:3] = 1.0   # row1 action span

    # token-level rewards: row0 gets +1 (success) at the action span, row1 gets 0
    token_level_rewards = torch.zeros(bs, L)
    token_level_rewards[0, 1:3] = 1.0
    token_level_rewards[1, 1:3] = 0.0

    # step_rewards (discounted returns per step) — for step_norm_reward grouping
    step_rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)

    anchor_obs = np.array(["obs_A", "obs_A"])  # identical -> one step group |G_t|=2
    index = np.array(["prompt_0", "prompt_0"])  # one episode group
    traj_index = np.array(["traj_0", "traj_1"])

    # teacher per-token log-probs: row0 higher (teacher prefers) -> Case1 L_T>0
    teacher_log_prob = torch.zeros(bs, L)
    teacher_log_prob[0, 1:3] = -1.0
    teacher_log_prob[1, 1:3] = -3.0
    old_log_probs = torch.zeros(bs, L)
    old_log_probs[0, 1:3] = -2.0
    old_log_probs[1, 1:3] = -2.0

    return dict(
        token_level_rewards=token_level_rewards, step_rewards=step_rewards,
        response_mask=response_mask, anchor_obs=anchor_obs, index=index,
        traj_index=traj_index, teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs, L=L,
    )


# --------------------------------------------------------------------------- #
# 1. helpers                                                                  #
# --------------------------------------------------------------------------- #
def test_step_group_size_per_row():
    uids = np.array(["g", "g", "h", "k", "k", "k"], dtype=object)
    g = core_gigpo._step_group_size_per_row(uids)
    assert g.tolist() == [2, 2, 1, 3, 3, 3]


def test_episode_group_return_std_degenerate():
    # one episode group, two trajectories BOTH return 1 -> std 0 (degenerate)
    token_level_rewards = torch.zeros(2, 4)
    token_level_rewards[0, 1] = 1.0   # traj_0 total return 1
    token_level_rewards[1, 1] = 1.0   # traj_1 total return 1
    index = np.array(["p", "p"])
    traj_index = np.array(["t0", "t1"])
    s = core_gigpo._episode_group_return_std(token_level_rewards, index, traj_index)
    assert s[0].item() == _approx(0.0)
    assert s[1].item() == _approx(0.0)


def test_episode_group_return_std_nondegenerate():
    # one episode group, returns 1 and 0 -> std 0.5
    token_level_rewards = torch.zeros(2, 4)
    token_level_rewards[0, 1] = 1.0
    token_level_rewards[1, 1] = 0.0
    index = np.array(["p", "p"])
    traj_index = np.array(["t0", "t1"])
    s = core_gigpo._episode_group_return_std(token_level_rewards, index, traj_index)
    assert s[0].item() == _approx(0.5)
    assert s[1].item() == _approx(0.5)


def test_episode_group_return_std_singleton_sentinel():
    token_level_rewards = torch.zeros(1, 4)
    token_level_rewards[0, 1] = 5.0
    index = np.array(["p"])
    traj_index = np.array(["t0"])
    s = core_gigpo._episode_group_return_std(token_level_rewards, index, traj_index)
    assert s[0].item() == _approx(1.0)  # sentinel -> 1_deg False


# --------------------------------------------------------------------------- #
# 2. backward compatibility: no teacher args -> original GiGPO                #
# --------------------------------------------------------------------------- #
def test_backward_compat_no_tgsa():
    b = _toy_batch()
    adv, ret = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"],
        step_rewards=b["step_rewards"],
        response_mask=b["response_mask"],
        anchor_obs=b["anchor_obs"],
        index=b["index"],
        traj_index=b["traj_index"],
        step_advantage_w=1.0,
        mode="mean_norm",
        enable_similarity=False,
    )
    # must match the original combination exactly: A^E + 1.0 * A^S
    # (recompute the pieces the same way core_gigpo does internally)
    remove_std = True
    ep = core_gigpo.episode_norm_reward(b["token_level_rewards"], b["response_mask"],
                                        b["index"], b["traj_index"], 1e-6, remove_std)
    uids = core_gigpo.build_step_group(b["anchor_obs"], b["index"], False, 0.95)
    st = core_gigpo.step_norm_reward(b["step_rewards"], b["response_mask"], uids, 1e-6, remove_std)
    expected = ep + 1.0 * st
    assert adv.shape == expected.shape
    assert torch.allclose(adv, expected, atol=1e-6)
    # teacher args omitted entirely -> still works (None defaults)
    adv2, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
    )
    assert torch.allclose(adv2, expected, atol=1e-6)


def test_tgsa_disabled_config_falls_back():
    """tgsa_config present but enabled=False -> original behavior."""
    b = _toy_batch()
    adv, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
        teacher_log_prob=b["teacher_log_prob"], old_log_probs=b["old_log_probs"],
        tgsa_config={"enabled": False},
    )
    adv_orig, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
    )
    assert torch.allclose(adv, adv_orig, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. TGSA path matches a direct compute_tgsa_advantage call                   #
# --------------------------------------------------------------------------- #
def test_tgsa_path_matches_direct_call():
    b = _toy_batch()
    cfg = dict(enabled=True, lambda_=0.3, mu=0.1, gamma=1.0, eps_deg=0.01,
               normalization_mode="minmax", replace_step_advantage=True,
               bounded_env_scaling="none", use_margin=False)
    adv, ret = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
        teacher_log_prob=b["teacher_log_prob"], old_log_probs=b["old_log_probs"],
        tgsa_config=cfg,
    )
    # rebuild the exact inputs compute_tgsa_advantage would receive
    uids = core_gigpo.build_step_group(b["anchor_obs"], b["index"], False, 0.95)
    gsize = core_gigpo._step_group_size_per_row(uids)
    sigma_r = core_gigpo._episode_group_return_std(b["token_level_rewards"], b["index"], b["traj_index"])
    mask_f = b["response_mask"].float()
    denom = mask_f.sum(-1).clamp(min=1.0)
    ep = core_gigpo.episode_norm_reward(b["token_level_rewards"], b["response_mask"],
                                        b["index"], b["traj_index"], 1e-6, True)
    ep_row = (ep * mask_f).sum(-1) / denom
    direct = compute_tgsa_advantage(
        episode_advantages_row=ep_row, teacher_log_prob=b["teacher_log_prob"],
        old_log_probs=b["old_log_probs"], response_mask=b["response_mask"],
        step_group_uids=uids, group_size_per_row=gsize, episode_group_std=sigma_r,
        response_length=b["L"], lambda_=0.3, mu=0.1, gamma=1.0, eps_deg=0.01,
        normalization_mode="minmax", replace_step_advantage=True,
        bounded_env_scaling="none", use_margin=False,
    )
    assert adv.shape == direct.shape
    assert torch.allclose(adv, direct, atol=1e-6)
    assert ret is adv  # TGSA returns (a_total, a_total)


def test_tgsa_path_rejects_missing_teacher():
    b = _toy_batch()
    cfg = dict(enabled=True)
    raised = False
    try:
        core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
            response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
            index=b["index"], traj_index=b["traj_index"],
            tgsa_config=cfg,
        )
    except ValueError:
        raised = True
    assert raised


def test_tgsa_path_keeps_step_advantage_ablation():
    """replace_step_advantage=False adds step_advantage_w * A^S on top."""
    b = _toy_batch()
    cfg_off = dict(enabled=True, lambda_=0.3, mu=0.1, replace_step_advantage=True)
    cfg_on = dict(enabled=True, lambda_=0.3, mu=0.1, replace_step_advantage=False)
    a_off, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
        teacher_log_prob=b["teacher_log_prob"], old_log_probs=b["old_log_probs"],
        tgsa_config=cfg_off, step_advantage_w=0.5,
    )
    a_on, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=b["token_level_rewards"], step_rewards=b["step_rewards"],
        response_mask=b["response_mask"], anchor_obs=b["anchor_obs"],
        index=b["index"], traj_index=b["traj_index"],
        teacher_log_prob=b["teacher_log_prob"], old_log_probs=b["old_log_probs"],
        tgsa_config=cfg_on, step_advantage_w=0.5,
    )
    diff = (a_on - a_off)
    # the difference should be exactly step_advantage_w * A^S where A^S != 0
    assert not torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)


if __name__ == "__main__":
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
