# Copyright 2025 TGSA-GRPO contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Unit tests for the TGSA env-gated teacher-KL regularizer math.

These mirror EXACTLY what verl/workers/actor/dp_actor.py computes in the
``use_tgsa_kl`` branch (which cannot be imported standalone because it needs the
full verl/Ray/pandas stack). The regularizer is:

    L_kl = beta_T * mean_row( gate * masked_mean(reverse_kl_token, mask, -1) )
    gate = 1[A^E_row > 0]              (hard, default; sign-exact for A^E)
         = sigmoid(eta * A^E_row)      (soft)

We verify: hard gate activates only on success rows; soft gate is sigmoid;
coef=0 disables; the KL uses the current policy log-prob (not old); and the
gate uses the per-row advantage (broadcast within the action span).
"""

import math
import os
import sys

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gigpo.tgsa import compute_reverse_kl_token  # noqa: E402

try:
    from verl.utils.torch_functional import masked_mean
except Exception:  # pragma: no cover
    def masked_mean(values, mask, axis=None):
        return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


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


def _gated_kl_loss(log_prob, teacher_log_prob, advantages, response_mask,
                   coef, kl_penalty="k3", gate_mode="hard", eta=1.0):
    """Replica of dp_actor's use_tgsa_kl block."""
    kl_tok = compute_reverse_kl_token(log_prob, teacher_log_prob, response_mask, kl_penalty)
    kl_per_row = masked_mean(kl_tok, response_mask.float(), axis=-1)            # (bs,)
    adv_per_row = masked_mean(advantages, response_mask.float(), axis=-1)       # (bs,)
    if gate_mode == "hard":
        gate = (adv_per_row > 0).to(kl_per_row.dtype)
    else:
        gate = torch.sigmoid(eta * adv_per_row)
    return (gate * kl_per_row).mean() * coef, gate


# --------------------------------------------------------------------------- #
# 1. hard gate: only success rows contribute                                  #
# --------------------------------------------------------------------------- #
def test_hard_gate_skips_fail_rows():
    # two rows: row0 success (adv>0), row1 fail (adv<0).
    # teacher KL is positive on both (student diverges from teacher), but only
    # row0 should contribute under the hard gate.
    bs, L = 2, 4
    mask = torch.ones(bs, L)
    log_prob = torch.zeros(bs, L)       # current policy
    teacher = torch.full((bs, L), -1.0)  # teacher prefers less -> reverse KL > 0
    adv = torch.tensor([[+1.0, +1.0, +1.0, +1.0],
                        [-1.0, -1.0, -1.0, -1.0]])
    loss, gate = _gated_kl_loss(log_prob, teacher, adv, mask, coef=1.0,
                                kl_penalty="kl", gate_mode="hard")
    # kl_per_row = log_prob - teacher = 0 - (-1) = 1 for both rows
    # gate = [1, 0]; loss = mean([1*1, 0*1]) = 0.5
    assert gate.tolist() == [1.0, 0.0]
    assert loss.item() == _approx(0.5)


def test_hard_gate_all_fail_zero_loss():
    bs, L = 2, 4
    mask = torch.ones(bs, L)
    log_prob = torch.zeros(bs, L)
    teacher = torch.full((bs, L), -1.0)
    adv = torch.full((bs, L), -1.0)  # both fail
    loss, gate = _gated_kl_loss(log_prob, teacher, adv, mask, coef=2.0,
                                kl_penalty="kl", gate_mode="hard")
    assert gate.tolist() == [0.0, 0.0]
    assert loss.item() == _approx(0.0)


# --------------------------------------------------------------------------- #
# 2. soft gate is sigmoid(eta * adv)                                          #
# --------------------------------------------------------------------------- #
def test_soft_gate_is_sigmoid():
    bs, L = 1, 4
    mask = torch.ones(bs, L)
    log_prob = torch.zeros(bs, L)
    teacher = torch.full((bs, L), -1.0)
    adv = torch.full((bs, L), 2.0)
    loss, gate = _gated_kl_loss(log_prob, teacher, adv, mask, coef=1.0,
                                kl_penalty="kl", gate_mode="soft", eta=0.5)
    expected_gate = float(torch.sigmoid(torch.tensor(0.5 * 2.0)))
    assert gate[0].item() == _approx(expected_gate)
    # kl_per_row = 1; loss = gate * 1
    assert loss.item() == _approx(expected_gate)


# --------------------------------------------------------------------------- #
# 3. coef=0 disables (default off)                                            #
# --------------------------------------------------------------------------- #
def test_coef_zero_disables():
    bs, L = 1, 4
    mask = torch.ones(bs, L)
    log_prob = torch.zeros(bs, L)
    teacher = torch.full((bs, L), -1.0)
    adv = torch.full((bs, L), 1.0)
    loss, _ = _gated_kl_loss(log_prob, teacher, adv, mask, coef=0.0,
                             kl_penalty="kl", gate_mode="hard")
    assert loss.item() == _approx(0.0)


# --------------------------------------------------------------------------- #
# 4. KL uses CURRENT policy log-prob (varies with log_prob), not a stale one  #
# --------------------------------------------------------------------------- #
def test_kl_responds_to_current_logprob():
    # doubling the divergence (log_prob further from teacher) increases the KL
    bs, L = 1, 4
    mask = torch.ones(bs, L)
    teacher = torch.full((bs, L), -1.0)
    adv = torch.full((bs, L), 1.0)
    lp_close = torch.full((bs, L), -0.5)   # closer to teacher (-1)
    lp_far = torch.full((bs, L), 0.5)      # further from teacher
    loss_close, _ = _gated_kl_loss(lp_close, teacher, adv, mask, coef=1.0,
                                   kl_penalty="kl", gate_mode="hard")
    loss_far, _ = _gated_kl_loss(lp_far, teacher, adv, mask, coef=1.0,
                                 kl_penalty="kl", gate_mode="hard")
    # kl = log_prob - teacher; |lp_far - (-1)|=1.5 > |lp_close-(-1)|=0.5
    assert loss_far.item() > loss_close.item()


# --------------------------------------------------------------------------- #
# 5. per-row advantage is broadcast within the action span (masked mean)      #
# --------------------------------------------------------------------------- #
def test_advantage_broadcast_within_span():
    # advantage is constant within the mask; gate uses that per-row value.
    bs, L = 1, 4
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # only first 2 tokens are action
    log_prob = torch.zeros(bs, L)
    teacher = torch.full((bs, L), -1.0)
    # advantage: +1 on action span, 0 elsewhere -> masked mean over mask = +1
    adv = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss, gate = _gated_kl_loss(log_prob, teacher, adv, mask, coef=1.0,
                                kl_penalty="kl", gate_mode="hard")
    assert gate[0].item() == _approx(1.0)  # adv_per_row = 1 > 0
    # kl_per_row = masked_mean(log_prob - teacher, mask) = 1
    assert loss.item() == _approx(1.0)


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
