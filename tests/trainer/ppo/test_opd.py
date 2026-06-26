# Copyright 2025 TGSA-GRPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""纯 OPD 消融基线的单元测试(gigpo/opd.py)。

校验纯 OPD advantage 估计器忠实复刻 ROLL 标准 OPD:
  * advantage = -coef * KL(pi_theta || pi_T),逐 token,乘 response_mask;
  * KL 估计 k3 与 ROLL compute_approx_kl 的 k3 逐字一致;
  * 无环境门控 —— 成功+失败轨迹都被无条件蒸馏(对比 TGSA test_hard_gate_skips_fail_rows,
    这是 OPD 相对 TGSA 的核心缺陷);
  * 向后兼容(OPD 关时不影响既有路径)。

纯 Python/torch,无 Ray/sglang/GPU。
"""

import os
import sys
import types

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# gigpo.tgsa 在 import 时尝试 from verl.utils.torch_functional import masked_mean;
# 独立测试环境可能缺 verl 全量依赖(pandas...),stub 掉 verl。
if "verl" not in sys.modules:
    try:
        import verl  # noqa: F401
    except Exception:
        _stub = types.ModuleType("verl")
        _stub.DataProto = object
        sys.modules["verl"] = _stub

from gigpo.opd import compute_opd_advantage
from gigpo.tgsa import compute_reverse_kl_token


def _roll_k3_manual(old_log_probs, teacher_log_prob):
    """手算 ROLL compute_approx_kl 的 k3,作为对照真值。
    k3 = (exp(teacher - student) - (teacher - student) - 1).clamp(-10, 10)
    (roll/utils/functionals.py:181, kl_penalty="k3" 分支)
    """
    diff = teacher_log_prob - old_log_probs  # teacher - student
    kld = (torch.exp(diff) - diff - 1.0)
    return kld.clamp(min=-10.0, max=10.0)


def test_opd_advantage_equals_neg_coef_kl():
    """adv = -coef * masked_kl,逐 token,数值精确。"""
    torch.manual_seed(0)
    bs, L = 4, 6
    old_lp = torch.randn(bs, L)
    tea_lp = torch.randn(bs, L)
    mask = torch.ones(bs, L)
    mask[0, 3:] = 0  # 第一行后 3 个 token 不在动作 span

    coef = 1.5
    adv, ret = compute_opd_advantage(old_lp, tea_lp, mask, L, kl_coef=coef, kl_penalty="k3")

    # 期望:adv = -coef * k3 * mask
    k3 = _roll_k3_manual(old_lp, tea_lp)
    expected = -(coef * k3) * mask
    assert torch.allclose(adv, expected, atol=1e-6)
    # returns == advantages(纯 OPD 无 critic)
    assert torch.allclose(ret, adv)
    # mask 外的 token advantage 必须为 0
    assert (adv[0, 3:] == 0).all()


def test_opd_kl_matches_roll_k3():
    """compute_opd_advantage 内部 KL 与手算 ROLL k3 逐字一致(coef=1)。"""
    torch.manual_seed(1)
    bs, L = 5, 8
    old_lp = torch.randn(bs, L)
    tea_lp = torch.randn(bs, L)
    mask = torch.ones(bs, L)

    adv, _ = compute_opd_advantage(old_lp, tea_lp, mask, L, kl_coef=1.0, kl_penalty="k3")
    # adv = -k3 * mask => -adv = k3(mask 外为 0)
    neg_adv = -adv
    expected_k3 = _roll_k3_manual(old_lp, tea_lp) * mask
    assert torch.allclose(neg_adv, expected_k3, atol=1e-6)

    # 与 tgsa.compute_reverse_kl_token 的 k3 也一致(它就是复用源)
    kl_tok = compute_reverse_kl_token(old_lp, tea_lp, mask, "k3")
    assert torch.allclose(neg_adv, kl_tok * mask, atol=1e-6)


def test_opd_no_gate_distills_all_trajectories():
    """无环境门控:成功轨迹(回报高)与失败轨迹(回报低)都被无条件蒸馏。

    这是 OPD 相对 TGSA 的核心缺陷 —— TGSA 门控 1[A^E>0] 只蒸成功轨迹(见
    test_tgsa_kl_regularizer.py::test_hard_gate_skips_fail_rows),而纯 OPD 对所有
    轨迹都施加 -coef·KL 拉向教师(失败轨迹也被拉向教师 = 强化错误行为)。
    本测试确认纯 OPD 路径不存在任何门控:不同"成功/失败"的轨迹 KL 项都非零生效。
    """
    torch.manual_seed(2)
    L = 4
    # 两条轨迹:一条"成功"(old_lp 接近 teacher)、一条"失败"(old_lp 远离 teacher)。
    # 纯 OPD 不看回报,只看 KL;两条都会被蒸馏(KL 越大拉力越大,但都非零)。
    old_lp = torch.tensor([[0.0, 0.0, 0.0, 0.0],      # 轨迹 A(任意"成功"标注)
                           [-3.0, -3.0, -3.0, -3.0]])  # 轨迹 B(任意"失败"标注)
    tea_lp = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1]])
    mask = torch.ones(2, L)

    adv, _ = compute_opd_advantage(old_lp, tea_lp, mask, L, kl_coef=1.0, kl_penalty="k3")

    # 两条轨迹的 advantage 都非零(都被蒸馏);无门控把任何一条置零。
    assert adv[0].abs().sum() > 0, "轨迹 A 必须被蒸馏(无门控)"
    assert adv[1].abs().sum() > 0, "轨迹 B 必须被蒸馏(无门控)"
    # 轨迹 B(KL 大)的 |adv| 应大于轨迹 A(KL 小)—— KL 越大蒸馏拉力越强
    assert adv[1].abs().sum() > adv[0].abs().sum()
    # 关键:两条都是负 advantage(KL>=0,adv=-coef·KL<=0),都在"拉近教师"方向
    assert (adv <= 0).all()


def test_opd_out_stats_populated():
    """out_stats 填充 opd/ 面板量。"""
    torch.manual_seed(3)
    bs, L = 3, 5
    old_lp = torch.randn(bs, L)
    tea_lp = torch.randn(bs, L)
    mask = torch.ones(bs, L)
    out_stats = {}
    compute_opd_advantage(old_lp, tea_lp, mask, L, kl_coef=2.0, kl_penalty="k3",
                          out_stats=out_stats)
    assert "opd/kl_advantage_mean" in out_stats
    assert "opd/kl_coef" in out_stats
    assert out_stats["opd/kl_coef"] == 2.0
    # advantage 均 <= 0,故 mean <= 0
    assert out_stats["opd/kl_advantage_mean"] <= 0


def test_opd_backward_compat_no_teacher_args():
    """纯 OPD 与 GiGPO/TGSA 路径互不干扰:不传 teacher 时 GiGPO 仍可正常调
    (此处只确认 compute_opd_advantage 自身签名稳定,不改既有行为)。"""
    # compute_opd_advantage 是独立函数,默认 kl_coef=1.0/kl_penalty="k3",
    # 不依赖任何全局态;既有 GiGPO 调用 compute_gigpo_outcome_advantage 不受影响。
    bs, L = 2, 3
    old_lp = torch.zeros(bs, L)
    tea_lp = torch.zeros(bs, L)
    mask = torch.ones(bs, L)
    adv, ret = compute_opd_advantage(old_lp, tea_lp, mask, L)  # 全默认
    # student==teacher => KL=0 => adv=0
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)
