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

"""纯 OPD(Online Policy Distillation)优势估计器 —— ROLL 标准实现的忠实复刻。

本模块用于 TGSA-GRPO 的消融实验:作为"纯 OPD 基线",与 TGSA 完整方法对比,
论证 OPD 无条件蒸馏(无环境门控)相对 TGSA 门控蒸馏的缺陷。

== ROLL 标准实现(已核实)==
  * KL 估计:roll/utils/functionals.py:181 compute_approx_kl,kl_penalty="k3"
    k3 = (exp(teacher - student) - (teacher - student) - 1).clamp(-10, 10)
    这是 reverse KL  D(pi_theta || pi_T) 的单 token Monte-Carlo 估计。
  * 纯 OPD advantage:roll/utils/functionals.py:1149
    advantages = -total_weighted_kld          # adv = -coef * KL,KL 折叠进优势
  * 强制配置:base_config.py:888-901
    gamma=0, adv_estimator="reinforce", critic_warmup=0, use_kl_loss=False
    (无 critic、无 GAE、KL 不作为独立 loss 项)
  * 环境门控:无 —— 无条件蒸馏所有轨迹(成功 + 失败轨迹都拉向教师)。
    这正是 OPD 相对 TGSA 的核心缺陷(TGSA 门控 1[A^E>0] 只蒸成功轨迹)。

== 与 TGSA 的关系 ==
纯 OPD 是 TGSA 的"退化特例":TGSA 已有教师前向(gigpo/teacher_client.py 的
sglang HTTP)+ KL 估计(gigpo/tgsa.py 的 compute_reverse_kl_token,其 k3 分支
与 ROLL compute_approx_kl 的 k3 逐字一致)。纯 OPD 只需把 advantage 换成
-coef*KL,去掉环境奖励/门控/critic。本模块零新基础设施,KL 估计直接复用
compute_reverse_kl_token。

== 采样说明 ==
纯 OPD advantage 逐轨迹独立,不依赖组内对比(与 GiGPO/TGSA 不同)。因此
rollout.n=1 合法(n 只是采样数,batch 增广用),实验时可在 run 脚本自行调整。
"""

from typing import Optional

import torch

# 复用 TGSA 的 reverse-KL 估计器:其 k3 分支与 ROLL compute_approx_kl 的 k3
# 逐字一致(由 tests/trainer/ppo/test_tgsa.py::test_reverse_kl_no_topk_no_vocab_sum
# 锁定)。纯 OPD 不重新实现 KL,避免数值漂移。
from gigpo.tgsa import compute_reverse_kl_token


def compute_opd_advantage(old_log_probs: torch.Tensor,
                          teacher_log_prob: torch.Tensor,
                          response_mask: torch.Tensor,
                          response_length: int,
                          kl_coef: float = 1.0,
                          kl_penalty: str = "k3",
                          out_stats: Optional[dict] = None):
    """纯 OPD 优势:advantage = -kl_coef * KL(pi_theta || pi_T),逐 token,mask 后。

    忠实复刻 ROLL:
      * advantages = -total_weighted_kld   (functionals.py:1149)
      * advantages = advantages * response_mask   (functionals.py:1180)
    无环境奖励、无环境门控(全轨迹无条件蒸馏)、无 critic。

    Args:
        old_log_probs: (bs, response_length) 学生 log pi_theta(行为策略)。
        teacher_log_prob: (bs, response_length) 教师 log pi_T(由 sglang HTTP 评分得到)。
        response_mask: (bs, response_length) 动作 token 的 {0,1} mask。
        response_length: L(用于 doc 一致性;实际 shape 由 mask 决定)。
        kl_coef: KL 蒸馏系数(对应 ROLL opd_kl_coef,默认 1.0)。
        kl_penalty: KL 估计形式,"k3"(默认,与 ROLL 一致)/"kl"(k1)/"abs"/"mse"。
        out_stats: 可选 dict,填充 opd/* 统计量(供 tensorboard 侧信道 logger,
            沿用 TGSA 的 out_stats 模式)。None 时不填。

    Returns:
        (advantages, returns): 均为 (bs, response_length),契约与 GiGPO 的 scores
        一致,可直接写入 batch['advantages']/batch['returns'],loss 侧零改动。
    """
    # 逐 token 的 reverse-KL 贡献(未 mask 聚合):(bs, response_length)
    # k3 = (exp(teacher - student) - (teacher - student) - 1).clamp(-10, 10)
    kl_tok = compute_reverse_kl_token(
        old_log_probs=old_log_probs,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        kl_penalty=kl_penalty,
    )

    # 纯 OPD 核心:advantage = -coef * KL(KL 折叠进优势,非独立 loss 项)。
    # 负号使学生被"拉近"教师(KL 大 → 负优势 → 降低该 token 概率 → 减小 KL)。
    adv = -(kl_coef * kl_tok)  # (bs, response_length)

    # 与 ROLL 一致:advantage 乘 response_mask,只在动作 token 上有效。
    mask = response_mask.to(device=adv.device, dtype=adv.dtype)
    adv = adv * mask

    # 侧信道统计量(可选):opd/ 面板。沿用 TGSA out_stats 模式,由 logger 在
    # compute_advantage 调用处 record。纯 OPD 的教师健康(tgsa_teacher/*)与 KL
    # 信号(tgsa_signal/delta_*)已由 TGSA stats logger 覆盖,这里只补 OPD 专属量。
    if out_stats is not None:
        valid = mask.sum().clamp(min=1.0)
        out_stats["opd/kl_advantage_mean"] = float((adv * mask).sum().item() / valid.item())
        out_stats["opd/kl_coef"] = float(kl_coef)

    # returns = advantages(纯 OPD 无 critic,returns 即 advantages;对齐 ROLL)。
    return adv, adv
