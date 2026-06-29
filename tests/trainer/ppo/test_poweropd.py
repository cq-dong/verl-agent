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

"""Unit tests for PowerOPD core functions."""

import torch
import pytest

from gigpo.poweropd import (
    logprob_to_prob,
    compute_power_diff,
    compute_power_margin,
    compute_power_kl_token,
    compute_power_advantage_singleton,
    compute_power_opd_advantage,
)


class TestLogprobToProb:
    def test_basic_conversion(self):
        log_prob = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        prob = logprob_to_prob(log_prob)
        expected = torch.tensor([0.5, 0.3, 0.2])
        assert torch.allclose(prob, expected, atol=1e-6)

    def test_normalization(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        prob = logprob_to_prob(logits, normalize=True)
        assert prob.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert (prob > 0).all()


class TestComputePowerDiff:
    def test_bounds_in_minus_one_one(self):
        """Power diff should always be in [-1, 1]"""
        log_T = torch.randn(100)  # Random log-probs
        log_S = torch.randn(100)
        # Convert to valid log-probs (normalize)
        log_T = torch.log_softmax(log_T, dim=-1)
        log_S = torch.log_softmax(log_S, dim=-1)

        diff = compute_power_diff(log_T, log_S, alpha=5.0)
        assert diff.min() >= -1.0 - 1e-6
        assert diff.max() <= 1.0 + 1e-6

    def test_sign_consistency(self):
        """sign(Δ^α) should equal sign(log π_T - log π_θ)"""
        log_T = torch.tensor([-1.0, -2.0, -0.5, -3.0])
        log_S = torch.tensor([-2.0, -1.0, -0.5, -3.0])

        diff = compute_power_diff(log_T, log_S, alpha=5.0)
        log_diff = log_T - log_S

        assert torch.sign(diff).eq(torch.sign(log_diff)).all()

    def test_alpha_effect(self):
        """Larger alpha should amplify margin when probs are far apart"""
        # Case: π_T = 0.9, π_S = 0.5 (both relatively high, teacher better)
        log_T = torch.log(torch.tensor([0.9]))
        log_S = torch.log(torch.tensor([0.5]))

        diff_small_alpha = compute_power_diff(log_T, log_S, alpha=1.0)
        diff_large_alpha = compute_power_diff(log_T, log_S, alpha=5.0)

        # Larger alpha amplifies the difference when both probs are high
        # (because 0.9^5 ≈ 0.59, 0.5^5 ≈ 0.03, diff ≈ 0.56 vs 0.9-0.5=0.4)
        assert abs(diff_large_alpha.item()) > abs(diff_small_alpha.item())

    def test_alpha_selectivity(self):
        """Larger alpha suppresses signal when either prob is low"""
        # Case: π_T = 0.9, π_S = 0.1 (teacher high, student low)
        log_T = torch.log(torch.tensor([0.9]))
        log_S = torch.log(torch.tensor([0.1]))

        diff_alpha1 = compute_power_diff(log_T, log_S, alpha=1.0)
        diff_alpha10 = compute_power_diff(log_T, log_S, alpha=10.0)

        # With large alpha, low prob contributes almost nothing
        # 0.9^10 ≈ 0.35, 0.1^10 ≈ 1e-10, so diff is suppressed
        assert abs(diff_alpha10.item()) < abs(diff_alpha1.item())

    def test_length_normalization(self):
        """Test with response mask"""
        bs, seq_len = 4, 10
        log_T = torch.randn(bs, seq_len)
        log_S = torch.randn(bs, seq_len)
        mask = torch.ones(bs, seq_len)
        mask[:, 5:] = 0  # Only first 5 tokens valid

        diff = compute_power_diff(
            log_T, log_S, alpha=5.0,
            length_normalize=True, response_mask=mask
        )
        assert diff.shape == (bs,)


class TestComputePowerMargin:
    def test_margin_bounds(self):
        """Margin should be in [-1, 1]"""
        log_T = torch.log(torch.tensor([0.7, 0.5, 0.3]))
        log_R = torch.log(torch.tensor([0.2, 0.4, 0.5]))

        margin = compute_power_margin(log_T, log_R, alpha=5.0)
        assert margin.min() >= -1.0 - 1e-6
        assert margin.max() <= 1.0 + 1e-6

    def test_positive_when_teacher_better(self):
        """Margin > 0 when teacher prob > runnerup"""
        log_T = torch.log(torch.tensor([0.8]))
        log_R = torch.log(torch.tensor([0.2]))

        margin = compute_power_margin(log_T, log_R, alpha=5.0)
        assert margin.item() > 0


class TestComputePowerAdvantageSingleton:
    def test_shape(self):
        bs, seq_len = 8, 20
        log_T = torch.randn(bs, seq_len)
        log_S = torch.randn(bs, seq_len)
        mask = torch.ones(bs, seq_len)

        adv = compute_power_advantage_singleton(
            log_T, log_S, mask, alpha=5.0, gamma=1.0
        )
        assert adv.shape == (bs,)
        # After z-score normalization, values can exceed [-1, 1] but typically bounded
        # The key property is that sign indicates teacher preference direction

    def test_gamma_scaling(self):
        bs, seq_len = 4, 10
        log_T = torch.randn(bs, seq_len)
        log_S = torch.randn(bs, seq_len)
        mask = torch.ones(bs, seq_len)

        adv1 = compute_power_advantage_singleton(log_T, log_S, mask, gamma=1.0)
        adv2 = compute_power_advantage_singleton(log_T, log_S, mask, gamma=2.0)

        # After z-score normalization, the scaling by gamma should approximately hold
        # (up to numerical precision of the normalization)
        ratio = (adv2 / adv1).nanmean()  # Average ratio ignoring NaNs
        assert 1.5 < ratio.item() < 2.5  # Should be roughly 2x


class TestComputePowerOPDAdvantage:
    def test_masked_output(self):
        bs, seq_len = 4, 10
        log_T = torch.randn(bs, seq_len)
        log_S = torch.randn(bs, seq_len)
        mask = torch.zeros(bs, seq_len)
        mask[:, :5] = 1

        adv, ret = compute_power_opd_advantage(
            log_T, log_S, mask, alpha=5.0, kl_coef=1.0
        )
        assert adv.shape == (bs, seq_len)
        # Masked positions should be zero
        assert (adv[:, 5:] == 0).all()

    def test_kl_coef_effect(self):
        bs, seq_len = 2, 5
        log_T = torch.randn(bs, seq_len)
        log_S = torch.randn(bs, seq_len)
        mask = torch.ones(bs, seq_len)

        adv1, _ = compute_power_opd_advantage(log_T, log_S, mask, kl_coef=1.0)
        adv2, _ = compute_power_opd_advantage(log_T, log_S, mask, kl_coef=2.0)

        assert torch.allclose(adv2, 2.0 * adv1, atol=1e-6)


class TestOPDWithPowerOPD:
    """Test pure OPD with PowerOPD mode"""

    def test_opd_poweropd_vs_vanilla(self):
        """OPD PowerOPD should produce bounded advantages"""
        from gigpo.opd import compute_opd_advantage

        bs, seq_len = 4, 10
        log_T = torch.randn(bs, seq_len) * 0.5 - 1.0
        log_S = torch.randn(bs, seq_len) * 0.5 - 2.0
        mask = torch.ones(bs, seq_len)

        # PowerOPD mode
        adv_power, _ = compute_opd_advantage(
            log_S, log_T, mask, seq_len,
            use_poweropd=True, power_alpha=5.0
        )

        # Vanilla mode
        adv_vanilla, _ = compute_opd_advantage(
            log_S, log_T, mask, seq_len,
            use_poweropd=False, kl_penalty="k3"
        )

        # Both should be properly masked
        assert (adv_power[mask == 0] == 0).all()
        assert (adv_vanilla[mask == 0] == 0).all()

    def test_opd_poweropd_stats(self):
        """PowerOPD OPD should log correct stats"""
        from gigpo.opd import compute_opd_advantage

        bs, seq_len = 4, 10
        log_T = torch.randn(bs, seq_len) * 0.5 - 1.0
        log_S = torch.randn(bs, seq_len) * 0.5 - 2.0
        mask = torch.ones(bs, seq_len)
        out_stats = {}

        compute_opd_advantage(
            log_S, log_T, mask, seq_len,
            use_poweropd=True, power_alpha=5.0,
            out_stats=out_stats
        )

        assert out_stats["opd/poweropd_enabled"] == 1.0
        assert out_stats["opd/power_alpha"] == 5.0
