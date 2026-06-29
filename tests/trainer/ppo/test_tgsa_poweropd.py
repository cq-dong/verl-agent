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

"""Integration tests for TGSA with PowerOPD mode."""

import torch
import numpy as np
import pytest

from gigpo.tgsa import compute_teacher_preference_signal, compute_tgsa_advantage


class TestTGSAWithPowerOPD:
    """Test TGSA functions with use_poweropd=True"""

    def _make_test_data(self, bs=8, seq_len=10):
        """Generate test tensors"""
        teacher_log_prob = torch.randn(bs, seq_len) * 0.5 - 1.0
        old_log_probs = torch.randn(bs, seq_len) * 0.5 - 2.0
        response_mask = torch.ones(bs, seq_len)
        step_group_uids = np.array([f"group_{i//2}" for i in range(bs)])
        group_size_per_row = torch.tensor([2 if i < bs-2 else 1 for i in range(bs)])
        return teacher_log_prob, old_log_probs, response_mask, step_group_uids, group_size_per_row

    def test_preference_signal_poweropd_bounds(self):
        """With PowerOPD, preference signal should be well-bounded"""
        teacher_lp, student_lp, mask, uids, gsize = self._make_test_data()

        l_tilde = compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_poweropd=True,
            power_alpha=5.0,
            gamma=1.0,
        )

        # Should be in [-1, 1] even for singletons (Case 2)
        assert l_tilde.min() >= -1.0 - 1e-5
        assert l_tilde.max() <= 1.0 + 1e-5

    def test_preference_signal_poweropd_vs_vanilla(self):
        """PowerOPD and vanilla should have same sign pattern"""
        teacher_lp, student_lp, mask, uids, gsize = self._make_test_data()

        l_tilde_power = compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_poweropd=True,
            power_alpha=5.0,
            gamma=1.0,
        )

        l_tilde_vanilla = compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_poweropd=False,
            gamma=1.0,
        )

        # Signs should generally agree (both indicate teacher preference direction)
        # Note: exact values differ, but polarity should match for most cases
        agreement = (torch.sign(l_tilde_power) == torch.sign(l_tilde_vanilla)).float().mean()
        assert agreement > 0.7  # At least 70% agreement

    def test_preference_signal_with_margin_poweropd(self):
        """Test margin mode with PowerOPD"""
        bs, seq_len = 8, 10
        teacher_lp = torch.randn(bs, seq_len) * 0.5 - 1.0
        student_lp = torch.randn(bs, seq_len) * 0.5 - 2.0
        mask = torch.ones(bs, seq_len)
        uids = np.array([f"g_{i//2}" for i in range(bs)])
        gsize = torch.tensor([2] * bs)
        teacher_top2 = torch.randn(bs) * 0.5 - 1.5  # Runner-up log-probs

        l_tilde = compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_margin=True,
            teacher_top2_logprob=teacher_top2,
            use_poweropd=True,
            power_alpha=5.0,
        )

        assert l_tilde.shape == (bs,)
        assert l_tilde.min() >= -1.0 - 1e-5
        assert l_tilde.max() <= 1.0 + 1e-5

    def test_tgsa_advantage_with_poweropd(self):
        """Full TGSA advantage with PowerOPD enabled"""
        bs, seq_len = 8, 10
        teacher_lp = torch.randn(bs, seq_len) * 0.5 - 1.0
        student_lp = torch.randn(bs, seq_len) * 0.5 - 2.0
        mask = torch.ones(bs, seq_len)
        uids = np.array([f"g_{i//2}" for i in range(bs)])
        gsize = torch.tensor([2 if i < 6 else 1 for i in range(bs)])

        episode_adv = torch.randn(bs)  # Simulated episode advantages
        episode_std = torch.ones(bs) * 0.5  # Non-degenerate

        adv = compute_tgsa_advantage(
            episode_advantages_row=episode_adv,
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            episode_group_std=episode_std,
            response_length=seq_len,
            lambda_=0.3,
            mu=0.1,
            use_poweropd=True,
            power_alpha=5.0,
        )

        assert adv.shape == (bs, seq_len)
        # Masked positions should be zero
        assert (adv[mask == 0] == 0).all()

    def test_poweropd_stats_logging(self):
        """Test that PowerOPD stats are properly logged"""
        teacher_lp, student_lp, mask, uids, gsize = self._make_test_data()
        out_stats = {}

        compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_poweropd=True,
            power_alpha=5.0,
            out_stats=out_stats,
        )

        assert "tgsa_signal/poweropd_enabled" in out_stats
        assert out_stats["tgsa_signal/poweropd_enabled"] == 1.0
        assert "tgsa_signal/power_alpha" in out_stats
        assert out_stats["tgsa_signal/power_alpha"] == 5.0

    def test_backward_compatibility(self):
        """Ensure vanilla mode still works (backward compatibility)"""
        teacher_lp, student_lp, mask, uids, gsize = self._make_test_data()
        out_stats = {}

        l_tilde = compute_teacher_preference_signal(
            teacher_log_prob=teacher_lp,
            old_log_probs=student_lp,
            response_mask=mask,
            step_group_uids=uids,
            group_size_per_row=gsize,
            use_poweropd=False,  # Explicitly disable
            out_stats=out_stats,
        )

        assert l_tilde.shape == (8,)
        assert out_stats["tgsa_signal/poweropd_enabled"] == 0.0
