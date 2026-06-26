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

"""Side-channel tensorboard logger for TGSA-GRPO.

Rationale
---------
TGSA produces many intermediate quantities (group structure, teacher signal,
advantage decomposition, teacher I/O health) that are computed INSIDE the
``gigpo`` package and are NOT visible to verl's own ``metrics`` dict (which is
assembled in ``ray_trainer.fit`` from worker ``meta_info['metrics']`` only).
Rather than threading all of them back through verl's metric plumbing -- which
would require modifying verl-original files (``ray_trainer.py`` metrics dict,
``metric_utils.py``, ``tracking.py``) -- this module provides a self-contained
``SummaryWriter`` that writes directly to the SAME tensorboard logdir verl uses
(``$TENSORBOARD_DIR`` or ``tensorboard_log``), so the ``tgsa_*`` curves merge
into the same tensorboard view as verl's ``actor/*`` / ``critic/*`` curves.

The logger is a process-wide singleton, DISABLED by default. It is enabled
once in ``RayPPOTrainer.fit`` (new TGSA code) when ``tgsa_enabled``; in unit
tests it stays disabled, so ``record_*`` / ``flush`` are no-ops and never
touch the filesystem. Teacher scoring (``teacher_client``) and advantage
compute (``core_gigpo`` / ``tgsa``) call ``record_*`` to buffer; the trainer
calls ``flush(global_steps)`` once per step.

Step axis: ``flush(step)`` is called with verl's ``self.global_steps`` from the
new TGSA block in ``ray_trainer``, so curves are on the true global-step axis
(resume-safe). If ``flush()`` is called with no argument, an internal counter
(1:1 with training steps on a fresh run) is used as a fallback.

Metric groups (tensorboard panels via the ``/`` prefix):
  tgsa_teacher/   teacher HTTP health (latency, success, retries)
  tgsa_group/     group-level structure, PER-GROUP (not per-row):
                 episode/step group counts, sizes, degeneration rate, and the
                 all-success / all-fail split of degenerate groups (idea mu+/mu-)
  tgsa_signal/    teacher preference signal L_T, delta, delta_norm, clip usage
  tgsa_advantage/ advantage decomposition: A_E, |A_E|, lambda/mu terms, A_total,
                 four-quadrant populations, sign-preservation invariant
  tgsa_coverage/  per-row population fractions (singleton / normal / degenerate)
  tgsa_cfg/       numeric config echo (lambda, mu, gamma, ...)

All scalars AND histograms are recorded EVERY step (no downsampling); the cost
is a handful of ``.item()`` calls + ``np.histogram`` on the driver CPU, which
is negligible next to the teacher HTTP forward and the actor backward.
"""

import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Same default verl's _TensorboardAdapter uses (verl/utils/tracking.py), so the
# two writers land in the same directory and tensorboard merges them.
_DEFAULT_LOG_DIR = "tensorboard_log"


class TGSAStatsLogger:
    """Process-wide singleton tensorboard logger for TGSA stats.

    Disabled by default. ``enable()`` lazily constructs a ``SummaryWriter``.
    All ``record_*`` / ``flush`` are no-ops while disabled, so unit tests (which
    never call ``enable``) pay nothing and never write files.
    """

    _instance: Optional["TGSAStatsLogger"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._enabled = False
        self._writer = None
        self._log_dir: Optional[str] = None
        self._buf_scalars: Dict[str, float] = {}
        self._buf_hists: Dict[str, np.ndarray] = {}
        self._step = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # singleton                                                          #
    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls) -> "TGSAStatsLogger":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset_for_test(cls) -> "TGSAStatsLogger":
        """Drop the singleton (tests only) so state never leaks between tests."""
        with cls._instance_lock:
            cls._instance = None
        inst = cls.get()
        return inst

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def enable(self, log_dir: Optional[str] = None, writer=None) -> None:
        """Enable logging and (lazily) create the SummaryWriter.

        Args:
            log_dir: tensorboard output dir. Defaults to ``$TENSORBOARD_DIR`` or
                ``tensorboard_log`` -- the SAME default verl's tensorboard
                adapter uses, so curves merge into one tensorboard view.
            writer: optional pre-built writer (test injection). When given,
                ``log_dir`` is ignored and the writer is used directly, so tests
                need neither tensorboard installed nor any filesystem.

        Degrades gracefully: if tensorboard is not installed (e.g. a wandb-only
        run env), this warns once and stays DISABLED, so TGSA stats are simply
        skipped rather than crashing the training loop. ``record_*`` / ``flush``
        then remain no-ops.
        """
        if self._enabled and self._writer is not None:
            return
        if writer is not None:
            self._writer = writer
            self._log_dir = log_dir
            self._enabled = True
            return
        self._log_dir = log_dir or os.environ.get("TENSORBOARD_DIR", _DEFAULT_LOG_DIR)
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self._log_dir)
            self._enabled = True
        except Exception as e:  # noqa: BLE001 - tensorboard absent / io error
            if not getattr(self, "_warned_no_tb", False):
                print(f"[TGSA] tensorboard SummaryWriter unavailable ({e!r}); "
                      f"tgsa_* stats will be skipped. Install tensorboard to enable.")
                self._warned_no_tb = True
            self._enabled = False
            self._writer = None

    def disable(self) -> None:
        """Disable and close the writer (idempotent)."""
        with self._lock:
            self._enabled = False
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:  # noqa: BLE001
                    pass
                self._writer = None
            self._buf_scalars.clear()
            self._buf_hists.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ #
    # recording (buffer; no-op when disabled)                            #
    # ------------------------------------------------------------------ #
    def record_scalar(self, name: str, value) -> None:
        if not self._enabled:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._buf_scalars[name] = v

    def record_histogram(self, name: str, values) -> None:
        if not self._enabled:
            return
        arr = _to_float_numpy(values)
        if arr is None or arr.size == 0:
            return
        with self._lock:
            self._buf_hists[name] = arr

    def record_scalars(self, scalars: Dict[str, float]) -> None:
        if not self._enabled:
            return
        with self._lock:
            for name, value in scalars.items():
                try:
                    self._buf_scalars[name] = float(value)
                except (TypeError, ValueError):
                    continue

    def record_histograms(self, hists: Dict[str, np.ndarray]) -> None:
        if not self._enabled:
            return
        with self._lock:
            for name, values in hists.items():
                arr = _to_float_numpy(values)
                if arr is not None and arr.size > 0:
                    self._buf_hists[name] = arr

    # ------------------------------------------------------------------ #
    # flush                                                              #
    # ------------------------------------------------------------------ #
    def flush(self, step: Optional[int] = None) -> None:
        """Write all buffered metrics at ``step`` and clear the buffer.

        Args:
            step: global step (verl ``self.global_steps``). If None, an internal
                counter is used and then incremented (fallback for callers
                without access to global_steps).
        """
        if not self._enabled or self._writer is None:
            return
        with self._lock:
            scalars = self._buf_scalars
            hists = self._buf_hists
            if not scalars and not hists:
                return  # nothing recorded this step (e.g. non-GiGPO estimator)
            if step is None:
                s = self._step
                self._step += 1
            else:
                s = int(step)
            writer = self._writer
            for name, value in scalars.items():
                writer.add_scalar(name, value, s)
            for name, arr in hists.items():
                if arr is None or arr.size == 0:
                    continue
                try:
                    writer.add_histogram(name, arr, s)
                except Exception:  # noqa: BLE001 - never let logging crash a step
                    pass
            try:
                writer.flush()
            except Exception:  # noqa: BLE001
                pass
            self._buf_scalars = {}
            self._buf_hists = {}


def _to_float_numpy(values) -> Optional[np.ndarray]:
    """Coerce a tensor / list / ndarray to a 1-D float64 numpy array, or None."""
    if values is None:
        return None
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return arr.reshape(-1)


# --------------------------------------------------------------------------- #
# Group-level statistics (PER-GROUP, not per-row).                            #
# Pure function: takes raw arrays, returns (scalars, hists) dicts ready for   #
# record_scalars / record_histograms. Called once per step from core_gigpo.   #
# --------------------------------------------------------------------------- #
def compute_group_stats(
    token_level_rewards: torch.Tensor,
    index,
    traj_index,
    step_group_uids,
    eps_deg: float = 0.01,
    success_thresh: float = 0.0,
    sigma_r=None,
    gsize=None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute PER-GROUP structural statistics for tensorboard logging.

    Decoupled from TGSA: this works for plain GiGPO (no teacher) too. The
    ``gigpo_group/`` tensorboard panel it produces is the diagnostic for whether
    GiGPO's step advantage A^S has any signal at all -- if anchor states rarely
    overlap (all singletons), A^S ~ 0 and GiGPO degenerates toward a pure
    REINFORCE baseline.

    Two group families (both counted per-group, i.e. after de-duplication, NOT
    per-row -- per-row fractions would be diluted by large groups):

    * **episode (prompt) groups** keyed by ``index``: the group over which A^E
      is normalized and whose return collapse (sigma^R_group <= eps) triggers
      the 1_deg mu-fallback (TGSA) / flags env-signal collapse (plain GiGPO).
      We report the count, mean size, the degenerate fraction, and -- crucially
      -- the all-success vs all-fail split of degenerate groups (idea mu+/mu-):
      these two degeneration modes have OPPOSITE semantics and opposite fixes,
      so a merged degenerate fraction alone is blind. The degenerate-group
      mean-return histogram exposes the bimodality directly.
    * **step (anchor) groups** keyed by ``step_group_uids``: the group over
      which A^S is normalized and which drives the Case1/Case2 switch
      (|G_t|>=2 -> Case1 ranking; |G_t|==1 -> Case2 difference, TGSA). We report
      the singleton vs multi-member fraction (the Case2 vs Case1-eligible
      population) and the size distribution.

    Args:
        token_level_rewards: (bs, L) per-token rewards.
        index: (bs,) episode-group uid per row.
        traj_index: (bs,) trajectory uid per row.
        step_group_uids: (bs,) anchor cluster id per row (from build_step_group).
        eps_deg: degeneration threshold on sigma^R_group.
        success_thresh: return threshold separating success (>thresh) from fail
            (<=thresh). For 0/1 search rewards, 0.0 is correct.
        sigma_r: optional (bs,) sigma^R_group per row (episode group return std;
            shared across all rows of a group; singleton episode groups = 1.0
            sentinel). When None (plain GiGPO), the per-group std is computed
            INTERNALLY from the trajectory returns -- numerically identical to
            core_gigpo._episode_group_return_std, which the TGSA path passes in
            explicitly. TGSA callers pass it to stay byte-identical with the
            advantage-side 1_deg computation; plain GiGPO callers omit it.
        gsize: optional (bs,) |G_t| per row (unused here; kept for API symmetry).

    Returns:
        (scalars, hists): dicts of {metric_name: float} and {metric_name: ndarray}.
    """
    row_returns = token_level_rewards.detach().to(torch.float64).sum(dim=-1).cpu().numpy()
    index = np.asarray(index)
    traj_index = np.asarray(traj_index)
    step_uids = np.asarray(step_group_uids, dtype=object)

    scalars: Dict[str, float] = {}
    hists: Dict[str, np.ndarray] = {}

    # ---- episode (prompt) groups ----
    # per-trajectory total return = sum of its per-step token rewards
    traj_return: dict = {}
    traj_to_idx: dict = {}
    for i in range(len(traj_index)):
        t = traj_index[i]
        traj_return[t] = traj_return.get(t, 0.0) + float(row_returns[i])
        traj_to_idx[t] = index[i]

    id2returns: dict = {}
    for t, r in traj_return.items():
        id2returns.setdefault(traj_to_idx[t], []).append(r)

    # per-episode-group sigma: either from the caller-provided sigma_r (TGSA,
    # byte-identical to the advantage-side 1_deg), or computed internally from
    # the group's trajectory returns (plain GiGPO). Singleton episode groups
    # (1 trajectory) get a 1.0 sentinel -> never degenerate, mirroring
    # _episode_group_return_std.
    id2sigma: dict = {}
    if sigma_r is not None:
        sigma_r_np = sigma_r.detach().cpu().numpy().astype(np.float64) \
            if torch.is_tensor(sigma_r) else np.asarray(sigma_r, dtype=np.float64)
        for i in range(len(index)):
            idx = index[i]
            if idx not in id2sigma:
                id2sigma[idx] = float(sigma_r_np[i])
    else:
        for g, rets in id2returns.items():
            id2sigma[g] = 1.0 if len(rets) <= 1 else float(np.std(rets))

    n_ep = len(id2returns)
    ep_group_ids = list(id2returns.keys())
    ep_sizes = np.array([len(id2returns[g]) for g in ep_group_ids], dtype=np.float64)
    ep_ret_mean = np.array([float(np.mean(id2returns[g])) for g in ep_group_ids], dtype=np.float64)

    # degenerate episode groups: size>=2 AND sigma^R_group <= eps_deg
    deg_mask = np.array(
        [(len(id2returns[g]) >= 2) and (id2sigma.get(g, 1.0) <= eps_deg) for g in ep_group_ids],
        dtype=bool,
    )
    n_deg = int(deg_mask.sum())

    scalars["gigpo_group/n_episode_groups"] = float(n_ep)
    scalars["gigpo_group/episode_group_size_mean"] = float(ep_sizes.mean()) if n_ep else 0.0
    scalars["gigpo_group/frac_degenerate"] = float(n_deg / n_ep) if n_ep else 0.0
    if n_deg > 0:
        deg_returns = ep_ret_mean[deg_mask]
        scalars["gigpo_group/degenerate_return_mean"] = float(deg_returns.mean())
        scalars["gigpo_group/frac_degenerate_all_success"] = float((deg_returns > success_thresh).mean())
        scalars["gigpo_group/frac_degenerate_all_fail"] = float((deg_returns <= success_thresh).mean())
        hists["gigpo_group/degenerate_return_hist"] = deg_returns
    else:
        scalars["gigpo_group/degenerate_return_mean"] = 0.0
        scalars["gigpo_group/frac_degenerate_all_success"] = 0.0
        scalars["gigpo_group/frac_degenerate_all_fail"] = 0.0
    if n_ep > 0:
        hists["gigpo_group/episode_group_size_hist"] = ep_sizes

    # ---- step (anchor) groups ----
    uniq, counts = np.unique(step_uids, return_counts=True)
    n_sg = len(uniq)
    counts_f = counts.astype(np.float64)
    scalars["gigpo_group/n_step_groups"] = float(n_sg)
    if n_sg > 0:
        scalars["gigpo_group/step_frac_singleton"] = float((counts_f == 1).mean())
        scalars["gigpo_group/step_frac_multimember"] = float((counts_f >= 2).mean())
        scalars["gigpo_group/step_group_size_mean"] = float(counts_f.mean())
        hists["gigpo_group/step_group_size_hist"] = counts_f
    else:
        scalars["gigpo_group/step_frac_singleton"] = 0.0
        scalars["gigpo_group/step_frac_multimember"] = 0.0
        scalars["gigpo_group/step_group_size_mean"] = 0.0

    return scalars, hists


# Back-compat alias: the TGSA path originally called compute_tgsa_group_stats.
# Kept so any external references still resolve; the canonical name is now
# compute_group_stats (decoupled from TGSA).
compute_tgsa_group_stats = compute_group_stats
