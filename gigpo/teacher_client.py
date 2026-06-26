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

"""HTTP client for a teacher model served by an sglang server (TGSA-GRPO).

The teacher scores the STUDENT's sampled action tokens: for each input row
(prompt context + action, in the student's token space) it returns
log pi_T(token | prefix) at every position. We extract the response (action)
window to form ``teacher_log_prob`` (bs, response_length) -- the input to
``gigpo.tgsa.compute_tgsa_advantage``.

Design decisions (see idea.md + project memory):
  * Teacher is an INDEPENDENT sglang HTTP service (port exposed), NOT a verl
    Ray worker. The 32B+ teacher runs on its own GPUs; verl just calls HTTP.
  * Assumption: teacher shares the student's tokenizer, so the student's
    input_ids are sent directly (sglang /generate accepts ``input_ids``).
    Different-tokenizer re-tokenization + position alignment is future work.
  * Reverse-KL / preference use SINGLE-TOKEN log-probs only (no top-k vocab
    sum). top_logprobs_num>=2 is used ONLY for the optional margin variant.
  * Layout contract (verl): each row of input_ids is [context | response] with
    the response occupying the LAST response_length columns; response_mask
    (bs, response_length) marks the action tokens within that window.

sglang v0.4.6.post5 /generate contract used here:
  request: {input_ids, sampling_params:{max_new_tokens:0},
            return_logprob:true, logprob_start_len:0, top_logprobs_num:N}
  response.meta_info: {input_token_logprobs:[...], input_token_ids:[...],
                       input_top_logprobs:[ [ {logprob,token_id}, ... ], ... ]}

Parsing is tolerant: input_token_logprobs entries may be float or
{"logprob": float, ...}; arrays may be full-length or drop the first
(unconditioned) token -- we align 1:1 with input_token_ids and take the last
response_length entries, which is correct under either convention.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch


class SGLangTeacherClient:
    """Client that scores student action tokens with a teacher sglang server."""

    def __init__(self, base_url: str,
                 max_concurrency: int = 8,
                 timeout: float = 60.0,
                 max_retries: int = 3,
                 retry_backoff: float = 2.0,
                 top_logprobs_num: int = 0):
        """
        Args:
            base_url: e.g. "http://teacher-host:30000".
            max_concurrency: parallel HTTP requests (one per row).
            timeout: per-request timeout (seconds).
            max_retries: retries on HTTP/parse failure.
            retry_backoff: exponential backoff base.
            top_logprobs_num: 0 = off (preference/KL only need single-token
                logprob); >=2 enables the optional margin variant (runner-up
                teacher logprob per action token).
        """
        self.base_url = base_url.rstrip("/")
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.top_logprobs_num = int(top_logprobs_num)

        # per-``compute_teacher_signals`` health counters (reset at the start of
        # each call, accumulated from every concurrent ``_post_row``). Used for
        # the tgsa_teacher/* tensorboard stats.
        self._health = {"n_success": 0, "n_retry": 0, "n_total": 0}
        self._health_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # HTTP layer (mockable for unit tests)                               #
    # ------------------------------------------------------------------ #
    def _post(self, payload: dict) -> dict:
        import requests
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_row(self, input_ids_row: List[int]) -> dict:
        """Score one row (no generation); return the raw sglang response dict."""
        payload = {
            "input_ids": [int(x) for x in input_ids_row],
            "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,  # full array; slice response window 1:1 with input_token_ids
            "top_logprobs_num": self.top_logprobs_num,
            "return_text_in_logprobs": False,
        }
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._post(payload)
                self._record_row_health(success=True, retries=attempt)
                return resp
            except Exception as e:  # noqa: BLE001 - retry any failure
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt)
        self._record_row_health(success=False, retries=self.max_retries)
        raise RuntimeError(
            f"Teacher request failed after {self.max_retries + 1} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    # health counters (for tgsa_teacher/* stats)                          #
    # ------------------------------------------------------------------ #
    def _reset_health(self) -> None:
        with self._health_lock:
            self._health = {"n_success": 0, "n_retry": 0, "n_total": 0}

    def _record_row_health(self, *, success: bool, retries: int) -> None:
        with self._health_lock:
            self._health["n_total"] += 1
            if success:
                self._health["n_success"] += 1
            self._health["n_retry"] += int(retries)

    # ------------------------------------------------------------------ #
    # parsing                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_logprobs(meta: dict) -> Tuple[List[float], List[int]]:
        """Return (logprobs, token_ids) 1:1 aligned, from sglang meta_info.

        Tolerant to: float or dict logprob entries; int/dict/str token entries;
        alternative key names across sglang patch versions.
        """
        lp = (meta.get("input_token_logprobs")
              or meta.get("input_logprobs")
              or meta.get("token_logprobs")
              or [])
        ids = (meta.get("input_token_ids")
               or meta.get("input_tokens")
               or [])
        lp_vals: List[float] = []
        for e in lp:
            if e is None:
                lp_vals.append(0.0)
            elif isinstance(e, dict):
                lp_vals.append(float(e.get("logprob", 0.0)))
            else:
                lp_vals.append(float(e))
        id_vals: List[int] = []
        for e in ids:
            if isinstance(e, dict):
                id_vals.append(int(e.get("token_id", -1)))
            elif isinstance(e, (int, np.integer)):
                id_vals.append(int(e))
            elif isinstance(e, str):
                id_vals.append(-1)  # decoded string; id not recoverable
            elif e is None:
                id_vals.append(-1)
            else:
                id_vals.append(int(e))
        return lp_vals, id_vals

    @staticmethod
    def _parse_top_logprobs(meta: dict) -> List[List[dict]]:
        """Per-token top-k list of {token_id, logprob}. Empty if absent."""
        tp = meta.get("input_top_logprobs") or meta.get("input_top_logprob") or []
        out: List[List[dict]] = []
        for e in tp:
            if e is None:
                out.append([])
                continue
            row: List[dict] = []
            for d in e:
                if isinstance(d, dict):
                    row.append({"token_id": int(d.get("token_id", -1)),
                                "logprob": float(d.get("logprob", 0.0))})
                else:
                    row.append({"token_id": -1, "logprob": float(d)})
            out.append(row)
        return out

    @staticmethod
    def _slice_response(seq: list, response_length: int) -> list:
        """Take the last response_length entries; left-pad with 0 if short."""
        if len(seq) >= response_length:
            return list(seq[-response_length:])
        return [0.0 for _ in range(response_length - len(seq))] + list(seq)

    # ------------------------------------------------------------------ #
    # main API                                                           #
    # ------------------------------------------------------------------ #
    def compute_teacher_signals(self, input_ids: torch.Tensor,
                                response_mask: torch.Tensor,
                                need_top2: bool = False
                                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Score every row; return teacher log-probs over the response window.

        One HTTP request per row (concurrent up to max_concurrency). Both the
        preference/KL signal and the optional margin runner-up are parsed from
        the SAME response, so enabling margin costs no extra requests.

        Args:
            input_ids: (bs, total_length) LongTensor (any device).
            response_mask: (bs, response_length).
            need_top2: if True, also return length-normalized runner-up logprob
                per row (requires top_logprobs_num>=2 at init).

        Returns:
            teacher_log_prob: (bs, response_length) FloatTensor on input_ids.device.
            teacher_top2: (bs,) FloatTensor if need_top2 else None.
        """
        device = input_ids.device
        bs = input_ids.shape[0]
        response_length = response_mask.shape[1]
        cpu_ids = input_ids.detach().cpu().tolist()
        mask_cpu = response_mask.detach().cpu().to(torch.float32)
        if need_top2:
            assert self.top_logprobs_num >= 2, \
                "need_top2=True requires top_logprobs_num>=2 at init."

        def _score(args):
            i, row = args
            meta = self._post_row(row).get("meta_info", {})
            lp_vals, id_vals = self._parse_logprobs(meta)
            resp_lp = self._slice_response(lp_vals, response_length)
            top2_row: Optional[List[float]] = None
            if need_top2:
                tops = self._parse_top_logprobs(meta)
                n = len(lp_vals)
                # top-k and ids are 1:1 with lp_vals (sglang contract). Whether
                # or not sglang dropped the first (unconditioned) token, the
                # last response_length entries still span the response window.
                if len(tops) == n:
                    tops_win = tops[-response_length:]
                else:
                    tops_win = [[] for _ in range(response_length)]
                if len(id_vals) == n:
                    ids_win = id_vals[-response_length:]
                else:
                    ids_win = [-1 for _ in range(response_length)]
                top2_row = []
                for j in range(response_length):
                    sampled_id = ids_win[j] if j < len(ids_win) else -1
                    best_other: Optional[float] = None
                    for d in (tops_win[j] if j < len(tops_win) else []):
                        if d["token_id"] != sampled_id:
                            if best_other is None or d["logprob"] > best_other:
                                best_other = d["logprob"]
                    top2_row.append(best_other if best_other is not None else resp_lp[j])
            return i, resp_lp, top2_row

        lp_results: List[List[float]] = [None] * bs
        top2_results: List[Optional[List[float]]] = [None] * bs
        self._reset_health()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            for i, resp_lp, top2_row in pool.map(_score, list(enumerate(cpu_ids))):
                lp_results[i] = resp_lp
                top2_results[i] = top2_row
        latency_s = time.perf_counter() - t0

        # ---- teacher health stats (side-channel; no-op when logger disabled) ----
        try:
            from gigpo.tgsa_stats import TGSAStatsLogger
            with self._health_lock:
                n_total = self._health["n_total"]
                n_success = self._health["n_success"]
                n_retry = self._health["n_retry"]
            TGSAStatsLogger.get().record_scalars({
                "tgsa_teacher/latency_s": float(latency_s),
                "tgsa_teacher/success_rate": float(n_success / n_total) if n_total else 0.0,
                "tgsa_teacher/retry_count": float(n_retry),
                "tgsa_teacher/rows": float(n_total),
            })
        except Exception:  # noqa: BLE001 - stats must never crash scoring
            pass

        teacher_log_prob = torch.tensor(lp_results, dtype=torch.float32, device=device)
        teacher_top2 = None
        if need_top2:
            teacher_top2 = torch.zeros(bs, dtype=torch.float32, device=device)
            for i in range(bs):
                r = torch.tensor(top2_results[i][:response_length], dtype=torch.float32)
                m = mask_cpu[i]
                valid = m.sum().clamp(min=1.0)
                teacher_top2[i] = (r * m).sum() / valid
        return teacher_log_prob, teacher_top2

    def compute_teacher_log_prob(self, input_ids: torch.Tensor,
                                 response_mask: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: teacher log-prob over the response window only."""
        lp, _ = self.compute_teacher_signals(input_ids, response_mask, need_top2=False)
        return lp

    def health_check(self) -> bool:
        """GET /health; return True if the teacher server is reachable."""
        try:
            import requests
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False
