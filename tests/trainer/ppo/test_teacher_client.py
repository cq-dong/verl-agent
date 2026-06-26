# Copyright 2025 TGSA-GRPO contributors
# Licensed under the Apache License, Version 2.0 (the "License").
"""Unit tests for gigpo/teacher_client.py with a MOCKED sglang HTTP layer.

No real server / no `requests` needed: we monkeypatch SGLangTeacherClient._post
to return a synthetic sglang /generate response and verify the extraction,
response-window alignment, margin runner-up, retry, and error handling.
"""

import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gigpo.teacher_client import SGLangTeacherClient  # noqa: E402


def _approx(expected, tol=1e-6):
    import math

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


def _fake_meta(logprobs, token_ids, top_logprobs=None):
    """Build an sglang-style meta_info dict.

    logprobs: list of floats (per input token).
    token_ids: list of ints (per input token, 1:1 with logprobs).
    top_logprobs: optional list (per token) of list-of-dicts {logprob, token_id}.
    """
    meta = {"input_token_logprobs": logprobs, "input_token_ids": token_ids}
    if top_logprobs is not None:
        meta["input_top_logprobs"] = top_logprobs
    return meta


def _make_client(fake_responses, **kw):
    """Client whose _post pops from fake_responses (list of meta_info dicts)."""
    kw.setdefault("max_concurrency", 4)
    c = SGLangTeacherClient(base_url="http://fake", **kw)
    calls = {"n": 0}

    def _fake_post(payload):
        calls["n"] += 1
        meta = fake_responses.pop(0)
        return {"text": "", "meta_info": meta}

    c._post = _fake_post
    c._calls = calls
    return c


# --------------------------------------------------------------------------- #
# 1. response-window extraction (full-length array)                           #
# --------------------------------------------------------------------------- #
def test_extract_response_window_full_length():
    # total_length=6, response_length=3 -> response window = positions [3,4,5]
    bs, total, resp = 2, 6, 3
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    # row0 teacher logprobs: full array of length 6; response = last 3 = [-1,-1,-1]
    # row1 response = last 3 = [-3,-3,-3]
    metas = [
        _fake_meta([0.0, -5.0, -5.0, -1.0, -1.0, -1.0], [10, 11, 12, 13, 14, 15]),
        _fake_meta([0.0, -5.0, -5.0, -3.0, -3.0, -3.0], [20, 21, 22, 23, 24, 25]),
    ]
    c = _make_client(metas)
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    assert lp.shape == (bs, resp)
    assert torch.allclose(lp[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6)
    assert torch.allclose(lp[1], torch.tensor([-3.0, -3.0, -3.0]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. first-token-dropped array (sglang convention A vs B)                     #
# --------------------------------------------------------------------------- #
def test_extract_response_window_first_token_dropped():
    # sglang may drop the first (unconditioned) token: arrays length = total-1 = 5.
    # response window (last 3) must still be the response tokens [-1,-1,-1].
    bs, total, resp = 1, 6, 3
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    # length-5 arrays (tokens 1..5); last 3 = tokens [3,4,5] = response
    metas = [_fake_meta([-5.0, -5.0, -1.0, -1.0, -1.0], [11, 12, 13, 14, 15])]
    c = _make_client(metas)
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    assert torch.allclose(lp[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. dict-form logprob entries (some sglang versions return dicts)            #
# --------------------------------------------------------------------------- #
def test_dict_form_logprob_entries():
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    metas = [_fake_meta(
        [{"logprob": 0.0, "token_id": 9}, {"logprob": -5.0, "token_id": 8},
         {"logprob": -2.0, "token_id": 7}, {"logprob": -4.0, "token_id": 6}],
        [9, 8, 7, 6])]
    c = _make_client(metas)
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    assert torch.allclose(lp[0], torch.tensor([-2.0, -4.0]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. margin runner-up (need_top2)                                             #
# --------------------------------------------------------------------------- #
def test_margin_runner_up():
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    # response tokens at positions [2,3] with sampled token_ids [7,6]
    # position 2: top-k = [{7,-2.0},{8,-2.5}] -> sampled 7 -> runner-up 8: -2.5
    # position 3: top-k = [{6,-4.0},{5,-4.8}] -> sampled 6 -> runner-up 5: -4.8
    # length-normalized runner-up over 2 action tokens = (-2.5 + -4.8)/2 = -3.65
    tops = [
        [{"logprob": 0.0, "token_id": 9}],
        [{"logprob": -5.0, "token_id": 8}],
        [{"logprob": -2.0, "token_id": 7}, {"logprob": -2.5, "token_id": 8}],
        [{"logprob": -4.0, "token_id": 6}, {"logprob": -4.8, "token_id": 5}],
    ]
    metas = [_fake_meta([0.0, -5.0, -2.0, -4.0], [9, 8, 7, 6], top_logprobs=tops)]
    c = _make_client(metas, top_logprobs_num=2)
    lp, top2 = c.compute_teacher_signals(input_ids, response_mask, need_top2=True)
    assert torch.allclose(lp[0], torch.tensor([-2.0, -4.0]), atol=1e-6)
    assert top2[0].item() == _approx((-2.5 + -4.8) / 2.0)


def test_margin_uses_response_mask_for_length_norm():
    # only one of two response positions is an action token (mask [1,0])
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.tensor([[1.0, 0.0]])
    tops = [
        [{"logprob": 0.0, "token_id": 9}],
        [{"logprob": -5.0, "token_id": 8}],
        [{"logprob": -2.0, "token_id": 7}, {"logprob": -2.5, "token_id": 8}],
        [{"logprob": -4.0, "token_id": 6}, {"logprob": -4.8, "token_id": 5}],
    ]
    metas = [_fake_meta([0.0, -5.0, -2.0, -4.0], [9, 8, 7, 6], top_logprobs=tops)]
    c = _make_client(metas, top_logprobs_num=2)
    _, top2 = c.compute_teacher_signals(input_ids, response_mask, need_top2=True)
    # only position 0 (action) counts -> runner-up = -2.5
    assert top2[0].item() == _approx(-2.5)


# --------------------------------------------------------------------------- #
# 5. retry on failure                                                         #
# --------------------------------------------------------------------------- #
def test_retry_then_succeed():
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    good = _fake_meta([0.0, 0.0, -1.0, -1.0], [1, 2, 3, 4])
    c = SGLangTeacherClient(base_url="http://fake", max_concurrency=1,
                            max_retries=3, retry_backoff=0.0)
    state = {"i": 0}

    def _flaky(payload):
        state["i"] += 1
        if state["i"] < 3:
            raise RuntimeError("transient")
        return {"meta_info": good}

    c._post = _flaky
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    assert torch.allclose(lp[0], torch.tensor([-1.0, -1.0]), atol=1e-6)
    assert state["i"] == 3


def test_retry_exhausted_raises():
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    c = SGLangTeacherClient(base_url="http://fake", max_concurrency=1,
                            max_retries=2, retry_backoff=0.0)
    c._post = lambda payload: (_ for _ in ()).throw(RuntimeError("down"))
    raised = False
    try:
        c.compute_teacher_log_prob(input_ids, response_mask)
    except RuntimeError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# 6. concurrency: rows scored out of order, results reassembled in order      #
# --------------------------------------------------------------------------- #
def test_results_reassembled_in_order():
    bs, total, resp = 5, 4, 2
    input_ids = torch.arange(bs * total).reshape(bs, total).to(torch.long)
    response_mask = torch.ones(bs, resp)
    # each row's response logprob = its row index (encode in last 2 positions)
    metas = []
    for i in range(bs):
        v = float(i)
        metas.append(_fake_meta([0.0, 0.0, v, v], [0, 0, i, i]))
    c = _make_client(metas, max_concurrency=5)
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    # row i -> response window = [i, i]
    for i in range(bs):
        assert lp[i, 0].item() == _approx(float(i))
        assert lp[i, 1].item() == _approx(float(i))


# --------------------------------------------------------------------------- #
# 7. device round-trip (CPU here; API must return on input device)            #
# --------------------------------------------------------------------------- #
def test_returns_on_input_device():
    bs, total, resp = 1, 4, 2
    input_ids = torch.zeros(bs, total, dtype=torch.long)
    response_mask = torch.ones(bs, resp)
    metas = [_fake_meta([0.0, 0.0, -1.0, -1.0], [1, 2, 3, 4])]
    c = _make_client(metas)
    lp = c.compute_teacher_log_prob(input_ids, response_mask)
    assert lp.device == input_ids.device
    assert lp.dtype == torch.float32


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
