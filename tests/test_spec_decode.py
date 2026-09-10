"""Tests for the acceptance rule and the metrics derived from it.

All CPU-only and model-free, so they run in a couple of seconds locally and
inside a Kaggle session without burning GPU quota.
"""
from __future__ import annotations

import math

import pytest
import torch

from analysis.report import load_runs, summarize, sweep_table
from eval.metrics import (BlockStats, best_block_size, break_even_acceptance,
                          expected_speedup, summarize_blocks)
from harness.device import plan_device
from harness.spec_decode import residual_distribution, speculative_step


def _uniform(rows: int, vocab: int) -> torch.Tensor:
    return torch.full((rows, vocab), 1.0 / vocab)


# ---------- acceptance rule ----------

def test_identical_distributions_accept_everything():
    # p == q makes the ratio min(1, p/q) == 1, so every draft token must survive.
    k, vocab = 4, 8
    res = speculative_step(_uniform(k + 1, vocab), _uniform(k, vocab), [1, 2, 3, 4])
    assert res.n_accepted == k
    assert res.acceptance_rate == 1.0
    assert not res.ended_on_rejection
    assert len(res.tokens) == k + 1, "all-accept must also emit the bonus token"


def test_zero_target_probability_always_rejects():
    # p(x) == 0 forces ratio 0, so the first drafted token can never be accepted.
    vocab = 4
    target = torch.tensor([[0.0, 0.5, 0.5, 0.0], [0.25] * 4])
    draft = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    res = speculative_step(target, draft, [0])
    assert res.n_accepted == 0
    assert res.ended_on_rejection
    assert len(res.tokens) == 1
    assert res.tokens[0] in (1, 2), "residual must resample from where p > q"


def test_rejection_stops_the_block_early():
    vocab = 4
    target = torch.zeros(4, vocab)
    target[:, 3] = 1.0                      # target only ever wants token 3
    draft = torch.zeros(3, vocab)
    draft[:, 0] = 1.0                       # draft only ever proposes token 0
    res = speculative_step(target, draft, [0, 0, 0])
    assert res.n_proposed == 3
    assert res.n_accepted == 0
    assert len(res.tokens) == 1, "must stop at the first rejection, not run the block"
    assert res.tokens[0] == 3


def test_shape_validation():
    with pytest.raises(ValueError, match="draft_probs"):
        speculative_step(_uniform(3, 4), _uniform(5, 4), [1, 2])
    with pytest.raises(ValueError, match="target_probs"):
        speculative_step(_uniform(9, 4), _uniform(2, 4), [1, 2])


# ---------- residual ----------

def test_residual_is_a_valid_distribution():
    p = torch.tensor([0.5, 0.3, 0.2, 0.0])
    q = torch.tensor([0.1, 0.4, 0.1, 0.4])
    r = residual_distribution(p, q)
    assert math.isclose(r.sum().item(), 1.0, rel_tol=1e-6)
    assert (r >= 0).all()
    assert r[1].item() == 0.0, "mass where q >= p must be clamped away"


def test_residual_handles_q_covering_p():
    # p - q <= 0 everywhere would divide by zero; we fall back to p.
    p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    r = residual_distribution(p, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert math.isclose(r.sum().item(), 1.0, rel_tol=1e-6)
    assert not torch.isnan(r).any()


# ---------- metrics ----------

def test_block_stats_arithmetic():
    s = BlockStats(n_blocks=10, n_proposed=40, n_accepted=30, n_emitted=40)
    assert s.acceptance_rate == 0.75
    assert s.tokens_per_block == 4.0


def test_block_stats_handles_empty():
    s = BlockStats(0, 0, 0, 0)
    assert s.acceptance_rate == 0.0 and s.tokens_per_block == 0.0


def test_summarize_blocks_matches_manual_totals():
    res = [
        speculative_step(_uniform(3, 8), _uniform(2, 8), [1, 2]),
        speculative_step(_uniform(3, 8), _uniform(2, 8), [3, 4]),
    ]
    s = summarize_blocks(res)
    assert s.n_blocks == 2 and s.n_proposed == 4 and s.n_accepted == 4


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_zero_acceptance_is_never_a_win(k):
    # alpha=0 means every block costs k draft passes and still emits one token.
    assert expected_speedup(0.0, k, 0.1) == pytest.approx(1 / (k * 0.1 + 1))


def test_speedup_is_monotonic_in_acceptance():
    vals = [expected_speedup(a, 4, 0.1) for a in (0.0, 0.25, 0.5, 0.75, 0.9)]
    assert vals == sorted(vals)


def test_speedup_alpha_one_matches_the_limit():
    # Guards the special-cased alpha == 1 branch against the 0/0 it replaces.
    assert expected_speedup(1.0, 4, 0.1) == pytest.approx(5 / 1.4)
    assert expected_speedup(0.999999, 4, 0.1) == pytest.approx(5 / 1.4, rel=1e-3)


def test_speedup_rejects_bad_input():
    for bad in ((1.5, 4, 0.1), (-0.1, 4, 0.1), (0.5, 0, 0.1), (0.5, 4, -1.0)):
        with pytest.raises(ValueError):
            expected_speedup(*bad)


def test_break_even_is_the_actual_crossing_point():
    k, c = 4, 0.1
    alpha = break_even_acceptance(k, c)
    assert alpha is not None
    # Just below it we must lose, just above it we must win.
    assert expected_speedup(alpha - 1e-3, k, c) < 1.0 < expected_speedup(alpha + 1e-3, k, c)


def test_break_even_impossible_when_draft_costs_as_much_as_target():
    assert break_even_acceptance(4, 1.0) is None


def test_cheaper_draft_prefers_bigger_blocks():
    assert best_block_size(0.9, 0.01) >= best_block_size(0.9, 0.5)


# ---------- analysis ----------

def test_load_runs_missing_file_is_empty():
    assert load_runs("/nonexistent/eval.jsonl") == []


def test_load_runs_skips_malformed_lines(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text('{"acceptance_rate": 0.5, "tokens_per_block": 3.0}\n\n{truncated\n')
    runs = load_runs(p)
    assert len(runs) == 1
    assert summarize(runs)["n_runs"] == 1


def test_summarize_empty():
    assert summarize([])["n_runs"] == 0


def test_sweep_table_shape():
    rows = sweep_table(0.8, 0.1, k_max=5)
    assert len(rows) == 5
    assert [r["k"] for r in rows] == [1, 2, 3, 4, 5]


# ---------- device ----------

def test_plan_device_is_coherent():
    plan = plan_device()
    assert plan.device.startswith(("cuda", "cpu"))
    if plan.device == "cpu":
        assert plan.dtype is torch.float32
    else:
        # The T4 finding: without native bf16 we must not hand back bfloat16.
        assert plan.bf16_is_native or plan.dtype is torch.float16
