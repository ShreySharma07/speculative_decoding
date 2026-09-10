"""Metrics for speculative decoding.

The speedup model follows Leviathan et al. (2023) eq. 3: with acceptance rate
alpha, block size k, and cost ratio c (draft forward / target forward), the
expected wall-clock improvement is

    (1 - alpha^(k+1)) / ((1 - alpha) * (k*c + 1))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BlockStats:
    n_blocks: int
    n_proposed: int
    n_accepted: int
    n_emitted: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target kept."""
        return self.n_accepted / self.n_proposed if self.n_proposed else 0.0

    @property
    def tokens_per_block(self) -> float:
        """Mean tokens emitted per target forward pass -- the real win metric."""
        return self.n_emitted / self.n_blocks if self.n_blocks else 0.0


def summarize_blocks(results: Iterable) -> BlockStats:
    """Aggregate SpecDecodeResult-shaped objects into one BlockStats."""
    n_blocks = n_prop = n_acc = n_emit = 0
    for r in results:
        n_blocks += 1
        n_prop += r.n_proposed
        n_acc += r.n_accepted
        n_emit += len(r.tokens)
    return BlockStats(n_blocks, n_prop, n_acc, n_emit)


def expected_speedup(alpha: float, k: int, c: float) -> float:
    """Theoretical speedup vs plain autoregressive decoding.

    alpha: acceptance rate in [0, 1]; k: draft block size; c: draft/target cost.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    if alpha == 1.0:
        return (k + 1) / (k * c + 1)          # limit of the ratio as alpha -> 1
    return (1 - alpha ** (k + 1)) / ((1 - alpha) * (k * c + 1))


def best_block_size(alpha: float, c: float, k_max: int = 16) -> int:
    """The k maximising expected_speedup for a given alpha and cost ratio."""
    return max(range(1, k_max + 1), key=lambda k: expected_speedup(alpha, k, c))


def break_even_acceptance(k: int, c: float, tol: float = 1e-6) -> float | None:
    """Minimum acceptance rate at which speculative decoding beats plain decoding.

    Below this alpha the drafting overhead costs more than the tokens it saves,
    so the whole scheme is a slowdown. Returns None when no alpha works, which
    happens once the draft is too expensive relative to the target (c >= 1).
    """
    if expected_speedup(1.0, k, c) <= 1.0:
        return None
    lo, hi = 0.0, 1.0
    while hi - lo > tol:                      # speedup is monotonic in alpha
        mid = (lo + hi) / 2
        if expected_speedup(mid, k, c) < 1.0:
            lo = mid
        else:
            hi = mid
    return round(hi, 6)


def tokens_per_second(n_tokens: int, seconds: float) -> float:
    return n_tokens / seconds if seconds > 0 else 0.0


def compare_runs(baseline: Sequence[float], candidate: Sequence[float]) -> dict:
    """Wall-clock comparison between two sets of per-request latencies."""
    b = sorted(baseline)
    c = sorted(candidate)
    def _p(xs, q):
        if not xs:
            return 0.0
        return xs[min(int(q * len(xs)), len(xs) - 1)]
    return {
        "baseline_median": _p(b, 0.5),
        "candidate_median": _p(c, 0.5),
        "baseline_p95": _p(b, 0.95),
        "candidate_p95": _p(c, 0.95),
        "median_speedup": (_p(b, 0.5) / _p(c, 0.5)) if _p(c, 0.5) else 0.0,
    }
