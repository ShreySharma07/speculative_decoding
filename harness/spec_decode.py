"""Core speculative-decoding acceptance rule (Leviathan et al., 2023).

Kept free of any model dependency so it is testable on CPU without weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpecDecodeResult:
    tokens: list[int] = field(default_factory=list)
    n_proposed: int = 0
    n_accepted: int = 0
    # True when the final token came from the residual distribution after a
    # rejection, rather than from the target's own bonus sample.
    ended_on_rejection: bool = False

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_proposed if self.n_proposed else 0.0


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) -- what the target samples from after a rejection.

    Falls back to p when the clamped mass is degenerate, which happens when q
    covers p entirely (and would otherwise divide by zero).
    """
    diff = torch.clamp(p - q, min=0.0)
    total = diff.sum()
    if total <= 0:
        return p / p.sum()
    return diff / total


def speculative_step(
    target_probs: torch.Tensor,   # [k+1, vocab] target model's distributions
    draft_probs: torch.Tensor,    # [k, vocab]   draft model's distributions
    draft_tokens: list[int],      # [k]          what the draft proposed
    rng: torch.Generator | None = None,
) -> SpecDecodeResult:
    """Run the accept/reject rule over one block of k drafted tokens.

    Accept draft token x with probability min(1, p(x)/q(x)); on the first
    rejection, emit one sample from the residual and stop. If every token is
    accepted, emit a bonus token from the target's extra distribution.
    """
    k = len(draft_tokens)
    if draft_probs.shape[0] != k:
        raise ValueError(f"draft_probs has {draft_probs.shape[0]} rows, expected {k}")
    if target_probs.shape[0] != k + 1:
        raise ValueError(f"target_probs has {target_probs.shape[0]} rows, expected {k + 1}")

    out = SpecDecodeResult(n_proposed=k)

    for i, tok in enumerate(draft_tokens):
        p, q = target_probs[i], draft_probs[i]
        ratio = (p[tok] / q[tok]).clamp(max=1.0) if q[tok] > 0 else torch.tensor(0.0)
        u = torch.rand(1, generator=rng, device=p.device).item()
        if u < ratio.item():
            out.tokens.append(int(tok))
            out.n_accepted += 1
            continue
        resid = residual_distribution(p, q)
        out.tokens.append(int(torch.multinomial(resid, 1, generator=rng).item()))
        out.ended_on_rejection = True
        return out

    bonus = torch.multinomial(target_probs[k], 1, generator=rng).item()
    out.tokens.append(int(bonus))
    return out
