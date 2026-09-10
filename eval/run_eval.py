"""Entry point: run a speculative-decoding benchmark and write JSONL results."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval.metrics import summarize_blocks, expected_speedup
from harness.device import plan_device
from harness.spec_decode import speculative_step


def run_synthetic(n_blocks: int, k: int, vocab: int, seed: int = 0) -> dict:
    """Model-free smoke benchmark -- exercises the accept/reject path end to end."""
    g = torch.Generator().manual_seed(seed)
    results, t0 = [], time.perf_counter()
    for _ in range(n_blocks):
        target = torch.softmax(torch.randn(k + 1, vocab, generator=g), dim=-1)
        draft = torch.softmax(torch.randn(k, vocab, generator=g), dim=-1)
        toks = torch.multinomial(draft, 1, generator=g).squeeze(-1).tolist()
        results.append(speculative_step(target, draft, toks, rng=g))
    stats = summarize_blocks(results)
    return {
        "n_blocks": stats.n_blocks,
        "acceptance_rate": round(stats.acceptance_rate, 4),
        "tokens_per_block": round(stats.tokens_per_block, 4),
        "expected_speedup_at_c0.1": round(
            expected_speedup(stats.acceptance_rate, k, 0.1), 4),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "plan": plan_device().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--out", default="/kaggle/working/eval.jsonl")
    args = ap.parse_args()

    rec = run_synthetic(args.blocks, args.k, args.vocab)
    print(json.dumps(rec, indent=2))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
