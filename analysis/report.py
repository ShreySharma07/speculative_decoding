from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from eval.metrics import expected_speedup


def load_runs(path: str | Path) -> list[dict]:
    """Read a JSONL eval log, skipping blank and malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    runs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a truncated final line is normal if a run was killed
    return runs


def summarize(runs: list[dict]) -> dict:
    if not runs:
        return {"n_runs": 0}
    acc = [r["acceptance_rate"] for r in runs if "acceptance_rate" in r]
    tpb = [r["tokens_per_block"] for r in runs if "tokens_per_block" in r]
    return {
        "n_runs": len(runs),
        "acceptance_rate_mean": round(mean(acc), 4) if acc else None,
        "acceptance_rate_median": round(median(acc), 4) if acc else None,
        "tokens_per_block_mean": round(mean(tpb), 4) if tpb else None,
        "best_run": max(runs, key=lambda r: r.get("tokens_per_block", 0)),
    }


def sweep_table(alpha: float, c: float, k_max: int = 8) -> list[dict]:
    """Expected speedup across block sizes -- the table you actually tune from."""
    return [
        {"k": k, "expected_speedup": round(expected_speedup(alpha, k, c), 4)}
        for k in range(1, k_max + 1)
    ]


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows)
    return f"{head}\n{sep}\n{body}"
