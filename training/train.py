"""Draft-model training loop.

Deliberately thin: the point of the repo is the speculative-decoding harness,
so this is the smallest loop that will actually fine-tune a draft model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from harness.device import plan_device
from training.config import TrainConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )


def train(cfg: TrainConfig, model, dataloader, log_path: str | None = None) -> dict:
    plan = plan_device()
    model = model.to(plan.device, dtype=plan.dtype)
    model.train()
    opt = build_optimizer(model, cfg)
    # GradScaler is only meaningful for fp16; bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler(enabled=plan.dtype is torch.float16)

    history, step, t0 = [], 0, time.perf_counter()
    for batch in dataloader:
        batch = {k: v.to(plan.device) for k, v in batch.items()}
        with torch.autocast(device_type=plan.device.split(":")[0], dtype=plan.dtype):
            loss = model(**batch).loss / cfg.grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            rec = {"step": step, "loss": float(loss.item() * cfg.grad_accum),
                   "elapsed_s": round(time.perf_counter() - t0, 1)}
            history.append(rec)
            print(json.dumps(rec), flush=True)

        step += 1
        if step >= cfg.max_steps:
            break

    out = {"steps": step, "plan": plan.to_dict(), "history": history}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(out, indent=2))
    return out
