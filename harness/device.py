"""Device and dtype selection.

The defaults here encode what we measured on Kaggle's T4 (see README): Turing
has no bf16 tensor cores, so bf16 runs ~8.6x slower than fp16 *and* carries
3 fewer mantissa bits. torch.cuda.is_bf16_supported() returns True anyway
because it counts emulation, so we decide from compute capability instead.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

# bf16 tensor cores landed with Ampere; FlashAttention-2 requires the same.
_BF16_MIN_MAJOR = 8


@dataclass(frozen=True)
class DevicePlan:
    device: str
    dtype: torch.dtype
    capability: tuple[int, int] | None
    bf16_is_native: bool
    attn_backend: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dtype"] = str(self.dtype)
        return d


def plan_device(prefer_bf16: bool = True, index: int = 0) -> DevicePlan:
    """Pick the fastest numerically-sane dtype for the GPU we actually landed on."""
    if not torch.cuda.is_available():
        return DevicePlan("cpu", torch.float32, None, False, "math",
                          "no CUDA device; fp32 on CPU")

    major, minor = torch.cuda.get_device_capability(index)
    native_bf16 = major >= _BF16_MIN_MAJOR

    if prefer_bf16 and native_bf16:
        dtype, reason = torch.bfloat16, f"sm_{major}{minor} has native bf16"
    elif prefer_bf16:
        dtype = torch.float16
        reason = (f"sm_{major}{minor} lacks bf16 tensor cores; bf16 would be "
                  f"emulated and slower than fp32 -- using fp16 instead")
    else:
        dtype, reason = torch.float16, "bf16 not requested"

    # FA2 needs sm_80+; on older cards the mem-efficient kernel is the best available.
    backend = "flash" if native_bf16 else "mem_efficient"
    return DevicePlan(f"cuda:{index}", dtype, (major, minor), native_bf16, backend, reason)
