"""Probe the Kaggle GPU session: device count, VRAM, bf16 behaviour, FlashAttention-2."""
import json, os, subprocess, sys, traceback

out = {}

def probe(name):
    """Record a probe's return value, or its exception, without killing the run."""
    def wrap(fn):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"ERROR": f"{type(e).__name__}: {e}"}
        return fn
    return wrap

import torch
import torch.nn.functional as F

# ---------- versions ----------
@probe("versions")
def _v():
    return {
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver_cuda_via_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip() if os.path.exists("/usr/bin/nvidia-smi")
            or subprocess.run(["which","nvidia-smi"],capture_output=True).returncode==0 else None,
        "python": sys.version.split()[0],
    }

# ---------- devices + VRAM ----------
@probe("devices")
def _d():
    n = torch.cuda.device_count()
    devs = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)   # bytes, live from the driver
        devs.append({
            "index": i,
            "name": p.name,
            "capability": f"sm_{p.major}{p.minor}",
            "total_vram_gb": round(p.total_memory / 1024**3, 2),
            "free_vram_gb": round(free / 1024**3, 2),
            "used_vram_gb": round((total - free) / 1024**3, 2),
            "multi_processor_count": p.multi_processor_count,
        })
    return {"count": n, "gpus": devs}

# ---------- bf16 ----------
@probe("bf16")
def _b():
    r = {"torch.cuda.is_bf16_supported": torch.cuda.is_bf16_supported()}
    try:
        r["is_bf16_supported_including_emulation"] = torch.cuda.is_bf16_supported(
            including_emulation=True)
    except TypeError:
        r["is_bf16_supported_including_emulation"] = "kwarg not available"

    # Storage fidelity. These two values discriminate real bf16 from fp16:
    #   1e30      -> bf16 holds it (max ~3.4e38); fp16 overflows to inf (max 65504)
    #   1+2^-8    -> bf16 has 7 mantissa bits so it rounds to 1.0;
    #                fp16 has 10 bits so it keeps 1.00390625
    t_big = torch.tensor([1e30], dtype=torch.bfloat16, device="cuda")
    t_eps = torch.tensor([1.0 + 2**-8], dtype=torch.bfloat16, device="cuda")
    r["storage"] = {
        "1e30_roundtrip": float(t_big.item()),
        "1+2^-8_roundtrip": float(t_eps.item()),
        "raw_bits_of_1.0": hex(torch.tensor([1.0], dtype=torch.bfloat16,
                                            device="cuda").view(torch.int16).item() & 0xFFFF),
        "itemsize_bytes": t_big.element_size(),
    }
    r["storage"]["verdict"] = (
        "true bf16" if t_big.item() > 1e29 and t_eps.item() == 1.0 else "NOT bf16 semantics")

    # Does a bf16 matmul actually run, and does it match an fp32 reference?
    torch.manual_seed(0)
    a32 = torch.randn(512, 512, device="cuda")
    b32 = torch.randn(512, 512, device="cuda")
    ref = a32 @ b32
    res = {}
    for tag, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            o = (a32.to(dt) @ b32.to(dt)).float()
            res[tag] = {
                "ran": True,
                "max_abs_err_vs_fp32": round((o - ref).abs().max().item(), 5),
                "mean_abs_err_vs_fp32": round((o - ref).abs().mean().item(), 6),
            }
        except Exception as e:
            res[tag] = {"ran": False, "error": f"{type(e).__name__}: {e}"}
    r["matmul"] = res

    # Overflow probe: 1e4 * 1e4 * 256 overflows fp16's 65504 but not bf16.
    # If bf16 were silently computed/stored as fp16, this comes back inf.
    big = torch.full((256, 256), 1e4, dtype=torch.bfloat16, device="cuda")
    ob = (big @ big).float()
    r["overflow_probe"] = {
        "bf16_256x(1e4*1e4)": None if torch.isinf(ob).any().item() else round(ob[0, 0].item(), 1),
        "any_inf": bool(torch.isinf(ob).any().item()),
        "note": "inf here would mean fp16 range, i.e. a silent downcast",
    }

    # Speed: on hardware without bf16 tensor cores it is typically much slower than fp16.
    import time
    def bench(dt):
        x = torch.randn(2048, 2048, device="cuda", dtype=dt)
        for _ in range(3): x @ x
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): x @ x
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) / 20 * 1000, 2)
    r["matmul_ms_2048"] = {}
    for tag, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try: r["matmul_ms_2048"][tag] = bench(dt)
        except Exception as e: r["matmul_ms_2048"][tag] = f"{type(e).__name__}: {e}"
    return r

# ---------- FlashAttention-2 ----------
@probe("flash_attention_2")
def _f():
    r = {}
    try:
        import flash_attn
        r["import"] = "OK"
        r["version"] = getattr(flash_attn, "__version__", "unknown")
        try:
            from flash_attn import flash_attn_func
            q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
            flash_attn_func(q, q, q)
            r["runs"] = True
        except Exception as e:
            r["runs"] = False
            r["run_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        r["import"] = "FAILED"
        r["import_error"] = f"{type(e).__name__}: {e}"
    return r

# ---------- what attention backends DO work ----------
@probe("sdpa_backends")
def _s():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    res = {}
    for be in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"):
        if not hasattr(SDPBackend, be):
            res[be] = "backend not in this torch"; continue
        try:
            with sdpa_kernel(getattr(SDPBackend, be)):
                F.scaled_dot_product_attention(q, q, q)
            res[be] = "works"
        except Exception as e:
            res[be] = f"{type(e).__name__}: {str(e)[:150]}"
    return res

print(json.dumps(out, indent=2))
smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if \
    subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0 else "(no nvidia-smi)"
print(smi)
with open("/kaggle/working/probe.json", "w") as f:
    json.dump(out, f, indent=2)
