"""Greedy decode benchmark for Qwen2.5 Instruct.

    python -m eval.qwen --model Qwen/Qwen2.5-1.5B-Instruct \
        --prompt-tokens 128 --new-tokens 128 --runs 5 --results-dir results

    # static KV cache + torch.compile (mode="reduce-overhead", i.e. CUDA graphs)
    python -m eval.qwen --cache static --compile --warmup 3 ...

Sizes on a Kaggle T4 (14.56 GiB): 0.5B, 1.5B and 3B fit in fp16. 7B does not --
its weights alone are 14.19 GiB, before KV cache and CUDA context.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, CompileConfig,
                          GenerationConfig, StaticCache)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Repeated to build a prompt of an exact token length. Prompt content barely
# moves decode speed; prompt length does, so length is what we hold fixed.
_FILLER = (
    "Speculative decoding drafts several tokens with a small model and verifies "
    "them in a single forward pass of the large model. "
)

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_CACHES = ("dynamic", "static")


@dataclass
class DecodeBenchResult:
    model_name: str
    dtype: str
    gpu: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    n_runs: int
    cache_implementation: str
    compiled: bool
    fullgraph: bool | None
    n_warmup: int
    warmup_times_s: list[float]
    run_times_s: list[float]
    tokens_generated: list[int]
    peak_memory_allocated_gb: float | None
    peak_memory_reserved_gb: float | None
    # generate() wall time includes the prefill pass. prefill_time_s is measured
    # separately so decode_tokens_per_s is decode-only; run_times_s stays raw.
    prefill_time_s: float
    decode_tokens_per_s: list[float]
    generated_ids: list[int]          # new tokens from the last timed run
    sample_output: str
    env: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return self.cache_implementation + ("+compile" if self.compiled else "")

    def summary(self) -> dict:
        med_run = statistics.median(self.run_times_s)
        med_tps = statistics.median(self.decode_tokens_per_s)
        return {
            "mode": self.mode,
            "median_run_time_s": round(med_run, 4),
            "mean_run_time_s": round(statistics.fmean(self.run_times_s), 4),
            "median_decode_tokens_per_s": round(med_tps, 2),
            "median_decode_ms_per_token": round(1000 / med_tps, 3) if med_tps > 0 else None,
            "prefill_time_s": round(self.prefill_time_s, 4),
            "first_warmup_s": self.warmup_times_s[0],
            "last_warmup_s": self.warmup_times_s[-1],
            # A last warmup still well above the timed runs means compilation or
            # CUDA-graph recording leaked into the measurement: add warmups.
            "warmed_up": self.warmup_times_s[-1] <= 1.15 * med_run,
            "peak_memory_allocated_gb": self.peak_memory_allocated_gb,
            "peak_memory_reserved_gb": self.peak_memory_reserved_gb,
            "all_runs_hit_token_target": all(n == self.max_new_tokens for n in self.tokens_generated),
        }


def build_fixed_prompt(tokenizer, n_tokens: int) -> torch.Tensor:
    """Exactly n_tokens ids. Built at the id level: re-tokenizing decoded text
    does not reliably round-trip to the same length."""
    unit = tokenizer(_FILLER, add_special_tokens=False)["input_ids"]
    ids = (unit * (n_tokens // len(unit) + 1))[:n_tokens]
    return torch.tensor([ids], dtype=torch.long)


def greedy_config(base: GenerationConfig, n_new: int, pad_fallback: int | None) -> GenerationConfig:
    """A clean greedy config carrying over only the special-token ids.

    Qwen2.5's generation_config.json sets do_sample=True, temperature=0.7,
    top_p=0.8, top_k=20 and repetition_penalty=1.1. Overriding do_sample alone
    leaves repetition_penalty active -- it is valid under greedy, so nothing
    warns -- which changes the chosen tokens and adds per-step work to the very
    thing being timed. So the model's config is replaced, not patched.
    """
    return GenerationConfig(
        do_sample=False,
        max_new_tokens=n_new,
        min_new_tokens=n_new,   # EOS cannot end a run early: every run decodes the same count
        repetition_penalty=1.0,
        eos_token_id=base.eos_token_id,
        bos_token_id=base.bos_token_id,
        pad_token_id=base.pad_token_id if base.pad_token_id is not None else pad_fallback,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_generate(model, input_ids, attention_mask, device, cache=None, **gen_kwargs):
    if cache is not None:
        # Outside the clock. Zeroes in place, so tensor addresses stay fixed for
        # CUDA graphs; without it generate would treat the prompt as a continuation.
        cache.reset()
        gen_kwargs["past_key_values"] = cache
    _sync(device)
    t0 = time.perf_counter()
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    _sync(device)
    return time.perf_counter() - t0, out


def benchmark_decode(
    model_name: str = DEFAULT_MODEL,
    prompt_tokens: int = 128,
    max_new_tokens: int = 128,
    n_runs: int = 5,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    cache_implementation: str = "dynamic",
    use_compile: bool = False,
    fullgraph: bool = True,
    n_warmup: int = 1,
    results_dir: str | Path = "results",
    report_json: str | Path | None = None,
    wandb_project: str | None = "speculative-decoding",
    wandb_mode: str | None = None,
) -> DecodeBenchResult:
    """Load the model, warm up, then time n_runs greedy generations.

    Results are written to results_dir (and report_json) before W&B is touched,
    so a W&B failure can never lose the numbers. wandb_project=None skips W&B.
    """
    if n_runs < 1 or n_warmup < 1 or prompt_tokens < 1 or max_new_tokens < 2:
        raise ValueError("need n_runs >= 1, n_warmup >= 1, prompt_tokens >= 1, max_new_tokens >= 2")
    if cache_implementation not in _CACHES:
        raise ValueError(f"cache_implementation must be one of {_CACHES}")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={device} requested but CUDA is not available")
    if use_compile and cache_implementation != "static":
        # generate() only compiles with a compileable cache. With a dynamic one it
        # quietly runs eager, and we would report eager numbers labelled compiled.
        raise ValueError("use_compile=True requires cache_implementation='static'")
    if use_compile and dev.type != "cuda":
        raise ValueError("use_compile=True requires CUDA; generate() will not auto-compile elsewhere")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # One device on purpose: device_map="auto" would shard across both T4s and
    # the timing would include inter-GPU copies rather than the model.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()

    gen_cfg = greedy_config(model.generation_config, max_new_tokens, tokenizer.eos_token_id)
    # transformers 5.0.0 compiles generate()'s decode loop by itself whenever the
    # cache is compileable (static) on CUDA, so model.forward must NOT also be
    # wrapped in torch.compile -- that compiles twice. Prefill stays eager either
    # way. The flip side: a static cache without compile needs disable_compile.
    gen_cfg.disable_compile = not use_compile
    gen_cfg.compile_config = (CompileConfig(fullgraph=fullgraph, mode="reduce-overhead")
                              if use_compile else None)
    model.generation_config = gen_cfg
    if use_compile:
        # Past the recompile limit dynamo silently falls back to eager. Make that a
        # crash: an eager run labelled "compiled" is worse than no run at all.
        torch._dynamo.config.fail_on_recompile_limit_hit = True

    # One cache for every call, sized for the longest one. Fixed shapes and
    # addresses are what CUDA graphs need, and it means the max_new_tokens=1
    # prefill timing below cannot change a shape and trigger a recompile.
    cache = (StaticCache(config=model.config, max_cache_len=prompt_tokens + max_new_tokens)
             if cache_implementation == "static" else None)

    input_ids = build_fixed_prompt(tokenizer, prompt_tokens).to(dev)
    attention_mask = torch.ones_like(input_ids)

    # Qwen2.5 is trained in bf16 and fp16 overflows past 65504. Greedy argmax
    # over NaN logits still "runs", so check once rather than benchmark garbage.
    # Done before the peak reset so these full-vocab logits are not counted.
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"{model_name} gives non-finite logits in {dtype}")
    del logits

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    # Warmup, results discarded. Eager needs one call; compiled needs several,
    # since compilation and CUDA-graph recording happen across the first calls.
    warmup_times, sample = [], ""
    for i in range(n_warmup):
        dt, warm = _timed_generate(model, input_ids, attention_mask, dev, cache)
        warmup_times.append(round(dt, 5))
        if i == n_warmup - 1:
            sample = tokenizer.decode(warm[0, prompt_tokens:], skip_special_tokens=True)
        del warm

    times, counts, last_ids = [], [], []
    for _ in range(n_runs):
        dt, out = _timed_generate(model, input_ids, attention_mask, dev, cache)
        times.append(dt)
        counts.append(int(out.shape[1] - prompt_tokens))
        last_ids = out[0, prompt_tokens:].tolist()
        del out

    peak_alloc = peak_res = None
    if dev.type == "cuda":
        peak_alloc = round(torch.cuda.max_memory_allocated(dev) / 1024**3, 3)
        peak_res = round(torch.cuda.max_memory_reserved(dev) / 1024**3, 3)

    # Prefill timed the same way: one new token = the prompt forward + first pick.
    prefill = [
        _timed_generate(model, input_ids, attention_mask, dev, cache,
                        max_new_tokens=1, min_new_tokens=1)[0]
        for _ in range(n_runs)
    ]
    prefill_s = statistics.median(prefill)
    decode_tps = [
        round((n - 1) / (t - prefill_s), 2) if t > prefill_s and n > 1 else 0.0
        for t, n in zip(times, counts)
    ]

    import transformers
    result = DecodeBenchResult(
        model_name=model_name,
        dtype=str(dtype).removeprefix("torch."),
        gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        prompt=tokenizer.decode(input_ids[0]),
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        n_runs=n_runs,
        cache_implementation=cache_implementation,
        compiled=use_compile,
        fullgraph=fullgraph if use_compile else None,
        n_warmup=n_warmup,
        warmup_times_s=warmup_times,
        run_times_s=[round(t, 5) for t in times],
        tokens_generated=counts,
        peak_memory_allocated_gb=peak_alloc,
        peak_memory_reserved_gb=peak_res,
        prefill_time_s=round(prefill_s, 5),
        decode_tokens_per_s=decode_tps,
        generated_ids=last_ids,
        sample_output=sample[:200],
        env={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "device": str(dev),
        },
    )

    csv_path = save_results(result, results_dir)
    print(f"results -> {csv_path}")
    if report_json:
        rp = Path(report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({**asdict(result), "summary": result.summary(),
                                  "results_csv": str(csv_path)}, indent=2))
    if wandb_project:
        log_to_wandb(result, wandb_project, wandb_mode, csv_path)
    return result


def save_results(result: DecodeBenchResult, results_dir: str | Path) -> Path:
    """One CSV per benchmark, one row per timed run, config repeated per row so
    the file stands alone."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_slug = result.mode.replace("+", "-")
    path = out / f"decode_{result.model_name.split('/')[-1]}_{result.dtype}_{mode_slug}_{stamp}.csv"

    config = {
        "model_name": result.model_name, "dtype": result.dtype, "gpu": result.gpu,
        "cache_implementation": result.cache_implementation, "compiled": result.compiled,
        "fullgraph": result.fullgraph, "n_warmup": result.n_warmup,
        "first_warmup_s": result.warmup_times_s[0],
        "prompt_tokens": result.prompt_tokens, "max_new_tokens": result.max_new_tokens,
        "n_runs": result.n_runs, "peak_memory_allocated_gb": result.peak_memory_allocated_gb,
        "peak_memory_reserved_gb": result.peak_memory_reserved_gb,
        "prefill_time_s": result.prefill_time_s, "prompt": result.prompt,
    }
    fields = ["run", "time_s", "tokens_generated", "decode_tokens_per_s", *config]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (t, n, tps) in enumerate(zip(result.run_times_s, result.tokens_generated,
                                            result.decode_tokens_per_s)):
            w.writerow({"run": i, "time_s": t, "tokens_generated": n,
                        "decode_tokens_per_s": tps, **config})
    return path


def log_to_wandb(result: DecodeBenchResult, project: str, mode: str | None,
                 csv_path: Path) -> str | None:
    """Config + per-run numbers + summary. Falls back to offline rather than
    failing, since the CSV is already safely on disk."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipped W&B logging")
        return None

    config = {
        "model_name": result.model_name, "dtype": result.dtype, "prompt": result.prompt,
        "prompt_tokens": result.prompt_tokens, "max_new_tokens": result.max_new_tokens,
        "n_runs": result.n_runs, "gpu": result.gpu,
        "cache_implementation": result.cache_implementation, "compiled": result.compiled,
        "fullgraph": result.fullgraph, "n_warmup": result.n_warmup, **result.env,
    }
    run = None
    for attempt_mode in ([mode] if mode == "offline" else [mode, "offline"]):
        try:
            run = wandb.init(project=project, config=config, job_type="decode-benchmark",
                             mode=attempt_mode)
            break
        except Exception as e:
            print(f"W&B init (mode={attempt_mode}) failed: {type(e).__name__}: {e}")
    if run is None:
        return None

    for i, (t, n, tps) in enumerate(zip(result.run_times_s, result.tokens_generated,
                                        result.decode_tokens_per_s)):
        run.log({"run": i, "time_s": t, "tokens_generated": n, "decode_tokens_per_s": tps})
    run.summary.update({**result.summary(), "results_csv": str(csv_path)})
    url = getattr(run, "url", None)
    run.finish()
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dtype", choices=_DTYPES, default="fp16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", choices=_CACHES, default="dynamic")
    ap.add_argument("--compile", action="store_true", help="requires --cache static")
    ap.add_argument("--no-fullgraph", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--wandb-project", default="speculative-decoding")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    r = benchmark_decode(
        model_name=a.model, prompt_tokens=a.prompt_tokens, max_new_tokens=a.new_tokens,
        n_runs=a.runs, dtype=_DTYPES[a.dtype], device=a.device,
        cache_implementation=a.cache, use_compile=a.compile, fullgraph=not a.no_fullgraph,
        n_warmup=a.warmup, results_dir=a.results_dir, report_json=a.report_json,
        wandb_project=None if a.no_wandb else a.wandb_project, wandb_mode=a.wandb_mode,
    )
    for k, v in r.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
