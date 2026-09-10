# speculative_decoding

    harness/    device planning + the accept/reject rule
    training/   draft-model config and training loop
    eval/       metrics (acceptance rate, speedup) and benchmark entry point
    analysis/   JSONL aggregation and sweep tables
    tests/      CPU-only, model-free, ~2s

Local setup:

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m pytest tests/ -q

## The dev loop

    # 1. edit + test locally (seconds, no quota)
    .venv/bin/python -m pytest tests/ -q

    # 2. push
    git add -A && git commit -m "..." && git push

    # 3. run it on Kaggle (clones fresh from GitHub, runs the suite)
    kaggle kernels push -p kernels/sync
    kaggle kernels status shreysharma07/spec-sync
    kaggle kernels output shreysharma07/spec-sync -p out/sync

`kernels/sync` reports the HEAD sha it cloned -- check it matches your local
`git rev-parse --short HEAD`, otherwise you are reading results from old code.
That mismatch is the single easiest way to waste an hour here.

The repo is public, so the session clones with no credentials. If it is ever
made private you will need a PAT in Kaggle Secrets (Add-ons -> Secrets), since
the API cannot set those for you.

`pip install -e . --no-deps` is deliberate: the pins in pyproject.toml already
match the image, and letting pip resolve them risks it replacing the CUDA torch
build with a CPU wheel from PyPI.

# Kaggle remote compute

Kaggle has no SSH and no attachable VM. The loop is: write code locally ->
push it as a kernel -> it runs on Kaggle's GPU -> pull the output back.

    source .venv/bin/activate

## GPU

    kaggle kernels push -p kernels/gpu-check      # queue the run
    kaggle kernels status shreysharma07/gpu-check # PENDING / RUNNING / COMPLETE / ERROR
    kaggle kernels logs shreysharma07/gpu-check   # stdout+stderr, needed on ERROR
    kaggle kernels output shreysharma07/gpu-check -p out/gpu-check

Runs are async: push returns immediately, it does not block until done.

### Turning the GPU on and off

`machine_shape` is the real switch. `enable_gpu` is close to decorative --
all three rows below were verified by running them:

| enable_gpu | machine_shape      | what you actually get |
|------------|--------------------|-----------------------|
| true       | "NvidiaTeslaT4"    | 2x Tesla T4           |
| **false**  | "NvidiaTeslaT4"    | **still 2x Tesla T4** |
| false      | ""                 | CPU only              |

Row 2 is the trap: flipping the boolean that looks like the off-switch leaves
the GPU attached and billing, with nothing to warn you. The client sends the
two fields independently and does no reconciliation (kaggle_api_extended.py
:6437 and :6450); the server resolves the conflict in favour of machine_shape.

    "machine_shape": "NvidiaTeslaT4"   # on   (also: NvidiaTeslaP100, Tpu1VmV38)
    "machine_shape": ""                # off

Per-run override, without editing the file:

    kaggle kernels push -p kernels/gpu-check --accelerator NvidiaTeslaT4

--accelerator beats the file when present, but there is no `--accelerator none`,
so turning the GPU *off* means editing machine_shape.

### Do not take the default GPU

`enable_gpu` alone, with no machine_shape, gives a **Tesla P100 (sm_60)**, and
Kaggle's preinstalled torch 2.10 dropped sm_60. Anything touching CUDA dies with
`no kernel image is available for execution on the device` -- confusing, because
cuda_available is True and the device name prints fine. Name T4 explicitly.

Kaggle also swaps the torch build to match: 2.10.0+cu128 on GPU sessions,
2.10.0+cpu on CPU ones. Do not assume CUDA is importable when GPU is off.

### Quota

    kaggle quota    # 30 GPU h/week, 20 TPU h/week

Measured: CPU-only runs cost **zero** GPU quota (0.49h before and after a
completed run). GPU sessions bill wall-clock, not compute -- an idle attached
GPU costs the same as a training one. So turn it off for data prep.

Quota accounting lags a completed run by a minute or two; do not read it
immediately after a run and expect the final number.

## Storage

Attach inputs by editing the kernel's metadata, then push. They mount read-only
at `/kaggle/input/<slug>/`:

    "dataset_sources": ["shreysharma07/my-data"],
    "competition_sources": ["titanic"],
    "model_sources": []

Upload your own (scaffold in `datasets/my-data`, edit the metadata first):

    kaggle datasets create  -p datasets/my-data   # first time
    kaggle datasets version -p datasets/my-data -m "note"   # updates
    kaggle datasets list --user shreysharma07

### Disk inside a run

- `/kaggle/input`  read-only mounts, does not count against your output
- `/kaggle/working` ~20GB, the ONLY path persisted as kernel output
- `/kaggle/temp`   scratch, discarded when the session ends

Write anything you want to keep to `/kaggle/working`.

## Layout

    kernels/gpu-check/   pushable unit: code + kernel-metadata.json
    datasets/my-data/    dataset scaffold (metadata only, not uploaded)
    data/                downloads from Kaggle
    out/                 outputs pulled back from runs

## What the T4 session actually gives you

Measured on a `NvidiaTeslaT4` run (see `kernels/gpu-probe`):

    torch 2.10.0+cu128   CUDA 12.8   cuDNN 9.10.2   driver 580.159.04   Python 3.12.13
    2x Tesla T4, sm_75, 14.56 GiB each, ~14.46 GiB free at session start

Two separate devices, not one 32GB pool -- a model must fit in 14.5 GiB or be
sharded. Note 14.56 GiB, not the "16GB" on the spec sheet.

### bf16: works, but is a trap on this hardware

It is real bf16, not a silent downcast -- 1e30 round-trips, 1+2^-8 flushes to
1.0, 1.0 is 0x3f80, and a matmul that would overflow fp16 returns 2.55e10 with
no inf. Semantics are correct.

The problem is speed. Turing (sm_75) has no bf16 tensor cores, so it is emulated:

    2048x2048 matmul:  fp16 0.83ms   fp32 6.03ms   bf16 7.13ms

bf16 is ~8.6x slower than fp16 and slower than fp32, while also being less
accurate (7 mantissa bits vs 10; max err 0.358 vs 0.044). On a T4 bf16 is
strictly worse than fp16 on both axes. **Use fp16.**

`torch.cuda.is_bf16_supported()` returns True here and will not warn you --
it counts emulation as support.

### FlashAttention-2 is not available

    import flash_attn  ->  ModuleNotFoundError

Installing it will not help: FA2 requires sm_80+ (Ampere) and the T4 is sm_75.
Torch's own flash backend agrees on this hardware:

    FLASH_ATTENTION      RuntimeError: No available kernel
    CUDNN_ATTENTION      RuntimeError: No available kernel
    EFFICIENT_ATTENTION  works   <- use this
    MATH                 works   (fallback, most memory)

Use `torch.nn.functional.scaled_dot_product_attention`, which picks
EFFICIENT_ATTENTION here automatically.
