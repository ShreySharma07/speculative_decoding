"""Same-session decode comparison on one T4:
{Qwen2.5-0.5B, Qwen2.5-1.5B} x {eager, static cache + torch.compile}.

Each config runs in its own subprocess: same host and GPU (so c is not polluted
by session-to-session variance), but a fresh dynamo/inductor state, so one
model's compiles cannot count toward the other's recompile limit.
"""
import json, os, re, shutil, statistics, subprocess, sys, time

REPO = "https://github.com/ShreySharma07/speculative_decoding.git"
DEST = "/kaggle/working/speculative_decoding"
OUT = "/kaggle/working"
MODELS = {"draft": "Qwen/Qwen2.5-0.5B-Instruct", "target": "Qwen/Qwen2.5-1.5B-Instruct"}
# Interleaved so slow drift across the session does not land on one mode.
PLAN = [("eager", "draft"), ("eager", "target"), ("compile", "draft"), ("compile", "target")]
N_RUNS, COMPILE_WARMUP = 5, 3

if os.path.exists(DEST):
    shutil.rmtree(DEST)
subprocess.run(["git", "clone", "--depth", "1", REPO, DEST], check=True)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=DEST,
                     capture_output=True, text=True).stdout.strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
               cwd=DEST, check=True)

wandb_mode = "offline"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb_mode = "online"
except Exception as e:
    print(f"no WANDB_API_KEY secret ({type(e).__name__}); W&B will run offline", flush=True)

env = dict(os.environ, WANDB_DIR=OUT, PYTHONUNBUFFERED="1",
           # Surface the ways compile silently loses its speedup.
           TORCH_LOGS="recompiles,graph_breaks,perf_hints")
os.makedirs(f"{OUT}/reports", exist_ok=True)
os.makedirs(f"{OUT}/logs", exist_ok=True)


def run(mode, role, fullgraph=True):
    name = f"{role}-{mode}" + ("" if fullgraph else "-nofullgraph")
    report = f"{OUT}/reports/{name}.json"
    cmd = [sys.executable, "-m", "eval.qwen", "--model", MODELS[role], "--runs", str(N_RUNS),
           "--results-dir", f"{OUT}/results", "--report-json", report, "--wandb-mode", wandb_mode]
    if mode == "compile":
        cmd += ["--cache", "static", "--compile", "--warmup", str(COMPILE_WARMUP)]
        if not fullgraph:
            cmd.append("--no-fullgraph")
    print(f"--> {name}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=DEST, env=env, capture_output=True, text=True)
    log = p.stdout + "\n" + p.stderr
    with open(f"{OUT}/logs/{name}.log", "w") as f:
        f.write(log)
    info = {
        "name": name, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1),
        "recompiles": len(re.findall(r"Recompiling function", log)),
        "graph_break_lines": len(re.findall(r"[Gg]raph break", log)),
        "cudagraph_skips": len(re.findall(r"skipping cudagraphs", log)),
        "recompile_limit_hit": "recompile_limit" in log,
    }
    if p.returncode != 0:
        info["error_tail"] = log[-1500:]
    print(f"    rc={p.returncode} wall={info['wall_s']}s recompiles={info['recompiles']} "
          f"cudagraph_skips={info['cudagraph_skips']}", flush=True)
    return info, (json.load(open(report)) if p.returncode == 0 else None)


runs, reports = [], {}
for mode, role in PLAN:
    info, rep = run(mode, role)
    if rep is None and mode == "compile":
        # fullgraph=True raises on any graph break; record that and retry without it.
        runs.append(info)
        info, rep = run(mode, role, fullgraph=False)
    runs.append(info)
    if rep is not None:
        reports[(mode, role)] = rep


def ms_per_token(rep):
    return 1000 / statistics.median(rep["decode_tokens_per_s"])


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


summary = {"repo_sha": sha, "wandb_mode": wandb_mode, "runs": runs, "configs": {}}
for (mode, role), rep in reports.items():
    s = rep["summary"]
    summary["configs"][f"{role}-{mode}"] = {
        "model": rep["model_name"], "gpu": rep["gpu"], "fullgraph": rep["fullgraph"],
        "ms_per_token": round(ms_per_token(rep), 3), **s,
        "warmup_times_s": rep["warmup_times_s"], "run_times_s": rep["run_times_s"],
    }

for mode in ("eager", "compile"):
    d, t = reports.get((mode, "draft")), reports.get((mode, "target"))
    if d and t:
        summary[f"c_{mode}"] = round(ms_per_token(d) / ms_per_token(t), 4)
for role in ("draft", "target"):
    e, c = reports.get(("eager", role)), reports.get(("compile", role))
    if e and c:
        summary[f"{role}_compile_speedup"] = round(ms_per_token(e) / ms_per_token(c), 3)
        # fp16 kernel fusion can change logits enough to flip a greedy argmax.
        summary[f"{role}_first_token_divergence_eager_vs_compile"] = first_divergence(
            e["generated_ids"], c["generated_ids"])

print(json.dumps(summary, indent=2))
with open(f"{OUT}/compare_report.json", "w") as f:
    json.dump(summary, f, indent=2)
