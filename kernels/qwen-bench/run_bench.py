"""Clone the repo into a T4 session and run eval/qwen.py's decode benchmark."""
import json, os, shutil, subprocess, sys

REPO = "https://github.com/ShreySharma07/speculative_decoding.git"
DEST = "/kaggle/working/speculative_decoding"
RESULTS = "/kaggle/working/results"

if os.path.exists(DEST):
    shutil.rmtree(DEST)
subprocess.run(["git", "clone", "--depth", "1", REPO, DEST], check=True)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=DEST,
                     capture_output=True, text=True).stdout.strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
               cwd=DEST, check=True)

# WANDB_API_KEY must be attached to this notebook under Add-ons -> Secrets; the
# API cannot do that. Without it the run logs offline instead of failing.
mode = "offline"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    mode = "online"
except Exception as e:
    print(f"no WANDB_API_KEY secret ({type(e).__name__}); W&B will run offline")
# A benchmark run is kilobytes, so keep it where it survives the session
# (offline runs can then be pushed later with `wandb sync`).
os.environ["WANDB_DIR"] = "/kaggle/working"

sys.path.insert(0, DEST)
from eval.qwen import benchmark_decode

r = benchmark_decode(results_dir=RESULTS, wandb_mode=mode)
report = {"repo_sha": sha, "wandb_mode": mode, "gpu": r.gpu, "model": r.model_name,
          "dtype": r.dtype, "prompt_tokens": r.prompt_tokens, "tokens_generated": r.tokens_generated,
          "run_times_s": r.run_times_s, "decode_tokens_per_s": r.decode_tokens_per_s,
          "sample_output": r.sample_output, "env": r.env, **r.summary()}
print(json.dumps(report, indent=2))
with open("/kaggle/working/bench_report.json", "w") as f:
    json.dump(report, f, indent=2)
