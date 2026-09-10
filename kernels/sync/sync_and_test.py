"""Clone the repo from GitHub into the Kaggle session and run its test suite.

This is the round-trip check: edit locally -> push -> re-run this kernel and
the reported HEAD sha must change to match. Requires enable_internet.
"""
import json, os, shutil, subprocess, sys

REPO = "https://github.com/ShreySharma07/speculative_decoding.git"
DEST = "/kaggle/working/speculative_decoding"
BRANCH = "main"


def sh(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=isinstance(cmd, str))
    if check and p.returncode != 0:
        print(f"$ {cmd}\n{p.stdout}\n{p.stderr}", flush=True)
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.stdout.strip()


report = {}

# Always start from a clean clone; a half-written dir from a killed run is worse
# than a slow re-clone, and depth=1 makes this a couple of seconds.
if os.path.exists(DEST):
    shutil.rmtree(DEST)
sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DEST])

report["head_sha"] = sh(["git", "rev-parse", "HEAD"], cwd=DEST)
report["head_short"] = report["head_sha"][:8]
report["head_subject"] = sh(["git", "log", "-1", "--pretty=%s"], cwd=DEST)
report["head_date"] = sh(["git", "log", "-1", "--pretty=%cI"], cwd=DEST)
report["tracked_files"] = len(sh(["git", "ls-files"], cwd=DEST).splitlines())

# --no-deps: the pins already match the image, and resolving them would risk
# pip deciding to reinstall torch from PyPI and lose the CUDA build.
inst = subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
                      cwd=DEST, capture_output=True, text=True)
report["pip_install_rc"] = inst.returncode
if inst.returncode != 0:
    report["pip_stderr"] = inst.stderr[-600:]

test = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                      cwd=DEST, capture_output=True, text=True)
report["pytest_rc"] = test.returncode
report["pytest_tail"] = test.stdout.strip().splitlines()[-1] if test.stdout.strip() else ""
report["tests_passed"] = test.returncode == 0

# Prove the cloned code is importable and actually runs in this session.
try:
    sys.path.insert(0, DEST)
    from eval.run_eval import run_synthetic
    report["smoke_eval"] = run_synthetic(n_blocks=50, k=4, vocab=256)
except Exception as e:
    report["smoke_eval"] = {"ERROR": f"{type(e).__name__}: {e}"}

print(json.dumps(report, indent=2))
if not report["tests_passed"]:
    print("---- pytest output ----")
    print(test.stdout[-3000:])

with open("/kaggle/working/sync_report.json", "w") as f:
    json.dump(report, f, indent=2)
