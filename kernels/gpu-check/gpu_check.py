"""Verify Kaggle GPU allocation, then write a result file to /kaggle/working."""
import json
import shutil
import subprocess

import torch

info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}

if info["cuda_available"]:
    # A real matmul, so this proves compute works and not just that a device is listed.
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    info["matmul_ok"] = bool(torch.isfinite((a @ b).sum()).item())
    info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

for path in ("/kaggle/working", "/kaggle/temp", "/kaggle/input"):
    usage = shutil.disk_usage(path) if shutil.os.path.exists(path) else None
    info[f"disk{path.replace('/kaggle', '')}_free_gb"] = round(usage.free / 1e9, 1) if usage else None

print(json.dumps(info, indent=2))
if shutil.which("nvidia-smi"):  # absent on CPU-only sessions
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

with open("/kaggle/working/result.json", "w") as f:
    json.dump(info, f, indent=2)
