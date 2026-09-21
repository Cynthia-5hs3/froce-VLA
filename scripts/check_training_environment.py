"""Audit the independent Conda prefix and record the executable training stack."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from force_vla.pi05.preparation import ROOT, write_json
from lab import isolation_report


def main():
    report = {"isolation": isolation_report(), "packages": {}, "independent_files": {}}
    if not (ROOT / "env/conda-meta/history").is_file():
        raise RuntimeError("The local prefix is not a Conda environment")
    required = {"torch": "torch", "transformers": "transformers", "lerobot": "lerobot",
                "accelerate": "accelerate", "pyarrow": "pyarrow", "av": "av",
                "safetensors": "safetensors", "numpy": "numpy", "pytest": "pytest"}
    for package, module in required.items():
        location = Path(importlib.import_module(module).__file__).resolve()
        if not location.is_relative_to(ROOT / "env"):
            raise RuntimeError(f"Foreign dependency: {location}")
        report["packages"][package] = {"version": importlib.metadata.version(package), "path": str(location)}
        original = Path("/data0/miniconda3/envs/evo-rlt") / location.relative_to(ROOT / "env")
        independent = not original.exists() or not os.path.samefile(original, location)
        report["independent_files"][package] = independent
        if not independent:
            raise RuntimeError(f"Shared environment inode: {location}")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check"] = result.stdout.strip() + result.stderr.strip()
    if result.returncode:
        raise RuntimeError(report["pip_check"])
    report["cuda_available"] = torch.cuda.is_available()
    report["torch_cuda"] = torch.version.cuda
    report["raw_31d_readonly"] = bool(os.statvfs(ROOT / "data").f_flag & os.ST_RDONLY)
    if not report["raw_31d_readonly"]:
        raise RuntimeError("31D data must be mounted read-only")
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        result = (torch.ones(8, device="cuda") * 2).sum()
        if result.item() != 16:
            raise RuntimeError("CUDA execution failed")
    destination = ROOT / "outputs/environment"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "training_environment.json", report)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)
    write_json(destination / "pip_inventory.json", json.loads(packages))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
