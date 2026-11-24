"""Install Triton for MOSS-Image's ConvRot INT8 layers, then check that sd-scripts can use it.

ConvRot INT8 (the low-VRAM way to train MOSS-Image) runs through Triton kernels. Without Triton it still works, through
a fallback that keeps the VRAM saving but is slower than not quantizing at all. On Windows Triton is the separate
package triton-windows, and it has to match the installed PyTorch: every PyTorch release names the Triton version it
was built for in its own package metadata, so that is the version installed here.

usage: python tools/install_triton.py [--check-only] [--spec "<specifier>"]
  --check-only   install nothing, only report whether sd-scripts would use Triton
  --spec         version specifier to install instead of PyTorch's own, e.g. --spec "<3.8"
"""
import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SD_SCRIPTS = ROOT / "training" / "sd-scripts"


def triton_spec_of_torch(requirements):
    """The version specifier of the `triton` requirement among a package's Requires-Dist lines, e.g. '~=3.8.0'
    (the line is conditional on Linux, where pip installs it itself; the version is what matters here).
    Returns None if PyTorch names no Triton version."""
    for line in requirements or []:
        m = re.match(r"^\s*triton\s*(?:\(([^)]*)\)|([^;]*))", line)
        if m and not re.match(r"^\s*triton[-_\w]", line):
            spec = (m.group(1) or m.group(2) or "").strip()
            if spec:
                return spec.replace(" ", "")
    return None


def install(spec):
    package = f"triton-windows{spec}"
    print(f"Installing {package} ...", flush=True)
    return subprocess.run([sys.executable, "-m", "pip", "install", "-U", package]).returncode == 0


def check():
    """Ask sd-scripts itself: does its ConvRot INT8 module find a usable Triton, and does a kernel compile and run?"""
    sys.path.insert(0, str(SD_SCRIPTS))
    try:
        import torch
    except ImportError:
        print("PyTorch isn't installed in this Python yet: run Install_Requirements.bat first.")
        return False
    try:
        from library import convrot_int8_kernels as kernels
    except Exception as e:  # noqa: BLE001
        print(f"Could not load sd-scripts' ConvRot INT8 module: {type(e).__name__}: {e}")
        return False
    if not torch.cuda.is_available():
        print("No CUDA GPU is visible to PyTorch, so Triton can't be tried here (its kernels are GPU-only).")
        return False
    if not kernels.HAS_TRITON:
        try:
            import triton  # noqa: F401
            print("Triton is installed, but sd-scripts could not set it up on this GPU, so INT8 training would use the slow fallback.")
        except Exception as e:  # noqa: BLE001
            print(f"Triton can't be imported ({type(e).__name__}: {e}), so INT8 training would use the slow fallback.")
        return False
    try:
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        kernels.triton_quantize_rowwise(x)       # compiles one of the kernels sd-scripts trains with, and runs it
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"Triton is installed, but compiling a kernel failed:\n  {type(e).__name__}: {e}\n"
              "INT8 training would stop with this error. See the triton-windows README for what it needs on your system:\n"
              "  https://github.com/triton-lang/triton-windows\n"
              "Until it's sorted out: pip uninstall triton-windows (INT8 then uses the slow fallback), or untick the two\n"
              "ConvRot INT8 boxes and use more Block Swap.")
        return False
    import triton
    print(f"OK: Triton {triton.__version__} compiled and ran one of sd-scripts' INT8 kernels on {torch.cuda.get_device_name(0)}.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--spec", default=None)
    a = ap.parse_args()
    if not a.check_only:
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            print("PyTorch isn't installed in this Python yet: run Install_Requirements.bat first.")
            return 1
        spec = a.spec or triton_spec_of_torch(importlib.metadata.requires("torch"))
        print(f"PyTorch {torch_version}" + (f" was built for Triton {spec}" if spec and not a.spec else ""))
        if os.name != "nt":
            print("Not Windows: PyTorch's own installation brings Triton where it supports it. Nothing to install.")
        elif not spec:
            print("This PyTorch doesn't name a Triton version, so none is installed (a mismatched one breaks at run time).\n"
                  "Pick the triton-windows version for your PyTorch from https://github.com/triton-lang/triton-windows\n"
                  'and run:  python_embeded\\python.exe tools\\install_triton.py --spec "<the specifier, e.g. <3.8>"')
            return 1
        elif not install(spec):
            print(f"\ntriton-windows{spec} could not be installed (see pip's message above; there may be no release for this\n"
                  "PyTorch yet). MOSS-Image training still works: ConvRot INT8 uses its slower fallback, or untick INT8.")
            return 1
    print()
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
