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
import json
import os
import re
import subprocess
import sys
import urllib.request
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


def triton_spec_from_pypi(torch_version: str, timeout: float = 20.0):
    """The Triton requirement of this PyTorch release as published on PyPI. PyTorch's CUDA builds for Windows (the
    +cu128 wheels) carry no Triton line of their own, but the same release on PyPI names the version it was built
    for. Returns None when PyPI can't be reached or names none."""
    public = torch_version.split("+")[0]
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/torch/{public}/json", timeout=timeout) as r:
            return triton_spec_of_torch(json.load(r)["info"].get("requires_dist"))
    except Exception:  # noqa: BLE001
        return None


def windows_spec(spec: str) -> str:
    """PyTorch's Triton requirement as one for triton-windows, whose releases are post-releases of the same
    numbers (3.6.0.post26): '==3.6.0' has to become '==3.6.0.*' to match them. Other forms already do."""
    m = re.fullmatch(r"==\s*(\d+(?:\.\d+)*)", spec.strip())
    return f"=={m.group(1)}.*" if m else spec.strip()


def this_python() -> str:
    """Which Python is running and how many packages it has, so "not installed" can be told from "wrong folder"."""
    import site
    try:
        folders = site.getsitepackages()
    except Exception:  # noqa: BLE001
        folders = []
    count = sum(1 for f in folders if Path(f).is_dir() for d in Path(f).iterdir() if d.name.endswith(".dist-info"))
    return (f"  This Python:     {sys.executable}\n"
            f"  Its packages:    {', '.join(folders) or '(none found)'}  ({count} installed)")


def load_torch():
    """Import PyTorch and say what really happened: (module, None) or (None, message). "Not installed", "installed
    but one of its own imports is missing" and "installed but it won't load" (a DLL, usually) are different problems
    with different fixes, and only the first is solved by Install_Requirements.bat."""
    try:
        import torch
        return torch, None
    except ModuleNotFoundError as e:
        if e.name == "torch":
            return None, ("PyTorch isn't installed in the Python this script runs with.\n" + this_python() + "\n"
                          "  Install_Requirements.bat installs it into python_embeded next to this folder's app.py. If you ran it,\n"
                          "  check that it was the one in this same folder, and that its step [1/3] ended without an error.")
        return None, (f"PyTorch is installed, but a package it needs is missing: {e.name}\n" + this_python() + "\n"
                      "  Re-running Install_Requirements.bat should bring it back; if step [2/3] ends in an error, that error is the cause.")
    except Exception as e:  # noqa: BLE001  (ImportError: DLL load failed…, OSError: [WinError 126]…)
        return None, (f"PyTorch is installed but won't load:\n  {type(e).__name__}: {e}\n" + this_python() + "\n"
                      "  This is not fixed by Install_Requirements.bat. A DLL error usually means the Microsoft Visual C++\n"
                      "  Redistributable is missing or the NVIDIA driver is too old for this PyTorch build. Does start_trainer.bat open the app?")


def install(spec):
    package = f"triton-windows{spec}"
    print(f"Installing {package} ...", flush=True)
    return subprocess.run([sys.executable, "-m", "pip", "install", "-U", package]).returncode == 0


def check():
    """Ask sd-scripts itself: does its ConvRot INT8 module find a usable Triton, and does a kernel compile and run?"""
    torch, problem = load_torch()
    if torch is None:
        print(problem)
        return False
    sys.path.insert(0, str(SD_SCRIPTS))   # after PyTorch is loaded: nothing in sd-scripts can stand in for its imports
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
            requirements = importlib.metadata.requires("torch")
        except importlib.metadata.PackageNotFoundError:
            # No package record. That alone doesn't mean PyTorch is missing: ask PyTorch itself.
            torch, problem = load_torch()
            if torch is None:
                print(problem)
                return 1
            torch_version, requirements = torch.__version__, []
            print("(PyTorch loads, but its package record wasn't found, so it can't say which Triton it was built for.)")
        spec, source = a.spec, ""
        if not spec:
            spec, source = triton_spec_of_torch(requirements), "its own package record"
            if not spec and os.name == "nt":
                spec, source = triton_spec_from_pypi(torch_version), f"PyPI's record of PyTorch {torch_version.split('+')[0]}"
            spec = windows_spec(spec) if spec else spec
        print(f"PyTorch {torch_version}" + (f" was built for Triton {spec} (from {source})" if spec and not a.spec else ""))
        if os.name != "nt":
            print("Not Windows: PyTorch's own installation brings Triton where it supports it. Nothing to install.")
        elif not spec:
            print("Neither this PyTorch's package record nor PyPI (not reachable?) names a Triton version for it, so none is\n"
                  "installed: a mismatched one breaks at run time.\n"
                  "Pick the triton-windows version for your PyTorch from https://github.com/triton-lang/triton-windows\n"
                  'and run:  python_embeded\\python.exe tools\\install_triton.py --spec "==3.6.0.*"   (with your version)')
            return 1
        elif not install(spec):
            print(f"\ntriton-windows{spec} could not be installed (see pip's message above; there may be no release for this\n"
                  "PyTorch yet). MOSS-Image training still works: ConvRot INT8 uses its slower fallback, or untick INT8.")
            return 1
    print()
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
