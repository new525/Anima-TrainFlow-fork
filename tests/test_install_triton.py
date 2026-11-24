"""tools/install_triton.py (run by Install_Triton_for_MOSS.bat): which Triton version it picks from PyTorch's package
record, and what it says when PyTorch can't be used. "Not installed", "a package it needs is missing" and "it won't
load" are different problems; only the first is answered by "run Install_Requirements.bat"."""
import builtins
import importlib.metadata
import io
import sys
from contextlib import redirect_stdout

from _harness import REPO, check, finish

sys.path.insert(0, str(REPO / "tools"))
import install_triton as it  # noqa: E402

spec = it.triton_spec_of_torch
check("the Triton version comes from PyTorch's own requirement line, whatever platform that line is for",
      spec(["filelock", "sympy>=1.13.3", 'triton~=3.8.0; platform_system == "Linux" and python_version < "3.15"']) == "~=3.8.0"
      and spec(['triton (==3.5.1) ; platform_system == "Linux"']) == "==3.5.1")
check("no Triton line, a bare one, or another package with a similar name: no version is made up",
      spec(["sympy"]) is None and spec(['triton; platform_system == "Linux"']) is None and spec(["triton-windows<3.8"]) is None and spec(None) is None)

check("PyTorch's exact pin becomes one that matches triton-windows' post-releases; other forms are kept",
      it.windows_spec("==3.6.0") == "==3.6.0.*" and it.windows_spec("~=3.8.0") == "~=3.8.0" and it.windows_spec("<3.8") == "<3.8"
      and it.windows_spec("==3.6.0.*") == "==3.6.0.*")


class FakeResponse(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


asked = []
real_urlopen = it.urllib.request.urlopen
it.urllib.request.urlopen = lambda url, timeout=None: (asked.append(url), FakeResponse(
    '{"info": {"requires_dist": ["filelock", "triton==3.6.0; platform_system == \\"Linux\\""]}}'))[1]
try:
    found = it.triton_spec_from_pypi("2.11.0+cu128")
finally:
    it.urllib.request.urlopen = real_urlopen
check("a Windows CUDA build that names no Triton version: the same release's record on PyPI is asked (2.11.0+cu128 → 2.11.0)",
      found == "==3.6.0" and asked == ["https://pypi.org/pypi/torch/2.11.0/json"], (found, asked))


def offline(url, timeout=None):
    raise OSError("no network")


it.urllib.request.urlopen = offline
try:
    check("PyPI unreachable: no version is made up", it.triton_spec_from_pypi("2.11.0+cu128") is None)
finally:
    it.urllib.request.urlopen = real_urlopen

# Triton builds a C module against the running Python: the bundled Python must carry its own headers and import library
import re  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

embedded = REPO / "python_embeded"
dll_versions = set(re.findall(rb"3\.10\.\d+", (embedded / "python310.dll").read_bytes()))
header_version = re.search(r'#define PY_VERSION\s+"([^"]+)"', (embedded / "Include" / "patchlevel.h").read_text(encoding="utf-8")).group(1)
check("the bundled Python carries its headers and import library, where triton-windows looks for them",
      it.missing_build_files(embedded, "310") == [] and (embedded / "Include" / "pyconfig.h").is_file()
      and (embedded / "Include" / "cpython" / "object.h").is_file(), it.missing_build_files(embedded, "310"))
check("…and the headers are those of the bundled Python's own version", dll_versions == {header_version.encode()}, (dll_versions, header_version))
bare = Path(tempfile.mkdtemp())
check("a Python without them: both files are named", [Path(p).name for p in it.missing_build_files(bare, "310")] == ["Python.h", "python310.lib"])

real_import = builtins.__import__


def with_torch_import(outcome, fn):
    """Run fn() while `import torch` does `outcome` (an exception to raise)."""
    saved = sys.modules.pop("torch", None)

    def fake(name, *a, **k):
        if name == "torch":
            raise outcome
        return real_import(name, *a, **k)
    builtins.__import__ = fake
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            result = fn()
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["torch"] = saved
    return result, out.getvalue()


missing = ModuleNotFoundError("No module named 'torch'", name="torch")
(mod, msg), _ = with_torch_import(missing, it.load_torch)
check("PyTorch not installed: said so, naming the Python that was used and its packages folder",
      mod is None and "isn't installed in the Python this script runs with" in msg and sys.executable in msg and "installed)" in msg, msg)
dll = ImportError("DLL load failed while importing _C: The specified module could not be found.")
(mod, msg), _ = with_torch_import(dll, it.load_torch)
check("PyTorch installed but failing to load: the real error is shown, and it isn't blamed on Install_Requirements.bat",
      mod is None and "won't load" in msg and "DLL load failed while importing _C" in msg and "not fixed by Install_Requirements.bat" in msg, msg)
(mod, msg), _ = with_torch_import(OSError("[WinError 126] The specified module could not be found"), it.load_torch)
check("…the same for an OSError from a missing DLL", mod is None and "WinError 126" in msg, msg)
dep = ModuleNotFoundError("No module named 'typing_extensions'", name="typing_extensions")
(mod, msg), _ = with_torch_import(dep, it.load_torch)
check("PyTorch installed but one of its own imports missing: that package is named",
      mod is None and "a package it needs is missing: typing_extensions" in msg, msg)

ok, said = with_torch_import(dll, it.check)
check("the check reports that error instead of asking for Install_Requirements.bat", ok is False and "DLL load failed" in said
      and "run Install_Requirements.bat first" not in said, said)


# no package record for torch, and torch doesn't load either: main() reports the load error and installs nothing
def no_record(name):
    raise importlib.metadata.PackageNotFoundError(name)


real_version, real_argv, installs = importlib.metadata.version, sys.argv, []
importlib.metadata.version, sys.argv, it.install = no_record, ["install_triton.py"], lambda s: (installs.append(s), True)[1]
try:
    code, said = with_torch_import(dll, it.main)
finally:
    importlib.metadata.version, sys.argv = real_version, real_argv
check("no package record and PyTorch won't load: the load error is what's reported, nothing is installed",
      code == 1 and "DLL load failed" in said and installs == [], (code, said))

# Triton's INT8 kernels only compile for RTX 30 series and newer GPUs (compute capability 8.0+)
import os  # noqa: E402
import types  # noqa: E402

check("a 2080 Ti (compute capability 7.5) is too old, and is told so by name; 8.0, 8.6, 8.9 and 12.0 are fine",
      "RTX 2080 Ti has compute capability 7.5" in (it.gpu_too_old((7, 5), "RTX 2080 Ti") or "") and "RTX 30 series" in it.gpu_too_old((7, 5))
      and it.gpu_too_old((6, 1)) is not None and [it.gpu_too_old(c) for c in ((8, 0), (8, 6), (8, 9), (12, 0))] == [None] * 4,
      it.gpu_too_old((7, 5), "RTX 2080 Ti"))
sys.path.insert(0, str(REPO / "training" / "sd-scripts" / "library"))
kernels_source = (REPO / "training" / "sd-scripts" / "library" / "convrot_int8_kernels.py").read_text(encoding="utf-8")
check("the installer and the trainer draw the line at the same compute capability",
      f"TRITON_MIN_CAPABILITY = {it.MIN_CAPABILITY}" in kernels_source, it.MIN_CAPABILITY)


def fake_torch(capability, name="NVIDIA GeForce RTX 2080 Ti", available=True):
    cuda = types.SimpleNamespace(is_available=lambda: available, get_device_capability=lambda: capability, get_device_name=lambda i: name)
    return types.SimpleNamespace(cuda=cuda, __version__="2.11.0+cu128")


def said_by(fn):
    out = io.StringIO()
    with redirect_stdout(out):
        result = fn()
    return result, out.getvalue()


real_installed = it.triton_installed
it.triton_installed = lambda: None
old, said = said_by(lambda: it.old_gpu_notice(fake_torch((7, 5))))
check("an old GPU without Triton: the reason is given, and no uninstall advice", old is True and "compute capability 7.5" in said
      and "uninstall" not in said, said)
it.triton_installed = lambda: "triton-windows 3.6.0.post26"
old, said = said_by(lambda: it.old_gpu_notice(fake_torch((7, 5))))
check("an old GPU with Triton already installed: how to remove it for 1.16 and older, with this Python",
      old is True and "triton-windows 3.6.0.post26 is installed" in said and f'"{sys.executable}" -m pip uninstall -y triton-windows' in said, said)
new, said = said_by(lambda: it.old_gpu_notice(fake_torch((8, 6), "NVIDIA GeForce RTX 3090")))
none, said2 = said_by(lambda: it.old_gpu_notice(fake_torch((7, 5), available=False)))
check("a new enough GPU, or no GPU at all: nothing is said here", (new, said, none, said2) == (False, "", False, ""), (new, said, none, said2))


class WindowsOs:
    """`os` as the installer sees it on Windows: only the name differs."""
    name = "nt"

    def __getattr__(self, attr):
        return getattr(os, attr)


real_load, real_os, real_argv = it.load_torch, it.os, sys.argv
it.os, sys.argv = WindowsOs(), ["install_triton.py"]
del installs[:]
try:
    it.load_torch = lambda: (fake_torch((7, 5)), None)
    code, said = said_by(it.main)
    check("on Windows with an old GPU: nothing is installed, and that isn't reported as a failure",
          code == 0 and installs == [] and "Nothing was installed." in said and "compute capability 7.5" in said, (code, installs, said))
    sys.argv = ["install_triton.py", "--spec", "==3.6.0.*"]
    code, said = said_by(it.main)
    check("…also when a version is given with --spec", code == 0 and installs == [], (code, installs))
finally:
    it.load_torch, it.os, sys.argv, it.triton_installed = real_load, real_os, real_argv, real_installed
finish()
