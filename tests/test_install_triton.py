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
finish()
