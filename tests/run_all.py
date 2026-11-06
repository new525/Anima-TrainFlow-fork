"""Run every test script (each in its own process) and summarise. Exit code 1 if any failed.
usage: python tests/run_all.py [name-filter ...]      e.g.  python tests/run_all.py rename e2e"""
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
scripts = sorted(TESTS.glob("test_*.py")) + sorted((TESTS / "e2e").glob("test_*.py"))
if sys.argv[1:]:
    scripts = [s for s in scripts if any(f in s.stem for f in sys.argv[1:])]
results = []
for script in scripts:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script)], cwd=script.parent, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    passed = sum(1 for line in out.splitlines() if line.startswith("PASS "))
    failed = [line for line in out.splitlines() if line.startswith("FAIL ")]
    skipped = next((line for line in out.splitlines() if line.startswith("SKIP")), None)
    status = "skip" if skipped and r.returncode == 0 else ("ok" if r.returncode == 0 else "FAILED")
    results.append((script, status))
    print(f"{status:7s} {script.relative_to(TESTS)}  ({passed} checks, {time.time() - t0:.0f}s)" + (f"  {skipped}" if status == "skip" else ""))
    for line in failed:
        print("        " + line)
    if r.returncode != 0 and not failed:
        print("\n".join("        " + line for line in out.strip().splitlines()[-15:]))
bad = [s for s, st in results if st == "FAILED"]
print(f"\n{len(results) - len(bad)}/{len(results)} test scripts passed" + (f"; failed: {', '.join(s.stem for s in bad)}" if bad else ""))
sys.exit(1 if bad else 0)
