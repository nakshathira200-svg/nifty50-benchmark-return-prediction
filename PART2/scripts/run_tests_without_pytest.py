"""Minimal test runner for environments where pytest is not installed (e.g. an offline sandbox).
Normal usage on your machine is simply:  pytest -q
Runs tests/test_*.py; honours unittest.SkipTest; stubs `yfinance` ONLY if it is missing (Part 1's module imports it
at the top, but the tests never download anything)."""
import importlib.util, sys, types, traceback, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec("yfinance") is None:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
    print("[note] yfinance not installed -> stubbed for import only")

passed = failed = skipped = 0
for f in sorted((ROOT / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)(); passed += 1; print(f"PASS  {f.stem}::{name}")
        except unittest.SkipTest as e:
            skipped += 1; print(f"SKIP  {f.stem}::{name}  ({e})")
        except Exception:
            failed += 1; print(f"FAIL  {f.stem}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
sys.exit(1 if failed else 0)
