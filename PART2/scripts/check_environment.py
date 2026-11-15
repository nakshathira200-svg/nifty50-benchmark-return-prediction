"""Report which Part 2 frameworks are importable here (does NOT install anything)."""
import importlib, sys
MODS = {"xgboost": "xgboost", "tensorflow": "tensorflow", "keras": "keras", "fastai": "fastai", "torch": "torch",
        "pytorch_tabnet": "pytorch-tabnet", "pytorch_lightning": "pytorch-lightning", "flash": "lightning-flash"}
print("python", sys.version.split()[0])
for m, n in MODS.items():
    try:
        print(f"OK       {n:18s}", getattr(importlib.import_module(m), "__version__", "?"))
    except Exception as e:
        print(f"MISSING  {n:18s}", type(e).__name__, str(e)[:80])
