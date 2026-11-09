"""Run `python check_setup.py` to verify that every file of the project is in place."""
import sys
from pathlib import Path

import config as cfg

missing = []
for s in cfg.SUBSETS:
    for kind in ("train", "test", "RUL"):
        f = cfg.RAW_DIR / f"{kind}_{s}.txt"
        if not f.exists():
            missing.append(("NASA dataset file", f))
    for f in (cfg.MODELS_DIR / f"{s}_rul_model.joblib", cfg.MODELS_DIR / f"{s}_risk_model.joblib",
              cfg.METRICS_DIR / f"{s}_summary.json"):
        if not f.exists():
            missing.append(("trained file", f))
    for k in ("train", "val", "test"):
        f = cfg.PREDICTIONS_DIR / f"{s}_{k}_predictions.csv.gz"
        if not f.exists():
            missing.append(("predictions file", f))
for name in ("app.py", "requirements.txt", "README.md"):
    if not (cfg.ROOT / name).exists():
        missing.append(("project file", cfg.ROOT / name))

if missing:
    print("MISSING FILES:")
    for kind, f in missing:
        print(f"  [{kind}] {f}")
    print("\nIf these are model/prediction files, extract ALL the ZIP parts into the same EngineHealth folder.")
    print("If these are NASA files, copy the 12 train_/test_/RUL_ .txt files into dataset\\raw\\")
    sys.exit(1)
print("All files present. Start the website with:  streamlit run app.py")
