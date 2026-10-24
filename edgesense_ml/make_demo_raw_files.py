"""Writes a tiny demo tree in the NLN-EMP-style raw layout (Vibration|Electric/<motor>/<speed>/<condition>/*.csv).
Only used to exercise build_dataset.py in tests; NOT used for the shipped model. Usage: python make_demo_raw_files.py OUT_DIR"""
import sys, numpy as np, pandas as pd
from pathlib import Path

FS, SECS = 20000, 8
rng = np.random.default_rng(0)
t = np.arange(1, FS * SECS + 1) / FS

CONDS = {  # name -> (is_fault, severity strength)
    "Healthy 1": 0, "Healthy 2": 0, "Healthy 3": 0,
    "Bearing BPFO sev1": 1, "Bearing BPFO sev2": 2, "Bearing BPFO sev3": 3,
    "Unbalance motor sev1": 1, "Unbalance motor sev2": 2, "Unbalance motor sev3": 3,
    "Cavitation inlet sev1": 1, "Cavitation inlet sev2": 2, "Impeller damage sev1": 1,
}

def vib(level, kind):
    out = []
    for _ in range(5):
        x = 0.05 * rng.standard_normal(t.size) + 0.1 * np.sin(2 * np.pi * 25 * t)
        if level:
            if kind.startswith("Bearing"):
                imp = (rng.random(t.size) < 0.002 * level) * rng.standard_normal(t.size) * 1.5
                x += np.convolve(imp, np.exp(-np.arange(60) / 10), "same")
            elif kind.startswith("Unbalance"):
                x += 0.08 * level * np.sin(2 * np.pi * 25 * t)
            else:
                x += 0.05 * level * rng.standard_normal(t.size)
        out.append(x)
    return np.stack(out, 1)

def elec(level):
    ph = [0, -2 * np.pi / 3, 2 * np.pi / 3]
    cur = np.stack([5 * np.sin(2 * np.pi * 50 * t + p) * (1 + 0.01 * level * np.sin(2 * np.pi * 5 * t))
                    + 0.05 * rng.standard_normal(t.size) for p in ph], 1)
    if level:
        cur[:, 0] *= 1 + 0.02 * level
    volt = np.stack([325 * np.sin(2 * np.pi * 50 * t + p) + rng.standard_normal(t.size) for p in ph], 1)
    return np.hstack([cur, volt])

root = Path(sys.argv[1]); 
for name, lvl in CONDS.items():
    d = root / "Vibration" / "Motor 2" / "100%" / name
    e = root / "Electric" / "Motor 2" / "100%" / name
    d.mkdir(parents=True, exist_ok=True); e.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.c_[t, vib(lvl, name)], columns=["time"] + [f"ch{i}" for i in range(1, 6)]).to_csv(d / "vib_m2_100_all.csv", index=False)
    pd.DataFrame(np.c_[t, elec(lvl)], columns=["time"] + [f"ch{i}" for i in range(1, 7)]).to_csv(e / "elec_m2_100_all.csv", index=False)
print("wrote", root)
