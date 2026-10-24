"""Stress test of model.joblib on FRESH simulated windows (seeds >= 900,000,000; training used < 100,000,000)
under neighbouring-machine hum. Scenarios 'inrange_*' use the hum range seen in training; 'ood_*' use hum OUTSIDE it
(60/120/180 Hz mains, 2x louder), reported as a limit, not a pass/fail target.
Usage: python evaluate_robustness.py [--n 600] [--out report_robustness.json]"""
import argparse, json, joblib, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
import simulate_dataset as sim
from features import FEATURE_NAMES

SEED0 = 900_000_000
HEALTHY = ["Healthy 1", "Healthy 2", "Healthy 3"]
FAULTS = list(sim.FAULTS)


def ood_hum(seed):
    r = np.random.default_rng(seed + 7_000_001)
    return {"freqs": [60.0, 120.0, 180.0][: int(r.integers(1, 4))], "amp": float(r.uniform(0.05, 0.10)), "bb": float(r.uniform(0.03, 0.06))}


# name -> (class of windows, hum generator or None)
SCENARIOS = {
    "clean_healthy": ("healthy", lambda s: None),
    "inrange_hum_healthy": ("healthy", lambda s: sim.draw_hum(s, 1.0)),
    "noisy_variant_healthy": ("noisy", lambda s: None),
    "clean_faults": ("fault", lambda s: None),
    "inrange_hum_faults": ("fault", lambda s: sim.draw_hum(s, 1.0)),
    "ood_hum_healthy": ("healthy", ood_hum),
    "ood_hum_faults": ("fault", ood_hum),
}


def one(task):
    fault, sev, spd, seed, hum = task
    return sim.make_window(fault, sev, spd, seed, hum)


def main(n, out):
    model = joblib.load(Path(__file__).with_name("model.joblib"))["model"]
    rng, tasks, meta = np.random.default_rng(SEED0), [], []
    for si, (name, (kind, hum_fn)) in enumerate(SCENARIOS.items()):
        for i in range(n):
            seed = SEED0 + si * 1_000_000 + i
            spd = float(rng.choice(list(sim.SPEEDS.values())))
            if kind == "fault":
                fault = FAULTS[int(rng.integers(len(FAULTS)))]
                sev = (int(rng.integers(sim.FAULTS[fault][0])), sim.FAULTS[fault][0])
            else:
                fault, sev = ("Healthy noisy environment" if kind == "noisy" else HEALTHY[int(rng.integers(3))]), (0, 1)
            tasks.append((fault, sev, spd, seed, hum_fn(seed))); meta.append((name, kind, fault))
    with Pool() as p:
        X = pd.DataFrame(p.map(one, tasks, chunksize=20), columns=FEATURE_NAMES)
    pred = model.predict(X[list(model.feature_names_in_)])
    df = pd.DataFrame(meta, columns=["scenario", "kind", "fault"]).assign(flagged=pred)
    rep = {"n_per_scenario": n, "model": "model.joblib", "scenarios": {}}
    for name, g in df.groupby("scenario", sort=False):
        healthy = g.kind != "fault"
        rate = float(g.flagged.mean())
        entry = {"kind": g.kind.iloc[0], "n": int(len(g)),
                 ("false_alarm_rate" if healthy.iloc[0] else "recall"): round(rate, 4)}
        if not healthy.iloc[0]:
            fam = g.groupby("fault").flagged.agg(["mean", "size"])
            entry["lowest_family_recall"] = {k: round(float(v), 3) for k, v in fam["mean"].nsmallest(3).items()}
        rep["scenarios"][name] = entry
    json.dump(rep, open(out, "w"), indent=2)
    for k, v in rep["scenarios"].items():
        print(f"{k:24s} n={v['n']:4d}  " + (f"false-alarm {v['false_alarm_rate']:.1%}" if "false_alarm_rate" in v else f"recall {v['recall']:.1%}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--out", default="report_robustness.json")
    a = ap.parse_args(); main(a.n, a.out)
