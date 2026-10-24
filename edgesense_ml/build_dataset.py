"""Walk an NLN-EMP-style tree, window the signals, extract features -> features.csv.

Assumed layout (from the dataset paper; UNVERIFIED against the real files): <root>/{Vibration|Electric}/<...>/<condition folder>/*.csv
CSV: first column time, remaining columns channels (vibration 5; electrical 6 = 3 current then 3 voltage).
If channels are in separate single-column files, files sharing a stem (minus trailing channel number) are merged.
Label: 'healthy' anywhere in the path (case-insensitive) -> 0, else 1. The assigned label of every condition is printed: CHECK IT.
Vibration and electrical were NOT time-synchronised in the dataset, so windows of the same condition are paired at random
(seeded); evaluation groups by condition to prevent leakage. Conditions are processed one at a time to bound memory.
"""
import argparse, re, numpy as np, pandas as pd
from pathlib import Path
from features import FEATURE_NAMES, split_windows, window_features


def load_recordings(folder):
    """Return list of (n_ch, n) arrays for every recording found directly in folder."""
    groups = {}
    for f in sorted(folder.glob("*.csv")):
        data = pd.read_csv(f).iloc[:, 1:].to_numpy(dtype=float).T
        if data.shape[0] > 1:
            groups[f.stem] = data
        else:
            key = re.sub(r"[_\-\s]*(ch(annel)?)?[_\-\s]*\d+$", "", f.stem, flags=re.I)
            groups.setdefault(key, []).append((f.stem, data[0]))
    return [v if isinstance(v, np.ndarray) else np.stack([d for _, d in sorted(v)]) for v in groups.values()]


def condition_dirs(root, modality):
    """{condition path parts below the modality folder: folder}"""
    return {tuple(d.relative_to(root / modality).parts): d for d in sorted((root / modality).rglob("*"))
            if d.is_dir() and any(d.glob("*.csv"))}


def windows_of(folder, n_channels):
    wins = [w for rec in load_recordings(folder) for w in split_windows(rec)]
    return [w for w in wins if w.shape[0] == n_channels], len(wins) - sum(w.shape[0] == n_channels for w in wins)


def main(root, out, max_windows, seed=0):
    root, rng, rows = Path(root), np.random.default_rng(seed), []
    vib, ele = condition_dirs(root, "Vibration"), condition_dirs(root, "Electric")
    for only, name in ((set(vib) - set(ele), "Vibration"), (set(ele) - set(vib), "Electric")):
        for c in sorted(only):
            print(f"WARNING: condition {'/'.join(c)} only has {name} data, skipped")
    pick = lambda n_avail, n: rng.permutation(n_avail)[:n] if n_avail >= n else rng.choice(n_avail, n)
    for cond in sorted(set(vib) & set(ele)):
        v, dv = windows_of(vib[cond], 5)
        e, de = windows_of(ele[cond], 6)
        if dv or de:
            print(f"WARNING: {'/'.join(cond)}: dropped {dv} vibration / {de} electrical windows with unexpected channel count")
        if not v or not e:
            print(f"WARNING: condition {'/'.join(cond)} has no usable windows, skipped"); continue
        n = min(max(len(v), len(e)), max_windows)
        label = 0 if "healthy" in "/".join(cond).lower() else 1
        for a, b in zip(pick(len(v), n), pick(len(e), n)):
            f = window_features(v[a], e[b][:3], e[b][3:6])
            rows.append([*f, label, cond[-1], "n/a", cond[1] if len(cond) > 2 else "n/a", "/".join(cond)])
    if not rows:
        raise SystemExit("no usable windows found: check the folder layout (see module docstring)")
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label", "fault", "severity", "speed", "condition"])
    df.to_csv(out, index=False)
    print(df.groupby("condition").label.first().to_string())
    print(f"{len(df)} windows, {df.condition.nunique()} conditions, anomalous share {df.label.mean():.2f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out", default="features.csv")
    ap.add_argument("--max-windows", type=int, default=200, help="per condition")
    a = ap.parse_args(); main(a.root, a.out, a.max_windows)
