"""Physics-inspired SIMULATED centrifugal-pump dataset (NOT real measurements).

Mirrors the NLN-EMP fault list / severities / speeds (Bruinsma et al., Data in Brief 2024) so the same
feature pipeline can later be pointed at the real files. Signals are synthesised at 20 kHz, windowed at
1 s, and reduced to the features of features.py. Deterministic for a given --seed. Raw waveforms are discarded to keep the output small.
"""
import argparse, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import fftconvolve
from features import window_features, FEATURE_NAMES, FS

N = FS
T = np.arange(N) / FS
SPEEDS = {"50%": 0.5, "75%": 0.75, "100%": 1.0}
FR100, VANES = 49.0, 5            # assumed shaft Hz at rated speed, impeller vanes (assumption)

# fault family -> (number of severities, kind)
FAULTS = {
    "Bearing motor BPFO": (3, "brg"), "Bearing motor BPFI": (3, "brg"), "Bearing pump BPFI+BPFO": (3, "brg_pump"),
    "Bearing motor BSF": (1, "brg"), "Bearing grease contamination": (1, "grease"),
    "Misalignment angular": (5, "mis"), "Misalignment parallel": (4, "mis"), "Misalignment combination": (4, "mis"),
    "Unbalance motor": (6, "unb"), "Unbalance pump": (3, "unb_pump"), "Loose foot": (2, "loose"),
    "Coupling degradation": (4, "coupling"), "Cavitation": (9, "cav"), "Impeller damage": (3, "imp"),
    "Stator winding short": (2, "stator"), "Broken rotor bar": (1, "rotor"), "Bent shaft": (1, "bent"),
    "Soft foot": (2, "loose"),
}
BPF = {"BPFO": 3.05, "BPFI": 4.95, "BSF": 2.0}
HEALTHY = ["Healthy 1", "Healthy 2", "Healthy 3", "Healthy noisy environment"]


def tone(f, a, ph=None, rng=None):
    return a * np.sin(2 * np.pi * f * T + (rng.uniform(0, 6.28) if ph is None else ph))


def impacts(rng, rate, amp, fres=3000.0, tau=0.0006):
    n = int(rate)
    pos = (np.arange(n) * N / rate + rng.normal(0, 0.01 * N / rate, n) + rng.uniform(0, N / rate)).astype(int)
    pos = pos[(pos >= 0) & (pos < N)]
    x = np.zeros(N); x[pos] = amp * rng.uniform(0.6, 1.4, pos.size)
    k = np.arange(300) / FS
    return fftconvolve(x, np.exp(-k / tau) * np.sin(2 * np.pi * fres * k), mode="same")


def make_window(fault, sev, spd, seed, hum=None):
    """hum: optional nuisance from neighbouring machines {freqs, amp, bb}, independent of the pump's fault state."""
    rng = np.random.default_rng(seed)
    s = spd
    fr = FR100 * s
    fl = 50.0 * s
    healthy = fault in HEALTHY
    kind = FAULTS[fault][1] if not healthy else None
    level = 0.0 if healthy else (sev[0] + 1) / sev[1]       # 0..1 severity position
    sev_amp = 0.12 + 0.88 * level                            # mild severities sit close to healthy
    noisy = fault == "Healthy noisy environment"

    # ---- vibration (5 accelerometers: 0-2 motor side, 3-4 pump side) ----
    base = 0.04 * rng.uniform(0.85, 1.2) * (0.6 + 0.8 * s)
    sig = tone(fr, 0.07 * s * rng.uniform(0.7, 1.3), rng=rng) + tone(fr * VANES, 0.015 * s, rng=rng)
    common = np.zeros(N)
    side_gain = np.array([1.0, 1.0, 0.9, 0.5, 0.4]) if kind not in ("brg_pump", "unb_pump") else np.array([0.4, 0.4, 0.5, 1, 1.0])
    nz = 0.0
    if hum:
        sig = sig + sum(tone(fq, hum["amp"], rng=rng) for fq in hum["freqs"])
        nz += hum["bb"]
    if noisy:
        nz = 0.03
        sig = sig + tone(100, 0.03, rng=rng) + tone(150, 0.02, rng=rng)
    if kind in ("brg", "brg_pump"):
        key = [k for k in BPF if k in fault]
        r = BPF[key[0]] if len(key) == 1 else rng.choice([3.05, 4.95])
        common = impacts(rng, r * fr, 0.02 + 0.25 * sev_amp * (0.5 + s))
    elif kind == "grease":
        nz += 0.02 * sev_amp; common = impacts(rng, 3.05 * fr * 2, 0.02 * sev_amp)
    elif kind in ("unb", "unb_pump"):
        common = tone(fr, 0.35 * sev_amp * s ** 1.5, rng=rng)
    elif kind == "mis":
        common = tone(2 * fr, 0.25 * sev_amp * s, rng=rng) + tone(fr, 0.1 * sev_amp * s, rng=rng)
    elif kind == "loose":
        common = sum(tone(k * fr, 0.12 * sev_amp * s / k, rng=rng) for k in (0.5, 2, 3, 4)) + impacts(rng, fr, 0.04 * sev_amp)
    elif kind == "coupling":
        common = tone(2 * fr, 0.1 * sev_amp, rng=rng) + tone(fr, 0.06 * sev_amp, rng=rng); nz += 0.02 * sev_amp
    elif kind == "cav":
        nz += 0.12 * sev_amp * s; common = impacts(rng, 400, 0.03 * sev_amp, fres=6000.0, tau=0.0003)
    elif kind == "imp":
        common = tone(fr * VANES, 0.15 * sev_amp * s, rng=rng) + tone(fr * (VANES - 1), 0.08 * sev_amp * s, rng=rng) + tone(fr, 0.1 * sev_amp, rng=rng)
    elif kind == "bent":
        common = tone(fr, 0.2 * sev_amp, rng=rng) + tone(2 * fr, 0.12 * sev_amp, rng=rng)
    elif kind in ("stator", "rotor"):
        common = tone(2 * fl, 0.05 * sev_amp, rng=rng)
    vib = np.stack([g * rng.uniform(0.85, 1.15) * (sig + common) + (base + nz) * rng.standard_normal(N) for g in side_gain])

    # ---- motor current / voltage (3 phases) ----
    amp = 4.0 + 3.0 * s ** 2 * rng.uniform(0.9, 1.1)
    slip = 0.02 * rng.uniform(0.8, 1.2)
    mod = 0.002 + (0.012 * sev_amp if kind in ("unb", "unb_pump", "mis", "loose", "coupling", "bent", "imp", "cav") else 0)
    if kind == "brg": mod += 0.004 * sev_amp
    ph0 = rng.uniform(0, 6.28)
    cur = np.empty((3, N)); volt = np.empty((3, N))
    unb = 1 + rng.normal(0, 0.004, 3)   # same natural phase imbalance for every window (no label leak)
    if kind == "stator": unb = unb * np.array([1 + 0.06 * sev_amp, 1 - 0.03 * sev_amp, 1.0])
    for k in range(3):
        p = ph0 - k * 2 * np.pi / 3
        i = amp * unb[k] * np.sin(2 * np.pi * fl * T + p) * (1 + mod * np.sin(2 * np.pi * fr * T + rng.uniform(0, 6.28)))
        i += 0.01 * amp * np.sin(2 * np.pi * 5 * fl * T + 5 * p) + 0.006 * amp * np.sin(2 * np.pi * 7 * fl * T + 7 * p)
        if kind == "rotor":
            sb = 0.02 * sev_amp * amp
            i += sb * np.sin(2 * np.pi * fl * (1 - 2 * slip) * T + p) + sb * np.sin(2 * np.pi * fl * (1 + 2 * slip) * T + p)
        if kind == "stator":
            i += 0.01 * sev_amp * amp * np.sin(2 * np.pi * 2 * fl * T + 2 * p)
        cur[k] = i + 0.04 * rng.standard_normal(N)
        volt[k] = 325 * s * (1 + rng.normal(0, 0.003)) * np.sin(2 * np.pi * fl * T + p) + 1.0 * rng.standard_normal(N)
    return window_features(vib, cur, volt)


def draw_hum(seed, prob):
    """Random mains-harmonic hum + broadband noise with probability prob (own RNG stream: does not alter other signals)."""
    hr = np.random.default_rng(seed + 5_000_003)
    if hr.random() >= prob:
        return None
    return {"freqs": [float(f) for f in hr.choice([50, 100, 150, 200], size=int(hr.integers(1, 4)), replace=False)],
            "amp": float(hr.uniform(0.01, 0.05)), "bb": float(hr.uniform(0, 0.03))}


def run_chunk(a):
    fault, sev, spd_name, start, count, seed0, hum_prob = a
    rows = []
    for j in range(start, start + count):
        hum = draw_hum(seed0 + j, hum_prob)
        f = make_window(fault, sev, SPEEDS[spd_name], seed0 + j, hum)
        rows.append([*f, 0 if fault in HEALTHY else 1, fault, sev[0] + 1 if fault not in HEALTHY else 0, spd_name,
                     f"{fault}|sev{sev[0] + 1 if fault not in HEALTHY else 0}|{spd_name}",
                     int(hum is not None or fault == "Healthy noisy environment")])
    return rows


def main(out, per_fault, per_healthy, seed, hum_prob):
    tasks, sd = [], seed
    for fault, (n, _) in FAULTS.items():
        for si in range(n):
            for spd in SPEEDS:
                for st in range(0, per_fault, 25):
                    tasks.append((fault, (si, n), spd, st, min(25, per_fault - st), sd, hum_prob)); sd += 100000
    for h in HEALTHY:
        for spd in SPEEDS:
            for st in range(0, per_healthy, 25):
                tasks.append((h, (0, 1), spd, st, min(25, per_healthy - st), sd, hum_prob)); sd += 100000
    with Pool() as p:
        rows = [r for chunk in p.imap(run_chunk, tasks, chunksize=4) for r in chunk]  # imap keeps task order -> deterministic rows
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label", "fault", "severity", "speed", "condition", "hum"])
    df.to_csv(out, index=False)
    print(f"{len(df)} windows | {df.condition.nunique()} conditions | anomalous share {df.label.mean():.2f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="simulated_features.csv")
    ap.add_argument("--per-fault", type=int, default=100, help="windows per fault/severity/speed")
    ap.add_argument("--per-healthy", type=int, default=450, help="windows per healthy variant/speed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hum-prob", type=float, default=0.25, help="share of windows (any class) with random neighbouring-machine hum; 0 = off")
    a = ap.parse_args()
    main(a.out, a.per_fault, a.per_healthy, a.seed, a.hum_prob)
