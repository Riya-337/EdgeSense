"""Standalone prediction. Usage: python predict.py VIB.csv ELEC.csv [--model model.joblib]
VIB.csv: time + 5 accelerometer columns; ELEC.csv: time + 3 current + 3 voltage columns (20 kHz).
Prints one JSON line per 1 s window: {window, anomaly, confidence, latency_ms} (confidence = P(anomaly))."""
import argparse, json, sys, time, joblib, pandas as pd
from pathlib import Path
from features import FEATURE_NAMES, split_windows, window_features


DEFAULT_MODEL = Path(__file__).with_name("model.joblib")


def load_model(path=DEFAULT_MODEL):
    return joblib.load(path)["model"]


def predict_window(model, vib, cur, volt):
    """vib (5,n) cur (3,n) volt (3,n) -> (is_anomaly, P(anomaly)); importable by the decision engine.
    Uses whichever feature set the model was trained on (model.feature_names_in_)."""
    cols = list(model.feature_names_in_)
    x = pd.DataFrame([window_features(vib, cur, volt, extended=len(cols) > 15)], columns=FEATURE_NAMES[:len(cols)])
    p = float(model.predict_proba(x[cols])[0, 1])
    return p >= 0.5, p


def _read(path, channels, name):
    sig = pd.read_csv(path).iloc[:, 1:].to_numpy(float).T
    if sig.shape[0] != channels:
        sys.exit(f"{name}: expected {channels} data columns after the time column, found {sig.shape[0]}")
    return split_windows(sig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("vib"); ap.add_argument("elec"); ap.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args(); m = load_model(a.model)
    v, e = _read(a.vib, 5, "VIB.csv"), _read(a.elec, 6, "ELEC.csv")
    if not v or not e:
        sys.exit("need at least 1 s (20,000 samples) of data in each file")
    for i, (vw, ew) in enumerate(zip(v, e)):
        t0 = time.perf_counter(); flag, p = predict_window(m, vw, ew[:3], ew[3:6])
        print(json.dumps({"window": i, "anomaly": bool(flag), "confidence": round(p, 3), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}))
