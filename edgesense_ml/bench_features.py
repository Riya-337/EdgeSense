"""Time feature extraction + prediction per 1 s window: python3 bench_features.py"""
import time, joblib, numpy as np
from features import window_features
from predict import predict_window

rng = np.random.default_rng(0)
vib, cur, volt = rng.standard_normal((5, 20000)), 5 * np.sin(2 * np.pi * 50 * np.arange(20000) / 20000) * np.ones((3, 1)), 300 * np.ones((3, 1)) * np.sin(2 * np.pi * 50 * np.arange(20000) / 20000)
m = joblib.load("model.joblib")["model"]

def ms(fn, n=100):
    fn(); t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

base, ext = ms(lambda: window_features(vib, cur, volt, False)), ms(lambda: window_features(vib, cur, volt, True))
print(f"features  base(15): {base:.1f} ms   extended(21): {ext:.1f} ms   extra cost: {ext - base:.1f} ms")
print(f"end-to-end predict_window (extended model): {ms(lambda: predict_window(m, vib, cur[:], volt[:])):.1f} ms")
