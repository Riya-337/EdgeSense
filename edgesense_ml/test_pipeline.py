"""Self-check: python3 test_pipeline.py   (also works with pytest). Takes ~1 min."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import BASE_FEATURES, FEATURE_NAMES, window_features  # noqa: E402
import simulate_dataset as sim  # noqa: E402


def run(*args, cwd=HERE, check=True):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise AssertionError(f"{args} failed:\n{r.stdout}\n{r.stderr}")
    return r


def test_features():
    assert len(BASE_FEATURES) == 15 and len(FEATURE_NAMES) == 21 and FEATURE_NAMES[:15] == BASE_FEATURES
    f = sim.make_window("Healthy 1", (0, 1), 0.5, 1)  # 50 % speed: supply ~25 Hz
    assert f.shape == (21,) and np.isfinite(f).all()
    assert 2 < f[BASE_FEATURES.index("cur_rms")] < 8
    assert abs(f[FEATURE_NAMES.index("shaft_hz")] - 0.98 * 25) < 1.5, "shaft_hz must track supply frequency"
    bad = sim.make_window("Bearing motor BPFO", (2, 3), 1.0, 1)
    assert bad[3] > f[3] or bad[2] > f[2], "bearing fault should raise kurtosis/crest"
    rng = np.random.default_rng(0)
    v, c, u = rng.standard_normal((5, 20000)), rng.standard_normal((3, 20000)), rng.standard_normal((3, 20000))
    assert np.allclose(window_features(v, c, u, extended=False), window_features(v, c, u)[:15])
    assert np.isfinite(window_features(v, c, u)).all()


def test_simulator_deterministic_and_loader_roundtrip():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for n in "ab":
            run("simulate_dataset.py", "--out", t / f"{n}.csv", "--per-fault", 2, "--per-healthy", 2)
        assert (t / "a.csv").read_bytes() == (t / "b.csv").read_bytes(), "simulator not deterministic"
        d = pd.read_csv(t / "a.csv")
        assert list(d.columns) == FEATURE_NAMES + ["label", "fault", "severity", "speed", "condition", "hum"]
        assert not d.isna().any().any() and set(d.label) == {0, 1}

        run("make_demo_raw_files.py", t / "raw")
        out = run("build_dataset.py", t / "raw", "--out", t / "f.csv").stdout
        assert "96 windows, 12 conditions" in out and "Healthy 1" in out
        for fs in ("base", "extended"):
            run("train.py", "--csv", t / "f.csv", "--features", fs, "--model", t / f"{fs}.joblib",
                "--report", t / f"{fs}.json", "--samples", t / f"{fs}_s.csv")
            rep = json.loads((t / f"{fs}.json").read_text())
            assert rep["n_features"] == (15 if fs == "base" else 21)
            m = joblib.load(t / f"{fs}.joblib")
            assert m["features"] == rep["features"] and list(m["model"].feature_names_in_) == rep["features"]
            vib = t / "raw/Vibration/Motor 2/100%/Healthy 1/vib_m2_100_all.csv"
            ele = t / "raw/Electric/Motor 2/100%/Healthy 1/elec_m2_100_all.csv"
            lines = run(HERE / "predict.py", vib, ele, "--model", t / f"{fs}.joblib", cwd=t).stdout.strip().splitlines()
            assert len(lines) == 8
            for i, ln in enumerate(lines):
                j = json.loads(ln)
                assert set(j) == {"window", "anomaly", "confidence", "latency_ms"} and j["window"] == i
                assert isinstance(j["anomaly"], bool) and 0 <= j["confidence"] <= 1
                assert j["anomaly"] == (j["confidence"] >= 0.5)


def test_hum_option():
    base = sim.make_window("Healthy 1", (0, 1), 1.0, 5)
    hum = sim.make_window("Healthy 1", (0, 1), 1.0, 5, {"freqs": [100.0], "amp": 0.05, "bb": 0.02})
    assert hum[0] > base[0], "hum must raise vibration RMS"
    assert sim.draw_hum(1, 0.0) is None and sim.draw_hum(1, 1.0) is not None
    assert sim.draw_hum(1, 1.0) == sim.draw_hum(1, 1.0)


def test_robustness_script_runs():
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "r.json"
        run("evaluate_robustness.py", "--n", 10, "--out", out)
        rep = json.loads(out.read_text())["scenarios"]
        assert list(rep) == ["clean_healthy", "inrange_hum_healthy", "noisy_variant_healthy", "clean_faults",
                             "inrange_hum_faults", "ood_hum_healthy", "ood_hum_faults"]
        assert all(v["n"] == 10 for v in rep.values())


def test_predict_rejects_bad_input():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        pd.DataFrame({"time": [0.1, 0.2], "a": [1, 2]}).to_csv(t / "short.csv", index=False)
        pd.DataFrame(np.random.randn(20000, 4), columns="time a b c".split()).to_csv(t / "three.csv", index=False)
        r = run("predict.py", t / "short.csv", t / "short.csv", check=False)
        assert r.returncode != 0 and "expected 5 data columns" in r.stderr
        r = run("predict.py", t / "three.csv", t / "three.csv", check=False)
        assert r.returncode != 0 and "expected 5 data columns" in r.stderr
        r = run("predict.py", t / "three.csv", t / "three.csv", "--model", t / "nope.joblib", check=False)
        assert r.returncode != 0


def test_shipped_artifacts_consistent():
    d = pd.read_csv(HERE / "simulated_features.csv")
    assert len(d) >= 20000 and not d.isna().any().any()
    rep, base = (json.loads((HERE / n).read_text()) for n in ("report.json", "report_base15.json"))
    assert rep["n_windows"] == base["n_windows"] == len(d) and rep["n_conditions"] == d.condition.nunique()
    m = joblib.load(HERE / "model.joblib")
    assert m["features"] == FEATURE_NAMES == rep["features"] and list(m["model"].feature_names_in_) == FEATURE_NAMES
    s = pd.read_csv(HERE / "sample_predictions.csv")
    assert list(s.columns) == ["condition", "label", "pred", "p_anomaly"] and len(s) == 40
    # every number quoted in the README results table must match the reports
    txt = (HERE / "README.md").read_text()
    for r in (rep, base):
        for key in ("unseen_conditions_cv", "seen_conditions_cv"):
            for k in ("precision", "recall", "f1"):
                assert f"{r[key]['pooled'][k]:.3f}" in txt, (key, k)
    for k, v in rep["unseen_conditions_cv"]["false_alarm_rate_by_healthy_variant"].items():
        assert f"{v:.1%}" in txt, k
    assert f"{rep['unseen_conditions_cv']['recall_by_fault']['Bearing grease contamination']:.1%}" in txt
    # noise-robustness section: before/after report and stress test
    nohum = json.loads((HERE / "report_no_hum_aug.json").read_text())["unseen_conditions_cv"]
    assert f"{nohum['false_alarm_rate_by_healthy_variant']['Healthy noisy environment']:.1%}" in txt
    stress = json.loads((HERE / "report_robustness.json").read_text())["scenarios"]
    for v in stress.values():
        assert f"{v.get('false_alarm_rate', v.get('recall')):.1%}" in txt
    # regression guards for in-range scenarios (set from measured values with margin); OOD scenarios are reported, not asserted
    assert stress["clean_healthy"]["false_alarm_rate"] <= 0.05
    assert stress["inrange_hum_healthy"]["false_alarm_rate"] <= 0.15
    assert stress["noisy_variant_healthy"]["false_alarm_rate"] <= 0.10
    assert stress["clean_faults"]["recall"] >= 0.95
    assert stress["inrange_hum_faults"]["recall"] >= 0.90
    assert d.hum.mean() > 0.2 and set(d.hum) == {0, 1}
    for f in ("report.json", "report_base15.json", "report_no_hum_aug.json", "report_robustness.json"):
        json.loads((HERE / f).read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{f} contains {c}")))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("ok")
