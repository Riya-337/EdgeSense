# EdgeSense – ML & pump data (Member 1)

Anomaly detector for a centrifugal pump: **normal (0) vs anomalous (1)** per 1-second window, from vibration (5 accelerometers)
and motor current/voltage (3 phases each) sampled at 20 kHz. Random Forest (200 trees) on 21 features.

> **Data status – read this first.** Every number below comes from a **simulated** dataset (`simulate_dataset.py`,
> 22,500 windows, 183 conditions) that mirrors the fault list, severities and speeds of the NLN-EMP dataset
> (Bruinsma et al., Data in Brief 52, 2024; 4TU DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0). They are **not measurements of a real pump**
> and must not be presented as such. The real NLN-EMP files could not be downloaded, so `build_dataset.py` (written for their documented
> layout) is **untested on them**: check the printed condition → label table when you first run it.

## Folder
| File | Purpose |
|---|---|
| `features.py` | the 21 features (15 base + 6 extra), windowing |
| `simulate_dataset.py` | generates `simulated_features.csv` (deterministic, ~90 s); `--hum-prob` sets neighbouring-machine noise |
| `build_dataset.py` | same features from real NLN-EMP-style raw CSVs |
| `train.py` | trains, evaluates, writes `model.joblib`, `report*.json`, `sample_predictions.csv` |
| `predict.py` | standalone prediction (CLI + importable `predict_window`) |
| `evaluate_robustness.py` | stress test on fresh windows under hum → `report_robustness.json` |
| `make_demo_raw_files.py` | tiny raw-layout demo tree, only used by the tests |
| `bench_features.py` | times feature extraction / prediction per window |
| `test_pipeline.py` | self-check (~1 min) |
| `simulated_features.csv` | the dataset: 21 feature columns + `label`, `fault`, `severity`, `speed`, `condition`, `hum` (1 = neighbouring-machine hum present) |
| `model.joblib` | final model (fitted on all windows; scikit-learn 1.8.0) |
| `report.json`, `report_base15.json`, `report_no_hum_aug.json`, `report_robustness.json` | evaluations: 21 features / 15 features / 21 features trained without hum / stress test |
| `sample_predictions.csv` | 40 random windows from held-out conditions (predicted by the evaluation model, not `model.joblib`) |

## Run
```
pip install -r requirements.txt
python3 test_pipeline.py
python3 simulate_dataset.py --out simulated_features.csv
python3 train.py --csv simulated_features.csv                 # 21 features -> model.joblib, report.json
python3 train.py --csv simulated_features.csv --features base --model model_base15.joblib --report report_base15.json --samples sample_base15.csv
python3 evaluate_robustness.py                                # -> report_robustness.json
python3 predict.py VIB.csv ELEC.csv                           # one JSON line per 1 s window
python3 bench_features.py
# real data, once downloaded:
python3 build_dataset.py <extracted NLN-EMP root> --out features.csv && python3 train.py --csv features.csv
```

## Interface for the other members
`predict.py VIB.csv ELEC.csv [--model PATH]` – `VIB.csv`: time + 5 accelerometer columns (g); `ELEC.csv`: time + 3 current (A) + 3 voltage (V) columns; 20 kHz. Output, one line per full 1 s window:
```
{"window": 0, "anomaly": true, "confidence": 0.965, "latency_ms": 33.6}
```
`confidence` is the Random Forest's P(anomaly) (**not calibrated**, not a severity); `anomaly` is `confidence >= 0.5`. `latency_ms` is feature extraction + prediction for that window.
Python: `from predict import load_model, predict_window` → `predict_window(model, vib(5,n), cur(3,n), volt(3,n)) -> (is_anomaly, p_anomaly)` for one window (n = 20000).
Inputs with the wrong number of channels or less than 1 s of data are rejected with an error message.

## Features (`features.py`)
Base 15: **vibration (10, averaged over the 5 sensors)** RMS, peak, crest factor, kurtosis, skewness, low-band (1–500 Hz) and high-band (0.5–5 kHz) energy share, spectral centroid, spectral entropy, dominant frequency; **current (4)** RMS, phase unbalance, harmonic distortion, sideband ratio; **voltage (1)** phase unbalance.
Extra 6: `shaft_hz` (estimated from the current fundamental, assuming a 2-pole motor with ~2 % slip), `vib_h1/h2/h3` (energy at 1×/2×/3× shaft speed), `vib_bpfo_band`, `vib_bpfi_band` (energy near textbook bearing-fault frequencies, ratios 3.05 and 4.95 – real bearings differ).

## Results (simulated data)
Pooled over all out-of-fold windows; each cell is precision / recall / F1 for the anomalous class. About 6,696 of 22,500 windows carry random neighbouring-machine hum.

| Evaluation | 15 base features | 21 features |
|---|---|---|
| Unseen conditions (GroupKFold by fault|severity|speed) – headline | 0.966 / 0.967 / 0.967 | 0.970 / 0.975 / 0.972 |
| Seen conditions (stratified window CV) – optimistic reference | 0.989 / 0.983 / 0.986 | 0.991 / 0.987 / 0.989 |

Held-out test (30 % of conditions, 21 features): precision 0.938, recall 0.995, F1 0.966; TN 1009, FP 341, FN 28, TP 5172.
The 6 extra features changed F1 by +0.006 (unseen conditions): not a meaningful difference on simulated data, and it cannot be shown to help on real data yet.

## Neighbouring-machine noise (the "noisy environment" problem)
**Problem found:** in the first version, the NLN-EMP-style "noisy environment" healthy variant (neighbouring machines running) was the only data containing hum, and only as healthy data. When its conditions were held out, the model flagged **99.9%** of those healthy windows as faults.
**Fix:** hum is a nuisance that exists regardless of pump state, so `simulate_dataset.py` now adds random mains-harmonic hum (50/100/150/200 Hz, amplitude 0.01–0.05 g, plus 0–0.03 g broadband) to 25 % of windows of every class (`--hum-prob 0` reproduces the old data exactly; checked).

| Healthy variant (unseen-conditions CV) | before: no hum augmentation | after: with hum augmentation | after, seen-conditions CV |
|---|---|---|---|
| Healthy 1 | 2.7% | 6.7% | 3.8% |
| Healthy 2 | 0.9% | 3.9% | 3.0% |
| Healthy 3 | 1.8% | 6.1% | 3.0% |
| Healthy noisy environment | 99.9% | 21.1% | 1.7% |

Cost of the fix – fault recall on unseen conditions (hum makes mild faults harder to see):

| Fault | before | after |
|---|---|---|
| Bearing grease contamination | 53.3% | 16.0% |
| Broken rotor bar | 100.0% | 75.7% |

(Before/after datasets differ, so overall F1 is not directly comparable; compare the rows above.) Other families stay at or above 96.4% recall ("Bearing pump BPFI+BPFO" is the lowest).

**Stress test on fresh windows** (`evaluate_robustness.py`: new seeds never used in training, random speeds/faults/severities, 21-feature `model.joblib`):

| Scenario | Windows | Result |
|---|---|---|
| Healthy, no hum | 600 | false-alarm rate 1.0% |
| Healthy + hum (in training range) | 600 | false-alarm rate 7.7% |
| Healthy 'noisy environment' variant | 600 | false-alarm rate 0.7% |
| Faults, no hum | 600 | recall 98.0% |
| Faults + hum (in training range) | 600 | recall 94.8% |
| Healthy + hum OUTSIDE training range (60/120/180 Hz, 2x louder) | 600 | false-alarm rate 86.2% |
| Faults + hum OUTSIDE training range | 600 | recall 99.8% |

The last two scenarios are deliberately outside the training distribution. They are a **documented limit, not a pass**: hum at frequencies/levels the model never saw is mistaken for a fault.

## Known weaknesses (reported as measured)
1. Healthy windows with in-range hum are still flagged more often than clean healthy ones (7.7% vs 1.0% on fresh windows); hum outside the training range is flagged 86.2% of the time.
2. Mild faults, especially "Bearing grease contamination" (16.0% recall on unseen conditions) and "Broken rotor bar" (75.7%), are often missed.
3. Probabilities are uncalibrated, and the simulator was written by the same person who designed the features (shared assumptions such as the bearing-frequency ratios), so scores are optimistic compared with real pumps.

## Speed (author's laptop, not an edge device)
Feature extraction ≈ 6.7 ms (15 features) vs ≈ 7.4 ms (21 features); full `predict_window` ≈ 21 ms (varies ±10 % between runs); model alone ≈ 14.0 ms of that. Measure again inside the edge container (`bench_features.py`).

## Reproducibility
Seeds are fixed (`--seed`, `random_state=0`); `simulate_dataset.py` output is byte-identical between runs (checked in `test_pipeline.py`). `model.joblib` was saved with scikit-learn 1.8.0; with another version retrain via `train.py`.
