"""Train the Random Forest normal(0)/anomalous(1) classifier and write an evaluation report.

Evaluation (all metrics come from models that did NOT see the evaluated windows):
  unseen_conditions_cv  GroupKFold(5) by condition (fault|severity|speed): the strict, headline protocol
  seen_conditions_cv    StratifiedKFold(5) over windows: conditions repeat across train/test, optimistic reference
  heldout_test          one GroupShuffleSplit (30 % of conditions), also used for timing-independent sample predictions
The saved model.joblib is then fitted on ALL windows (so its own training data is not a valid test set).
"""
import argparse, json, platform, time, joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from features import BASE_FEATURES, FEATURE_NAMES

SEED = 0


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1)}


def cross_validate(X, y, splits):
    pred, fold_f1 = np.zeros(len(y), int), []
    for tr, te in splits:
        pred[te] = rf().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        fold_f1.append(scores(y[te], pred[te])["f1"])
    return pred, {"pooled": scores(y, pred), "fold_f1_mean": float(np.mean(fold_f1)), "fold_f1_std": float(np.std(fold_f1)),
                  "n_folds": len(fold_f1)}


def by_family(df, pred):
    """Share of windows flagged anomalous per fault family: = recall for faults, = false-alarm rate for healthy variants."""
    rate = pd.Series(pred, index=df.index).groupby(df.fault).mean()
    lab = df.groupby("fault").label.first()
    return ({k: round(float(v), 4) for k, v in rate.items() if lab[k] == 1},
            {k: round(float(v), 4) for k, v in rate.items() if lab[k] == 0})


def main(csv, model_out, report_out, fset, samples_out):
    df = pd.read_csv(csv)
    cols = BASE_FEATURES if fset == "base" else FEATURE_NAMES
    X, y, g = df[cols], df.label.to_numpy(), df.condition.to_numpy()
    n_g = len(set(g))

    pred_u, unseen = cross_validate(X, y, GroupKFold(n_splits=min(5, n_g)).split(X, y, g))
    unseen["recall_by_fault"], unseen["false_alarm_rate_by_healthy_variant"] = by_family(df, pred_u)
    if "hum" in df:
        unseen["by_hum"] = {}
        for h, tag in ((0, "clean"), (1, "hum")):
            m_h = (df.hum == h).to_numpy()
            mean = lambda sel: round(float(pred_u[sel].mean()), 4) if sel.any() else None  # None -> JSON null, never NaN
            unseen["by_hum"][tag] = {"healthy_false_alarm_rate": mean(m_h & (y == 0)), "fault_recall": mean(m_h & (y == 1)),
                                     "n_healthy": int((m_h & (y == 0)).sum()), "n_fault": int((m_h & (y == 1)).sum())}
    strat = df.fault.to_numpy()
    k = int(min(5, pd.Series(strat).value_counts().min()))
    pred_s, seen = cross_validate(X, y, StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, strat))
    seen["recall_by_fault"], seen["false_alarm_rate_by_healthy_variant"] = by_family(df, pred_s)

    tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=SEED).split(X, y, g))
    ev = rf().fit(X.iloc[tr], y[tr])
    p_te = ev.predict(X.iloc[te])
    tn, fp, fn, tp = confusion_matrix(y[te], p_te, labels=[0, 1]).ravel()
    test = {**scores(y[te], p_te), "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "n_windows": int(len(te)), "test_conditions": sorted(set(g[te]))}
    smp = df.iloc[te].sample(n=min(40, len(te)), random_state=SEED)[["condition", "label"]].copy()
    smp["pred"] = ev.predict(X.loc[smp.index]); smp["p_anomaly"] = ev.predict_proba(X.loc[smp.index])[:, 1].round(3)
    smp.to_csv(samples_out, index=False)

    final = rf().fit(X, y)
    row = X.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(200):
        final.predict_proba(row)
    inf_ms = (time.perf_counter() - t0) / 200 * 1000

    rep = {
        "feature_set": fset, "n_features": len(cols), "features": list(cols), "n_windows": int(len(df)), "n_conditions": n_g,
        "anomalous_share": round(float(y.mean()), 4), "seed": SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "unseen_conditions_cv": unseen, "seen_conditions_cv": seen, "heldout_test": test,
        "model_only_inference_ms_per_window": round(inf_ms, 2),
        "feature_importance": {c: round(float(v), 4) for c, v in sorted(zip(cols, final.feature_importances_), key=lambda kv: -kv[1])},
        "notes": "model.joblib is fitted on all windows; metrics come from models that excluded the evaluated windows.",
    }
    joblib.dump({"model": final, "features": list(cols)}, model_out)
    json.dump(rep, open(report_out, "w"), indent=2, allow_nan=False)
    print(json.dumps({"unseen_conditions_cv": unseen["pooled"], "seen_conditions_cv": seen["pooled"],
                      "heldout_test": {k: test[k] for k in ("precision", "recall", "f1", "confusion")},
                      "model_only_inference_ms": rep["model_only_inference_ms_per_window"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features.csv")
    ap.add_argument("--features", choices=["base", "extended"], default="extended", help="base = 15 features, extended = 21")
    ap.add_argument("--model", default="model.joblib"); ap.add_argument("--report", default="report.json")
    ap.add_argument("--samples", default="sample_predictions.csv")
    a = ap.parse_args(); main(a.csv, a.model, a.report, a.features, a.samples)
