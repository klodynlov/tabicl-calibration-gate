"""Calibration gate for tabular classifiers (TabICL-first, model-agnostic).

A reusable QA harness that answers one question before you trust a model's
probabilities: *is its confidence calibrated?* A model can be accurate yet
pathologically over-confident — its predicted probabilities then mean nothing,
and any confidence threshold you set downstream is noise.

This gate runs stratified cross-validation and reports, for each model:
  - accuracy, balanced accuracy, f1-macro, f1-weighted, log-loss
  - Expected Calibration Error (ECE) + a text reliability curve
  - a saturation flag (share of predictions above 0.99 confidence)
and gates a candidate model against a baseline on a chosen metric (exit code).

Works on any sklearn-bundled dataset or your own CSV. Runs fully offline.
TabICL (https://github.com/soda-inria/tabicl) is used as the candidate model if
installed; otherwise the gate runs baseline-only (still a valid calibration check).

Examples
--------
    python calibration_gate.py --dataset breast_cancer --calibration
    python calibration_gate.py --dataset wine --device mps --calibration
    python calibration_gate.py --csv data.csv --target label --n-pca 64
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "neg_log_loss": "neg_log_loss",
}

SKLEARN_DATASETS = {
    "breast_cancer": "load_breast_cancer",
    "wine": "load_wine",
    "iris": "load_iris",
    "digits": "load_digits",
}


def load_dataset(args) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a bundled sklearn dataset or a user CSV. No download."""
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.target not in df.columns:
            sys.exit(f"--target '{args.target}' absent du CSV (colonnes: {list(df.columns)})")
        y = df[args.target]
        X = df.drop(columns=[args.target])
        return X, y
    if args.dataset not in SKLEARN_DATASETS:
        sys.exit(f"--dataset inconnu. Choix: {', '.join(SKLEARN_DATASETS)} (ou utilise --csv)")
    import sklearn.datasets as ds

    data = getattr(ds, SKLEARN_DATASETS[args.dataset])()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target")
    return X, y


def make_features(X: pd.DataFrame, n_pca: int) -> Pipeline | str:
    """Optional PCA when the feature count is large (e.g. embeddings); else passthrough."""
    if n_pca and X.shape[1] > n_pca:
        return ColumnTransformer([("pca", PCA(n_components=n_pca, random_state=0), list(X.columns))])
    return "passthrough"


def make_models(device: str | None, n_estimators: int, random_state: int) -> dict:
    models = {"baseline_HGB": HistGradientBoostingClassifier(random_state=random_state)}
    try:
        from tabicl import TabICLClassifier

        models["TabICL"] = TabICLClassifier(
            n_estimators=n_estimators, device=device, random_state=random_state
        )
    except ImportError:
        print("[info] tabicl non installé -> baseline-seul (le gate de calibration reste valide)\n")
    return models


def evaluate(X, y, models, n_pca, folds) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("feat", make_features(X, n_pca)), ("clf", model)])
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        wall = time.perf_counter() - t0
        rows.append({
            "model": name,
            "accuracy": res["test_accuracy"].mean(),
            "bal_acc": res["test_balanced_accuracy"].mean(),
            "f1_macro": res["test_f1_macro"].mean(),
            "f1_weighted": res["test_f1_weighted"].mean(),
            "log_loss": -res["test_neg_log_loss"].mean(),
            "wall_s": wall,
        })
    return pd.DataFrame(rows).set_index("model")


def calibration_report(X, y, models, n_pca, folds) -> None:
    """Out-of-fold calibration: top-label confidence vs accuracy, ECE, reliability curve.

    Multiclass-safe: confidence = max predicted probability, correctness = argmax == y.
    ECE = mean over bins of |accuracy - confidence|, weighted by bin population.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, np.asarray(y))
    for name, model in models.items():
        pipe = Pipeline([("feat", make_features(X, n_pca)), ("clf", model)])
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)
        conf = proba.max(axis=1)
        correct = proba.argmax(axis=1) == y_idx

        edges = np.linspace(0.0, 1.0, 11)
        bins = np.clip(np.digitize(conf, edges[1:-1]), 0, 9)
        ece, lines = 0.0, []
        for b in range(10):
            m = bins == b
            if not m.any():
                continue
            acc_b, conf_b, n_b = correct[m].mean(), conf[m].mean(), int(m.sum())
            ece += (n_b / len(conf)) * abs(acc_b - conf_b)
            lines.append(f"  [{edges[b]:.1f}-{edges[b+1]:.1f}) n={n_b:5d}  conf={conf_b:.3f}  acc={acc_b:.3f}  gap={acc_b-conf_b:+.3f}")

        sat = float((conf > 0.99).mean())
        # Saturation is only a problem when it comes WITH miscalibration (high ECE).
        # On easy datasets, confidently-correct predictions saturate legitimately.
        warn = "   <-- SATURATED + miscalibrated (confidence unreliable)" if (sat > 0.5 and ece > 0.05) else ""
        print(f"\n=== calibration: {name} ===")
        print(f"mean confidence={conf.mean():.3f} | accuracy={correct.mean():.3f} | ECE={ece:.4f}")
        print(f"share conf>0.99 = {sat:.1%}{warn}")
        print("reliability curve (confidence vs accuracy per bin):")
        print("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source")
    src.add_argument("--dataset", default="breast_cancer", help=f"sklearn bundled: {', '.join(SKLEARN_DATASETS)}")
    src.add_argument("--csv", default=None, help="path to a CSV (use with --target)")
    src.add_argument("--target", default="target", help="target column name when using --csv")
    p.add_argument("--n-pca", type=int, default=64, help="PCA components if n_features exceeds it (0=off)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size")
    p.add_argument("--device", default=None, help="TabICL device: 'mps', 'cpu', None=auto")
    p.add_argument("--gate-metric", default="f1_macro",
                   choices=["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"])
    p.add_argument("--epsilon", type=float, default=0.0, help="tolerance: TabICL must be >= baseline - epsilon")
    p.add_argument("--calibration", action="store_true", help="add ECE + reliability diagnostics")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    X, y = load_dataset(args)
    print(f"dataset: n={len(y)}, features={X.shape[1]}, classes={y.nunique()}"
          f"{f' | PCA-{args.n_pca}' if args.n_pca and X.shape[1] > args.n_pca else ''} | folds={args.folds}\n")

    models = make_models(args.device, args.n_estimators, args.random_state)
    report = evaluate(X, y, models, args.n_pca, args.folds)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(report.to_string())

    if args.calibration:
        calibration_report(X, y, models, args.n_pca, args.folds)

    # --- gate ---
    if "TabICL" not in report.index:
        print("\n[gate] TabICL absent -> baseline-only, no verdict. exit 0")
        return 0
    gm = args.gate_metric
    tab, base = report.loc["TabICL", gm], report.loc["baseline_HGB", gm]
    ok = (tab - base) >= -args.epsilon
    print(f"\n[gate] {gm}: TabICL={tab:.4f} vs baseline={base:.4f} "
          f"(delta={tab-base:+.4f}, epsilon={args.epsilon}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
