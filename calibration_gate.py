"""Calibration gate for tabular classifiers (TabICL-first, model-agnostic).

A reusable QA harness that answers one question before you trust a model's
probabilities: *is its confidence calibrated?* A model can be accurate yet
pathologically over-confident — its predicted probabilities then mean nothing,
and any confidence threshold you set downstream is noise.

The gate computes out-of-fold predictions ONCE per model (stratified CV) and
derives every metric from that single pass:
  - accuracy, balanced accuracy, f1-macro, f1-weighted, log-loss, ECE
  - a reliability curve + saturation flag (--calibration, no extra compute)
  - a paired per-fold delta candidate-vs-baseline on the gated metric
    (both models are evaluated on identical folds, so the fold-wise difference
    is a paired sample — far less noisy than comparing two independent means)

Exit codes: 0 = PASS (or baseline-only, no verdict), 1 = gate FAIL, 2 = invalid input.

Works on any sklearn-bundled dataset or your own CSV. Runs fully offline.
TabICL (https://github.com/soda-inria/tabicl) is used as the candidate model if
installed; otherwise the gate runs baseline-only (still a valid calibration check).

Examples
--------
    python calibration_gate.py --dataset breast_cancer --calibration
    python calibration_gate.py --dataset wine --device mps --calibration
    python calibration_gate.py --csv data.csv --target label --n-pca 64
    python calibration_gate.py --csv data.csv --target y --gate-metric ece --max-ece 0.05
    tabgate --csv data.csv --target y --candidate sklearn.ensemble.RandomForestClassifier \
        --gate-metric ece --json report.json      # after `pip install .`
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

__version__ = "0.2.0"

METRICS = ["accuracy", "bal_acc", "f1_macro", "f1_weighted", "log_loss", "ece"]
LOWER_IS_BETTER = {"log_loss", "ece"}

SKLEARN_DATASETS = {
    "breast_cancer": "load_breast_cancer",
    "wine": "load_wine",
    "iris": "load_iris",
    "digits": "load_digits",
}

BASELINE, CANDIDATE = "baseline_HGB", "TabICL"


def fail(msg: str) -> None:
    """Invalid input/setup: clear message on stderr, exit 2 (distinct from gate FAIL=1)."""
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(2)


def load_dataset(args) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a bundled sklearn dataset or a user CSV. No download."""
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.target not in df.columns:
            fail(f"--target '{args.target}' not found in CSV (columns: {list(df.columns)})")
        y = df[args.target]
        X = df.drop(columns=[args.target])
        return X, y
    if args.dataset not in SKLEARN_DATASETS:
        fail(f"--dataset unknown. Choose from: {', '.join(SKLEARN_DATASETS)} (or use --csv)")
    import sklearn.datasets as ds

    data = getattr(ds, SKLEARN_DATASETS[args.dataset])()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target")
    return X, y


def validate_inputs(X: pd.DataFrame, y: pd.Series, folds: int, pca_active: bool) -> None:
    """Fail loud (exit 2) on inputs the gate cannot handle, BEFORE any training."""
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        fail(f"non-numeric feature column(s): {non_numeric}. "
             "Encode them (e.g. OrdinalEncoder / one-hot) before running the gate.")
    if y.isna().any():
        fail(f"target contains {int(y.isna().sum())} missing value(s); drop or impute them first.")
    if y.nunique() < 2:
        fail(f"target has a single class ({y.unique().tolist()}); need >= 2 for classification.")
    counts = y.value_counts()
    rare = counts[counts < folds]
    if not rare.empty:
        fail(f"class(es) with fewer members than folds={folds}: {rare.to_dict()}. "
             "Stratified CV would produce folds with missing classes (silent NaN metrics). "
             "Merge/drop rare classes or lower --folds.")
    arr = X.to_numpy()
    if np.isinf(arr).any():
        fail(f"X contains {int(np.isinf(arr).sum())} infinite value(s); clean them first.")
    n_nan = int(pd.isna(arr).sum())
    if n_nan and pca_active:
        fail(f"X contains {n_nan} NaN(s) and PCA would be applied (features > n_pca): "
             "PCA cannot handle NaN. Impute first, or set --n-pca 0.")
    if n_nan:
        print(f"[warn] X contains {n_nan} NaN(s): HistGradientBoosting handles them natively, "
              "but some candidate models may not.\n")


def make_features(X: pd.DataFrame, n_pca: int) -> "ColumnTransformer | str":
    """Optional PCA when the feature count exceeds n_pca (strictly greater); else passthrough."""
    if n_pca and X.shape[1] > n_pca:
        return ColumnTransformer([("pca", PCA(n_components=n_pca, random_state=0), list(X.columns))])
    return "passthrough"


def load_candidate(spec: str, args_json: str | None, random_state: int):
    """Import and instantiate a plugin candidate ('module.path:ClassName' or dotted).

    The class must be sklearn-compatible and expose predict_proba. random_state is
    injected when the estimator accepts one and --candidate-args did not set it.
    """
    mod_name, _, cls_name = spec.replace(":", ".").rpartition(".")
    if not mod_name:
        fail(f"--candidate '{spec}' is not an import path (expected module.path:ClassName)")
    try:
        cls = getattr(importlib.import_module(mod_name), cls_name)
    except (ImportError, AttributeError) as e:
        fail(f"--candidate '{spec}' cannot be imported: {e}")
    kwargs = {}
    if args_json:
        try:
            kwargs = json.loads(args_json)
        except ValueError as e:
            fail(f"--candidate-args is not valid JSON: {e}")
        if not isinstance(kwargs, dict):
            fail(f"--candidate-args must be a JSON object, got {type(kwargs).__name__}")
    try:
        est = cls(**kwargs)
    except TypeError as e:
        fail(f"--candidate {cls_name}(**{kwargs}) failed to instantiate: {e}")
    if not hasattr(est, "predict_proba"):
        fail(f"--candidate {cls_name} has no predict_proba; the gate needs probabilities.")
    if "random_state" not in kwargs and hasattr(est, "get_params") \
            and "random_state" in est.get_params():
        est.set_params(random_state=random_state)
    return cls_name, est


def make_models(args) -> dict:
    models = {BASELINE: HistGradientBoostingClassifier(random_state=args.random_state)}
    if args.candidate:
        name, est = load_candidate(args.candidate, args.candidate_args, args.random_state)
        models[name] = est
        return models
    try:
        from tabicl import TabICLClassifier

        models[CANDIDATE] = TabICLClassifier(
            n_estimators=args.n_estimators, device=args.device, random_state=args.random_state
        )
    except ImportError:
        print("[info] tabicl not installed -> baseline-only (the calibration gate is still valid)\n")
    return models


def ece_top_label(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Top-label ECE with equal-width bins.

    ECE = mean over bins of |accuracy - confidence|, weighted by bin population.
    Returns (ece, rows) where rows = [(lo, hi, n, mean_conf, mean_acc), ...].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        acc_b, conf_b, n_b = correct[m].mean(), conf[m].mean(), int(m.sum())
        ece += (n_b / len(conf)) * abs(acc_b - conf_b)
        rows.append((edges[b], edges[b + 1], n_b, conf_b, acc_b))
    return ece, rows


def evaluate_oof(X, y, models, n_pca, folds) -> dict:
    """One out-of-fold predict_proba pass per model; every metric derives from it.

    Returns {name: {"per_fold": DataFrame(folds x METRICS), "conf": ..., "correct": ...,
    "wall_s": float}}. All models share the exact same folds (paired comparison).
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, np.asarray(y))
    fold_test = [test for _, test in cv.split(X, y)]
    labels = np.arange(len(classes))  # proba columns follow sorted class order
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("feat", make_features(X, n_pca)), ("clf", model)])
        t0 = time.perf_counter()
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)
        wall = time.perf_counter() - t0
        conf, pred = proba.max(axis=1), proba.argmax(axis=1)
        rows = []
        for test in fold_test:
            yt, pt = y_idx[test], pred[test]
            rows.append({
                "accuracy": accuracy_score(yt, pt),
                "bal_acc": balanced_accuracy_score(yt, pt),
                "f1_macro": f1_score(yt, pt, average="macro", zero_division=0),
                "f1_weighted": f1_score(yt, pt, average="weighted", zero_division=0),
                "log_loss": log_loss(yt, proba[test], labels=labels),
                "ece": ece_top_label(conf[test], pt == yt)[0],
            })
        results[name] = {"per_fold": pd.DataFrame(rows), "conf": conf,
                         "correct": pred == y_idx, "wall_s": wall}
    return results


def summary_table(results: dict) -> pd.DataFrame:
    rows = [{"model": name, **r["per_fold"].mean().to_dict(), "wall_s": r["wall_s"]}
            for name, r in results.items()]
    return pd.DataFrame(rows).set_index("model")[METRICS + ["wall_s"]]


def print_calibration(name: str, conf: np.ndarray, correct: np.ndarray) -> None:
    """Reliability curve on the pooled OOF predictions (the table's ece column is
    the per-fold mean; the two differ slightly by construction)."""
    ece, rows = ece_top_label(conf, correct)
    sat = float((conf > 0.99).mean())
    # Saturation is only a problem when it comes WITH miscalibration (high ECE).
    # On easy datasets, confidently-correct predictions saturate legitimately.
    warn = "   <-- SATURATED + miscalibrated (confidence unreliable)" if (sat > 0.5 and ece > 0.05) else ""
    print(f"\n=== calibration: {name} ===")
    print(f"mean confidence={conf.mean():.3f} | accuracy={correct.mean():.3f} | ECE={ece:.4f} (pooled)")
    print(f"share conf>0.99 = {sat:.1%}{warn}")
    print("reliability curve (confidence vs accuracy per bin):")
    for lo, hi, n, c, a in rows:
        print(f"  [{lo:.1f}-{hi:.1f}) n={n:5d}  conf={c:.3f}  acc={a:.3f}  gap={a - c:+.3f}")


def compute_gate(results: dict, metric: str, epsilon: float, max_ece: float | None,
                 candidate: str = CANDIDATE) -> dict:
    """Verdict on the paired per-fold delta, plus an optional absolute ECE ceiling.

    Pure computation (all values JSON-serializable); printing lives in print_gate.
    """
    cand = results[candidate]["per_fold"][metric].to_numpy()
    base = results[BASELINE]["per_fold"][metric].to_numpy()
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    deltas = sign * (cand - base)  # '+' = candidate better, whatever the metric direction
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    delta_ok = mean_d >= -epsilon
    pooled = float(ece_top_label(results[candidate]["conf"], results[candidate]["correct"])[0])
    ece_ok = None if max_ece is None else bool(pooled <= max_ece)
    return {
        "metric": metric, "epsilon": epsilon, "candidate": candidate, "baseline": BASELINE,
        "candidate_mean": float(cand.mean()), "baseline_mean": float(base.mean()),
        "delta_mean": mean_d, "delta_std": std_d, "delta_min": float(deltas.min()),
        "n_folds": int(len(deltas)), "delta_pass": bool(delta_ok),
        "pooled_ece": pooled, "max_ece": max_ece, "ece_pass": ece_ok,
        "pass": bool(delta_ok and ece_ok is not False),
    }


def print_gate(g: dict) -> None:
    print(f"\n[gate] {g['metric']}: {g['candidate']}={g['candidate_mean']:.4f} "
          f"vs {g['baseline']}={g['baseline_mean']:.4f} (epsilon={g['epsilon']})")
    print(f"[gate] paired delta per fold ('+' = candidate better): "
          f"{g['delta_mean']:+.4f} ± {g['delta_std']:.4f} "
          f"(min {g['delta_min']:+.4f}, {g['n_folds']} folds) "
          f"-> {'PASS' if g['delta_pass'] else 'FAIL'}")
    if g["max_ece"] is not None:
        print(f"[gate] ECE ceiling: candidate pooled ECE={g['pooled_ece']:.4f} "
              f"{'<=' if g['ece_pass'] else '>'} {g['max_ece']} "
              f"-> {'PASS' if g['ece_pass'] else 'FAIL'}")
        print(f"[gate] overall -> {'PASS' if g['pass'] else 'FAIL'}")


def run_gate(results: dict, metric: str, epsilon: float, max_ece: float | None,
             candidate: str = CANDIDATE) -> int:
    g = compute_gate(results, metric, epsilon, max_ece, candidate)
    print_gate(g)
    return 0 if g["pass"] else 1


def write_json(path: str, args, X, y, results: dict, gate: dict | None, code: int,
               pca_active: bool) -> None:
    """Machine-readable report (written for PASS and FAIL alike — a CI artifact)."""
    doc = {
        "schema_version": 1,
        "source": args.csv or args.dataset,
        "dataset": {"n": int(len(y)), "n_features": int(X.shape[1]),
                    "n_classes": int(y.nunique()), "folds": args.folds,
                    "pca": args.n_pca if pca_active else None},
        "models": {
            name: {
                "metrics": {k: float(v) for k, v in r["per_fold"].mean().items()},
                "per_fold": {k: [float(x) for x in v]
                             for k, v in r["per_fold"].to_dict("list").items()},
                "pooled_ece": float(ece_top_label(r["conf"], r["correct"])[0]),
                "wall_s": float(r["wall_s"]),
            } for name, r in results.items()},
        "gate": gate,
        "exit_code": code,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\n[json] report written to {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source")
    src.add_argument("--dataset", default="breast_cancer", help=f"sklearn bundled: {', '.join(SKLEARN_DATASETS)}")
    src.add_argument("--csv", default=None, help="path to a CSV (use with --target)")
    src.add_argument("--target", default="target", help="target column name when using --csv")
    p.add_argument("--n-pca", type=int, default=64, help="PCA components if n_features exceeds it (0=off)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--candidate", default=None, metavar="MODULE:CLASS",
                   help="plugin candidate: import path of any sklearn-compatible classifier "
                        "with predict_proba (e.g. sklearn.ensemble.RandomForestClassifier); "
                        "replaces TabICL")
    p.add_argument("--candidate-args", default=None, metavar="JSON",
                   help="constructor kwargs for --candidate as a JSON object, "
                        "e.g. '{\"n_estimators\": 200}'")
    p.add_argument("--json", default=None, metavar="PATH", dest="json_path",
                   help="write a machine-readable JSON report (written for PASS and FAIL alike)")
    p.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size")
    p.add_argument("--device", default=None, help="TabICL device: 'mps', 'cpu', None=auto")
    p.add_argument("--gate-metric", default="f1_macro", choices=METRICS,
                   help="metric the PASS/FAIL is decided on (log_loss and ece gate lower-is-better)")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="tolerance on the paired mean delta: candidate may be worse by up to epsilon")
    p.add_argument("--max-ece", type=float, default=None,
                   help="absolute ceiling on the candidate's pooled ECE (second gate condition)")
    p.add_argument("--require-candidate", action="store_true",
                   help="exit 2 when tabicl is not installed (instead of green baseline-only)")
    p.add_argument("--calibration", action="store_true",
                   help="print reliability curves (ECE itself is always computed)")
    p.add_argument("--random-state", type=int, default=42,
                   help="seed for the models (CV splits stay fixed so both models share folds)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args()
    if args.candidate_args and not args.candidate:
        fail("--candidate-args requires --candidate")

    X, y = load_dataset(args)
    pca_active = bool(args.n_pca and X.shape[1] > args.n_pca)
    validate_inputs(X, y, args.folds, pca_active)
    print(f"dataset: n={len(y)}, features={X.shape[1]}, classes={y.nunique()}"
          f"{f' | PCA-{args.n_pca}' if pca_active else ''} | folds={args.folds}\n")

    models = make_models(args)
    candidate_name = next((k for k in models if k != BASELINE), None)
    if candidate_name is None and args.require_candidate:
        print("[error] --require-candidate set but tabicl is not installed", file=sys.stderr)
        return 2

    results = evaluate_oof(X, y, models, args.n_pca, args.folds)
    report = summary_table(results)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(report.to_string())
    if report[METRICS].isna().any().any():
        print("[error] non-finite metric(s) in the report — check warnings above", file=sys.stderr)
        return 2

    if args.calibration:
        for name in models:
            print_calibration(name, results[name]["conf"], results[name]["correct"])

    gate, code = None, 0
    if candidate_name is None:
        print(f"\n[gate] {CANDIDATE} not installed -> baseline-only, no verdict. exit 0")
    else:
        gate = compute_gate(results, args.gate_metric, args.epsilon, args.max_ece, candidate_name)
        print_gate(gate)
        code = 0 if gate["pass"] else 1
    if args.json_path:
        write_json(args.json_path, args, X, y, results, gate, code, pca_active)
    return code


if __name__ == "__main__":
    sys.exit(main())
