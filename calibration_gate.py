"""Calibration gate for tabular classifiers (TabICL-first, model-agnostic).

A reusable QA harness that answers one question before you trust a model's
probabilities: *is its confidence calibrated?* A model can be accurate yet
pathologically over-confident — its predicted probabilities then mean nothing,
and any confidence threshold you set downstream is noise.

The gate computes out-of-fold predictions ONCE per model (stratified CV) and
derives every metric from that single pass:
  - accuracy, balanced accuracy, f1-macro, f1-weighted, log-loss, Brier, ECE
  - a reliability curve + saturation flag + classwise-ECE (--calibration,
    no extra compute); equal-width or equal-mass bins (--ece-binning)
  - a paired per-fold delta candidate-vs-baseline on the gated metric
    (both models are evaluated on identical folds, so the fold-wise difference
    is a paired sample — far less noisy than comparing two independent means)
  - optional nested temperature-scaling diagnostic (--temperature): is the
    miscalibration fixable by simple post-hoc scaling? Never affects the gate.

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
    tabgate --csv data.csv --target y --max-classwise-ece 0.08 --ece-binning mass
    tabgate --dataset breast_cancer --temperature --plot reliability.png
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

__version__ = "0.3.0"

METRICS = ["accuracy", "bal_acc", "f1_macro", "f1_weighted", "log_loss", "brier", "ece"]
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}

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


def bin_edges(conf: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin edges over [0, 1]: 'width' = equal-width, 'mass' = equal-mass (quantiles).

    Equal-mass bins keep per-bin population balanced when confidences cluster
    (e.g. most predictions above 0.99): equal-width bins then leave most bins
    empty and one bin dominant, so the ECE rests on a single noisy cell.
    Duplicate quantiles (heavy ties) are collapsed — fewer effective bins.
    """
    if strategy == "mass":
        qs = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        return np.unique(np.concatenate(([0.0], qs[1:-1], [1.0])))
    return np.linspace(0.0, 1.0, n_bins + 1)


def ece_top_label(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10,
                  strategy: str = "width"):
    """Binned calibration error of a score against a binary outcome.

    Used for top-label ECE (conf = max proba, outcome = prediction correct) and,
    per class, for classwise-ECE (conf = p_class, outcome = sample is that class).
    ECE = mean over bins of |outcome rate - confidence|, weighted by bin population.
    Returns (ece, rows) where rows = [(lo, hi, n, mean_conf, mean_acc), ...].
    """
    edges = bin_edges(conf, n_bins, strategy)
    nb = len(edges) - 1
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, nb - 1)
    ece, rows = 0.0, []
    for b in range(nb):
        m = bins == b
        if not m.any():
            continue
        acc_b, conf_b, n_b = correct[m].mean(), conf[m].mean(), int(m.sum())
        ece += (n_b / len(conf)) * abs(acc_b - conf_b)
        rows.append((edges[b], edges[b + 1], n_b, conf_b, acc_b))
    return ece, rows


def brier_multiclass(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error of the FULL probability vector
    against the one-hot truth, range [0, 2]. Strictly proper — unlike accuracy,
    it punishes confident mistakes more than hesitant ones.

    Binary note: this is 2x sklearn's brier_score_loss (which scores the
    positive-class probability only).
    """
    onehot = np.eye(proba.shape[1])[y_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def classwise_ece(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10,
                  strategy: str = "width"):
    """Classwise-ECE: for each class c, calibration of p_c against the empirical
    frequency of c, over ALL samples (Kull et al. 2019) — not just those predicted c.

    Top-label ECE can look fine while one class is badly miscalibrated. Any
    PER-CLASS confidence threshold downstream (auto-label class c above 0.6...)
    depends on this, not on the top-label curve.
    Returns (mean_over_classes, {class_index: ece}).
    """
    per_class = {c: ece_top_label(proba[:, c], y_idx == c, n_bins, strategy)[0]
                 for c in range(proba.shape[1])}
    return float(np.mean(list(per_class.values()))), per_class


def evaluate_oof(X, y, models, n_pca, folds, ece_bins: int = 10,
                 ece_binning: str = "width") -> dict:
    """One out-of-fold predict_proba pass per model; every metric derives from it.

    Returns {name: {"per_fold": DataFrame(folds x METRICS), "proba": ..., "conf": ...,
    "correct": ..., "y_idx": ..., "classes": ..., "wall_s": float}}.
    All models share the exact same folds (paired comparison).
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
                "brier": brier_multiclass(proba[test], yt),
                "ece": ece_top_label(conf[test], pt == yt, ece_bins, ece_binning)[0],
            })
        results[name] = {"per_fold": pd.DataFrame(rows), "proba": proba, "conf": conf,
                         "correct": pred == y_idx, "y_idx": y_idx, "classes": classes,
                         "wall_s": wall}
    return results


def summary_table(results: dict) -> pd.DataFrame:
    rows = [{"model": name, **r["per_fold"].mean().to_dict(), "wall_s": r["wall_s"]}
            for name, r in results.items()]
    return pd.DataFrame(rows).set_index("model")[METRICS + ["wall_s"]]


def print_calibration(name: str, r: dict, n_bins: int = 10, strategy: str = "width") -> None:
    """Reliability curve + classwise-ECE on the pooled OOF predictions (the table's
    ece column is the per-fold mean; the two differ slightly by construction)."""
    conf, correct = r["conf"], r["correct"]
    ece, rows = ece_top_label(conf, correct, n_bins, strategy)
    sat = float((conf > 0.99).mean())
    # Saturation is only a problem when it comes WITH miscalibration (high ECE).
    # On easy datasets, confidently-correct predictions saturate legitimately.
    warn = "   <-- SATURATED + miscalibrated (confidence unreliable)" if (sat > 0.5 and ece > 0.05) else ""
    print(f"\n=== calibration: {name} ===")
    print(f"mean confidence={conf.mean():.3f} | accuracy={correct.mean():.3f} | ECE={ece:.4f} (pooled)")
    print(f"share conf>0.99 = {sat:.1%}{warn}")
    cw_mean, per_class = classwise_ece(r["proba"], r["y_idx"], n_bins, strategy)
    worst = sorted(per_class.items(), key=lambda kv: -kv[1])[:3]
    print(f"classwise-ECE mean={cw_mean:.4f} | worst: "
          + ", ".join(f"{r['classes'][c]}={e:.4f}" for c, e in worst))
    print(f"reliability curve (confidence vs accuracy per bin, {strategy} bins):")
    for lo, hi, n, c, a in rows:
        print(f"  [{lo:.2f}-{hi:.2f}) n={n:5d}  conf={c:.3f}  acc={a:.3f}  gap={a - c:+.3f}")


def compute_gate(results: dict, metric: str, epsilon: float, max_ece: float | None,
                 candidate: str = CANDIDATE, max_classwise_ece: float | None = None,
                 ece_bins: int = 10, ece_binning: str = "width") -> dict:
    """Verdict on the paired per-fold delta, plus optional absolute ceilings on the
    candidate's pooled ECE and on its WORST per-class ECE.

    Pure computation (all values JSON-serializable); printing lives in print_gate.
    """
    cand = results[candidate]["per_fold"][metric].to_numpy()
    base = results[BASELINE]["per_fold"][metric].to_numpy()
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    deltas = sign * (cand - base)  # '+' = candidate better, whatever the metric direction
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    delta_ok = mean_d >= -epsilon
    pooled = float(ece_top_label(results[candidate]["conf"], results[candidate]["correct"],
                                 ece_bins, ece_binning)[0])
    ece_ok = None if max_ece is None else bool(pooled <= max_ece)
    cw_worst, cw_ok = None, None
    if max_classwise_ece is not None:
        _, per_class = classwise_ece(results[candidate]["proba"], results[candidate]["y_idx"],
                                     ece_bins, ece_binning)
        wc = max(per_class, key=per_class.get)
        cw_worst = {"class": str(results[candidate]["classes"][wc]),
                    "ece": float(per_class[wc])}
        cw_ok = bool(per_class[wc] <= max_classwise_ece)
    return {
        "metric": metric, "epsilon": epsilon, "candidate": candidate, "baseline": BASELINE,
        "candidate_mean": float(cand.mean()), "baseline_mean": float(base.mean()),
        "delta_mean": mean_d, "delta_std": std_d, "delta_min": float(deltas.min()),
        "n_folds": int(len(deltas)), "delta_pass": bool(delta_ok),
        "pooled_ece": pooled, "max_ece": max_ece, "ece_pass": ece_ok,
        "classwise_worst": cw_worst, "max_classwise_ece": max_classwise_ece,
        "classwise_pass": cw_ok,
        "pass": bool(delta_ok and ece_ok is not False and cw_ok is not False),
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
    if g["max_classwise_ece"] is not None:
        print(f"[gate] classwise-ECE ceiling: worst class '{g['classwise_worst']['class']}' "
              f"ECE={g['classwise_worst']['ece']:.4f} "
              f"{'<=' if g['classwise_pass'] else '>'} {g['max_classwise_ece']} "
              f"-> {'PASS' if g['classwise_pass'] else 'FAIL'}")
    if g["max_ece"] is not None or g["max_classwise_ece"] is not None:
        print(f"[gate] overall -> {'PASS' if g['pass'] else 'FAIL'}")


def run_gate(results: dict, metric: str, epsilon: float, max_ece: float | None,
             candidate: str = CANDIDATE) -> int:
    g = compute_gate(results, metric, epsilon, max_ece, candidate)
    print_gate(g)
    return 0 if g["pass"] else 1


def fit_temperature(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Scalar temperature minimizing the NLL of softmax(log(p)/T), using log-probs
    as pseudo-logits (standard when true logits are unavailable; the rescaling is
    monotone, so predictions never change — only confidence does).

    T > 1 softens over-confident probabilities, T < 1 sharpens under-confident
    ones. Scalar + smooth in log T -> a log-spaced grid beats an optimizer dep.
    """
    logp = np.log(np.clip(proba, 1e-12, 1.0))
    idx = np.arange(len(y_idx))
    best_t, best_nll = 1.0, np.inf
    for t in np.logspace(np.log10(0.05), np.log10(20.0), 161):
        z = logp / t
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        nll = float(-np.mean(np.log(np.clip(p[idx, y_idx], 1e-12, None))))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def temperature_pass(X, y, models, results, n_pca, folds, ece_bins: int = 10,
                     ece_binning: str = "width", inner_folds: int = 3) -> dict:
    """Nested temperature-scaling diagnostic: is the miscalibration fixable by
    simple post-hoc scaling?

    For each outer fold, T is fitted on inner-CV out-of-fold probabilities of the
    TRAINING part only (cloned pipeline — the test fold never touches the T fit),
    then applied to the outer test probabilities of the main pass. Leak-free, but
    costs ~(folds-1)x extra predictions per model. Diagnostic only: the gate
    never uses tempered values.
    """
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    y_arr = np.asarray(y)
    out = {}
    for name, model in models.items():
        r = results[name]
        tempered = np.empty_like(r["proba"])
        ts = []
        for train, test in cv.split(X, y):
            min_count = int(np.bincount(r["y_idx"][train]).min())
            k = min(inner_folds, min_count)
            if k < 2:
                fail(f"--temperature needs >= 2 members per class inside each training "
                     f"split (smallest has {min_count}); lower --folds or drop rare classes.")
            pipe = Pipeline([("feat", make_features(X, n_pca)), ("clf", clone(model))])
            inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
            p_inner = cross_val_predict(pipe, X.iloc[train], y_arr[train], cv=inner_cv,
                                        method="predict_proba", n_jobs=1)
            t = fit_temperature(p_inner, r["y_idx"][train])
            ts.append(t)
            tempered[test] = apply_temperature(r["proba"][test], t)
        # argmax is unchanged by monotone scaling -> same predictions, same "correct"
        ece_after = ece_top_label(tempered.max(axis=1), r["correct"], ece_bins, ece_binning)[0]
        ece_before = ece_top_label(r["conf"], r["correct"], ece_bins, ece_binning)[0]
        out[name] = {"t_per_fold": [float(t) for t in ts], "t_mean": float(np.mean(ts)),
                     "ece_before": float(ece_before), "ece_after": float(ece_after)}
    return out


def print_temperature(temp: dict) -> None:
    print("\n=== temperature scaling (nested, diagnostic — never gates) ===")
    for name, t in temp.items():
        note = ""
        if t["ece_before"] > 0.02 and t["ece_after"] < 0.5 * t["ece_before"]:
            note = "   <-- post-hoc scaling would fix most of the miscalibration"
        elif abs(t["t_mean"] - 1.0) < 0.1:
            note = "   (T ~= 1: probabilities already well scaled)"
        print(f"{name}: T={t['t_mean']:.3f} "
              f"(per fold: {', '.join(f'{x:.2f}' for x in t['t_per_fold'])}) | "
              f"pooled ECE {t['ece_before']:.4f} -> {t['ece_after']:.4f}{note}")


def write_plot(path: str, results: dict, ece_bins: int = 10, ece_binning: str = "width") -> None:
    """Reliability-diagram PNG on the pooled OOF predictions (all models)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe, no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        fail("--plot requires matplotlib: pip install matplotlib (or: pip install '.[plot]')")
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, r in results.items():
        ece, rows = ece_top_label(r["conf"], r["correct"], ece_bins, ece_binning)
        xs = [row[3] for row in rows]
        ys = [row[4] for row in rows]
        sizes = [20 + 180 * row[2] / len(r["conf"]) for row in rows]  # area ~ bin population
        line, = ax.plot(xs, ys, "-", alpha=0.8, label=f"{name} (ECE={ece:.4f})")
        ax.scatter(xs, ys, s=sizes, color=line.get_color(), alpha=0.7, zorder=3)
    ax.set_xlabel("mean predicted confidence (per bin)")
    ax.set_ylabel("empirical accuracy (per bin)")
    ax.set_title(f"Reliability diagram — pooled OOF, {ece_binning} bins\n"
                 f"(marker area ~ bin population)", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] reliability diagram written to {path}")


def write_json(path: str, args, X, y, results: dict, gate: dict | None, code: int,
               pca_active: bool, temperature: dict | None = None) -> None:
    """Machine-readable report (written for PASS and FAIL alike — a CI artifact).

    schema_version 2: adds brier to metrics, classwise_ece per model, the binning
    params, and the optional temperature block (all additive vs v1).
    """
    def model_block(r: dict) -> dict:
        cw_mean, per_class = classwise_ece(r["proba"], r["y_idx"],
                                           args.ece_bins, args.ece_binning)
        return {
            "metrics": {k: float(v) for k, v in r["per_fold"].mean().items()},
            "per_fold": {k: [float(x) for x in v]
                         for k, v in r["per_fold"].to_dict("list").items()},
            "pooled_ece": float(ece_top_label(r["conf"], r["correct"],
                                              args.ece_bins, args.ece_binning)[0]),
            "classwise_ece": {"mean": cw_mean,
                              "per_class": {str(r["classes"][c]): float(e)
                                            for c, e in per_class.items()}},
            "wall_s": float(r["wall_s"]),
        }

    doc = {
        "schema_version": 2,
        "source": args.csv or args.dataset,
        "dataset": {"n": int(len(y)), "n_features": int(X.shape[1]),
                    "n_classes": int(y.nunique()), "folds": args.folds,
                    "pca": args.n_pca if pca_active else None},
        "params": {"ece_bins": args.ece_bins, "ece_binning": args.ece_binning},
        "models": {name: model_block(r) for name, r in results.items()},
        "gate": gate,
        "temperature": temperature,
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
                   help="metric the PASS/FAIL is decided on "
                        "(log_loss, brier and ece gate lower-is-better)")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="tolerance on the paired mean delta: candidate may be worse by up to epsilon")
    p.add_argument("--max-ece", type=float, default=None,
                   help="absolute ceiling on the candidate's pooled ECE (second gate condition)")
    p.add_argument("--max-classwise-ece", type=float, default=None,
                   help="ceiling on the candidate's WORST per-class ECE — the guard to set "
                        "when downstream uses per-class confidence thresholds")
    p.add_argument("--ece-bins", type=int, default=10, help="number of calibration bins")
    p.add_argument("--ece-binning", choices=["width", "mass"], default="width",
                   help="bin strategy for every ECE (table, gate, curves): equal-width or "
                        "equal-mass (quantile) — mass stays informative when confidences "
                        "cluster near 1.0")
    p.add_argument("--temperature", action="store_true",
                   help="nested temperature-scaling diagnostic (T fitted on inner-CV train "
                        "probabilities, leak-free): shows whether post-hoc scaling would fix "
                        "the miscalibration; never affects the gate; ~(folds-1)x extra predictions")
    p.add_argument("--plot", default=None, metavar="PATH",
                   help="write a reliability-diagram PNG (requires matplotlib)")
    p.add_argument("--require-candidate", action="store_true",
                   help="exit 2 when tabicl is not installed (instead of green baseline-only)")
    p.add_argument("--calibration", action="store_true",
                   help="print reliability curves + classwise-ECE (ECE itself is always computed)")
    p.add_argument("--random-state", type=int, default=42,
                   help="seed for the models (CV splits stay fixed so both models share folds)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args()
    if args.candidate_args and not args.candidate:
        fail("--candidate-args requires --candidate")
    if args.ece_bins < 2:
        fail("--ece-bins must be >= 2")

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

    results = evaluate_oof(X, y, models, args.n_pca, args.folds,
                           args.ece_bins, args.ece_binning)
    report = summary_table(results)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(report.to_string())
    if report[METRICS].isna().any().any():
        print("[error] non-finite metric(s) in the report — check warnings above", file=sys.stderr)
        return 2

    if args.calibration:
        for name in models:
            print_calibration(name, results[name], args.ece_bins, args.ece_binning)

    temperature = None
    if args.temperature:
        temperature = temperature_pass(X, y, models, results, args.n_pca, args.folds,
                                       args.ece_bins, args.ece_binning)
        print_temperature(temperature)

    gate, code = None, 0
    if candidate_name is None:
        print(f"\n[gate] {CANDIDATE} not installed -> baseline-only, no verdict. exit 0")
    else:
        gate = compute_gate(results, args.gate_metric, args.epsilon, args.max_ece,
                            candidate_name, args.max_classwise_ece,
                            args.ece_bins, args.ece_binning)
        print_gate(gate)
        code = 0 if gate["pass"] else 1
    if args.plot:
        write_plot(args.plot, results, args.ece_bins, args.ece_binning)
    if args.json_path:
        write_json(args.json_path, args, X, y, results, gate, code, pca_active, temperature)
    return code


if __name__ == "__main__":
    sys.exit(main())
