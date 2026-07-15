"""Tests for calibration_gate. Run: python -m pytest test_calibration_gate.py -q

TabICL is NOT required: end-to-end tests force the baseline-only path (tabicl
import blocked in the subprocess), and the gate logic is tested directly with
synthetic per-fold results.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import calibration_gate as cg

SCRIPT = Path(__file__).with_name("calibration_gate.py")


def run_script(*argv: str, block: tuple = ("tabicl",)) -> subprocess.CompletedProcess:
    """Run the gate in a subprocess with the given imports blocked (tabicl by
    default -> baseline-only, deterministic even in an env where it is installed)."""
    blocks = "; ".join(f"sys.modules['{m}'] = None" for m in block)
    code = (f"import sys; {blocks}; "
            f"exec(compile(open(r'{SCRIPT}').read(), r'{SCRIPT}', 'exec'))")
    return subprocess.run([sys.executable, "-c", code, *argv],
                          capture_output=True, text=True, timeout=180)


def write_csv(tmp_path: Path, df: pd.DataFrame) -> str:
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


# --- ECE unit tests ---------------------------------------------------------

def test_ece_perfectly_calibrated():
    conf = np.full(400, 0.75)
    correct = np.zeros(400, dtype=bool)
    correct[:300] = True  # accuracy exactly 0.75 in the [0.7-0.8) bin
    ece, _ = cg.ece_top_label(conf, correct)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_ece_overconfident():
    conf = np.full(200, 0.995)
    correct = np.zeros(200, dtype=bool)
    correct[:100] = True  # confidence 0.995, accuracy 0.5
    ece, _ = cg.ece_top_label(conf, correct)
    assert ece == pytest.approx(0.495, abs=1e-12)


def test_ece_weighted_by_bin_population():
    conf = np.array([0.95] * 90 + [0.65] * 10)
    correct = np.array([True] * 90 + [False] * 10)  # gaps: 0.05 (90%), 0.65 (10%)
    ece, _ = cg.ece_top_label(conf, correct)
    assert ece == pytest.approx(0.9 * 0.05 + 0.1 * 0.65, abs=1e-12)


# --- gate logic (synthetic per-fold results, no model training) --------------

def make_results(base_vals, cand_vals, metric="f1_macro", conf=None, correct=None):
    def pack(vals):
        return {"per_fold": pd.DataFrame({metric: vals}),
                "conf": conf if conf is not None else np.array([0.9]),
                "correct": correct if correct is not None else np.array([True])}
    return {cg.BASELINE: pack(base_vals), cg.CANDIDATE: pack(cand_vals)}


def test_gate_pass_higher_is_better():
    r = make_results([0.90, 0.91, 0.92], [0.93, 0.94, 0.95])
    assert cg.run_gate(r, "f1_macro", 0.0, None) == 0


def test_gate_fail_higher_is_better():
    r = make_results([0.93, 0.94, 0.95], [0.90, 0.91, 0.92])
    assert cg.run_gate(r, "f1_macro", 0.0, None) == 1


def test_gate_lower_is_better_direction():
    # candidate has LOWER ece = better -> must PASS
    r = make_results([0.10, 0.12, 0.11], [0.03, 0.04, 0.05], metric="ece")
    assert cg.run_gate(r, "ece", 0.0, None) == 0
    # and the reverse must FAIL
    r = make_results([0.03, 0.04, 0.05], [0.10, 0.12, 0.11], metric="ece")
    assert cg.run_gate(r, "ece", 0.0, None) == 1


def test_gate_epsilon_tolerance():
    r = make_results([0.900, 0.900, 0.900], [0.895, 0.895, 0.895])
    assert cg.run_gate(r, "f1_macro", 0.0, None) == 1
    assert cg.run_gate(r, "f1_macro", 0.01, None) == 0


def test_gate_max_ece_ceiling():
    conf = np.full(100, 0.99)
    correct = np.zeros(100, dtype=bool)
    correct[:50] = True  # pooled ECE ~0.49: candidate wins the metric but is miscalibrated
    r = make_results([0.90] * 3, [0.95] * 3, conf=conf, correct=correct)
    assert cg.run_gate(r, "f1_macro", 0.0, 0.05) == 1
    assert cg.run_gate(r, "f1_macro", 0.0, 0.60) == 0


def test_gate_metric_bal_acc_exists():
    # regression: --gate-metric on balanced accuracy used to KeyError (column is bal_acc)
    r = make_results([0.90, 0.91, 0.92], [0.93, 0.94, 0.95], metric="bal_acc")
    assert cg.run_gate(r, "bal_acc", 0.0, None) == 0


# --- end-to-end exit codes (baseline-only subprocess) -------------------------

def test_e2e_ok_exit0(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    df["label"] = (df["a"] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label")
    assert r.returncode == 0, r.stderr
    assert "baseline_HGB" in r.stdout and "no verdict" in r.stdout


def test_e2e_string_labels_multiclass_exit0(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    df["label"] = rng.choice(["rouge", "vert", "bleu"], 150)
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label", "--calibration")
    assert r.returncode == 0, r.stderr
    assert "reliability curve" in r.stdout


def test_e2e_rare_class_exit2(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    df["label"] = ["A"] * 47 + ["B"] * 3  # 3 members < 5 folds
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label")
    assert r.returncode == 2
    assert "fewer members than folds" in r.stderr


def test_e2e_nan_plus_pca_exit2(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 80)), columns=[f"c{i}" for i in range(80)])
    df.iloc[0, 0] = np.nan  # 80 features > n_pca=64 -> PCA would crash on NaN
    df["label"] = rng.choice([0, 1], 100)
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label")
    assert r.returncode == 2
    assert "PCA" in r.stderr


def test_e2e_infinite_values_exit2(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    df.iloc[0, 0] = np.inf
    df["label"] = rng.choice([0, 1], 60)
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label")
    assert r.returncode == 2
    assert "infinite" in r.stderr


def test_e2e_non_numeric_exit2(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0] * 10, "b": ["x", "y"] * 20,
                       "label": [0, 1] * 20})
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label")
    assert r.returncode == 2
    assert "non-numeric" in r.stderr


def test_e2e_require_candidate_exit2(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    df["label"] = rng.choice([0, 1], 60)
    r = run_script("--csv", write_csv(tmp_path, df), "--target", "label", "--require-candidate")
    assert r.returncode == 2
    assert "require-candidate" in r.stderr


# --- plugin candidate (--candidate) -------------------------------------------

def binary_csv(tmp_path: Path, n: int = 120) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    df["label"] = (df["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return write_csv(tmp_path, df)


def test_e2e_candidate_plugin_verdict(tmp_path):
    # epsilon=1.0 -> the delta condition always passes: this asserts the plumbing
    # (import, fit, paired gate, naming), not which model wins
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "sklearn.ensemble.RandomForestClassifier",
                   "--candidate-args", '{"n_estimators": 30}', "--epsilon", "1.0")
    assert r.returncode == 0, r.stderr
    assert "RandomForestClassifier" in r.stdout and "PASS" in r.stdout


def test_e2e_candidate_bad_import_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "not.a.module:Nope")
    assert r.returncode == 2
    assert "cannot be imported" in r.stderr


def test_e2e_candidate_without_predict_proba_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "sklearn.svm.LinearSVC")
    assert r.returncode == 2
    assert "predict_proba" in r.stderr


def test_e2e_candidate_args_bad_json_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "sklearn.ensemble.RandomForestClassifier",
                   "--candidate-args", "{nope")
    assert r.returncode == 2
    assert "JSON" in r.stderr


def test_e2e_candidate_args_without_candidate_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate-args", '{"n_estimators": 30}')
    assert r.returncode == 2
    assert "requires --candidate" in r.stderr


# --- JSON report (--json) -----------------------------------------------------

def test_e2e_json_report_baseline_only(tmp_path):
    out = tmp_path / "report.json"
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label", "--json", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == 2
    assert doc["gate"] is None and doc["exit_code"] == 0
    assert doc["temperature"] is None  # --temperature not passed
    m = doc["models"]["baseline_HGB"]
    assert set(m["metrics"]) == set(cg.METRICS)  # includes brier
    assert all(len(v) == 5 for v in m["per_fold"].values())  # default --folds 5
    assert isinstance(m["pooled_ece"], float)
    assert isinstance(m["classwise_ece"]["mean"], float)
    assert len(m["classwise_ece"]["per_class"]) == 2  # binary_csv -> 2 classes


def test_e2e_json_report_with_candidate(tmp_path):
    out = tmp_path / "report.json"
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "sklearn.ensemble.RandomForestClassifier",
                   "--candidate-args", '{"n_estimators": 30}',
                   "--epsilon", "1.0", "--json", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    g = doc["gate"]
    assert g["candidate"] == "RandomForestClassifier" and g["baseline"] == "baseline_HGB"
    assert g["pass"] is True and doc["exit_code"] == 0
    assert isinstance(g["delta_mean"], float) and isinstance(g["pooled_ece"], float)


# --- P2: Brier ----------------------------------------------------------------

def test_brier_perfect_wrong_uniform():
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cg.brier_multiclass(perfect, np.array([0, 1])) == pytest.approx(0.0)
    wrong = np.array([[0.0, 1.0]])
    assert cg.brier_multiclass(wrong, np.array([0])) == pytest.approx(2.0)  # max penalty
    uniform = np.array([[0.5, 0.5]])
    assert cg.brier_multiclass(uniform, np.array([0])) == pytest.approx(0.5)


def test_gate_metric_brier_lower_is_better():
    r = make_results([0.30, 0.32, 0.31], [0.10, 0.12, 0.11], metric="brier")
    assert cg.run_gate(r, "brier", 0.0, None) == 0
    r = make_results([0.10, 0.12, 0.11], [0.30, 0.32, 0.31], metric="brier")
    assert cg.run_gate(r, "brier", 0.0, None) == 1


# --- P2: equal-mass bins --------------------------------------------------------

def test_bin_edges_mass_balances_population():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 1000)
    edges = cg.bin_edges(conf, 10, "mass")
    counts = np.histogram(conf, edges)[0]
    assert len(edges) == 11 and counts.min() >= 90  # ~100 per bin
    # heavy ties near 1.0: duplicate quantiles collapse instead of crashing
    clustered = np.concatenate([np.full(950, 0.99), rng.uniform(0.5, 0.9, 50)])
    edges = cg.bin_edges(clustered, 10, "mass")
    assert edges[0] == 0.0 and edges[-1] == 1.0 and len(edges) >= 2
    ece, rows = cg.ece_top_label(clustered, np.ones(1000, dtype=bool), 10, "mass")
    assert np.isfinite(ece) and rows


def test_ece_equal_mass_known_value():
    conf = np.array([0.6] * 50 + [0.9] * 50)
    correct = np.array([True] * 25 + [False] * 25 + [True] * 40 + [False] * 10)
    # 2 mass bins split at the median -> bin1: conf .6 acc .5, bin2: conf .9 acc .8
    ece, _ = cg.ece_top_label(conf, correct, 2, "mass")
    assert ece == pytest.approx(0.5 * 0.1 + 0.5 * 0.1, abs=1e-9)


# --- P2: classwise-ECE ----------------------------------------------------------

def test_classwise_ece_known_values():
    proba = np.tile([0.7, 0.3], (100, 1))
    y = np.array([0] * 70 + [1] * 30)  # frequencies match the probabilities
    mean, per_class = cg.classwise_ece(proba, y)
    assert mean == pytest.approx(0.0, abs=1e-9)
    y = np.array([0] * 50 + [1] * 50)  # p(class0)=0.7 but freq=0.5 -> gap 0.2 each
    mean, per_class = cg.classwise_ece(proba, y)
    assert per_class[0] == pytest.approx(0.2, abs=1e-9)
    assert per_class[1] == pytest.approx(0.2, abs=1e-9)


def test_gate_max_classwise_ece_ceiling():
    r = make_results([0.90] * 3, [0.95] * 3)
    proba = np.tile([0.9, 0.1], (200, 1))
    y_idx = np.array([0] * 120 + [1] * 80)  # p=0.9 vs freq 0.6 -> worst class ECE 0.3
    r[cg.CANDIDATE].update(proba=proba, y_idx=y_idx, classes=np.array(["A", "B"]))
    g = cg.compute_gate(r, "f1_macro", 0.0, None, max_classwise_ece=0.05)
    assert g["classwise_pass"] is False and g["pass"] is False
    assert g["classwise_worst"]["ece"] == pytest.approx(0.3, abs=1e-9)
    g = cg.compute_gate(r, "f1_macro", 0.0, None, max_classwise_ece=0.5)
    assert g["classwise_pass"] is True and g["pass"] is True


def test_e2e_max_classwise_ece_exit_codes(tmp_path):
    csv = binary_csv(tmp_path)
    common = ("--csv", csv, "--target", "label",
              "--candidate", "sklearn.ensemble.RandomForestClassifier",
              "--candidate-args", '{"n_estimators": 30}', "--epsilon", "1.0")
    r = run_script(*common, "--max-classwise-ece", "1.0")
    assert r.returncode == 0, r.stderr
    assert "classwise-ECE ceiling" in r.stdout and "overall -> PASS" in r.stdout
    r = run_script(*common, "--max-classwise-ece", "0.0000001")
    assert r.returncode == 1
    assert "overall -> FAIL" in r.stdout


# --- P2: temperature scaling ----------------------------------------------------

def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 3))
    p_true = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = (rng.random(4000)[:, None] > np.cumsum(p_true, axis=1)).sum(axis=1)
    over = cg.apply_temperature(p_true, 0.5)  # sharpened by 2 -> needs T ~= 2 to undo
    t = cg.fit_temperature(over, y)
    assert 1.6 < t < 2.5
    calibrated = cg.fit_temperature(p_true, y)  # already calibrated -> T ~= 1
    assert 0.8 < calibrated < 1.25


def test_apply_temperature_preserves_argmax():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(4), size=200)
    for t in (0.3, 3.0):
        assert (cg.apply_temperature(p, t).argmax(1) == p.argmax(1)).all()


def test_e2e_temperature_diagnostic(tmp_path):
    out = tmp_path / "report.json"
    r = run_script("--csv", binary_csv(tmp_path, n=160), "--target", "label",
                   "--candidate", "sklearn.ensemble.RandomForestClassifier",
                   "--candidate-args", '{"n_estimators": 30}', "--epsilon", "1.0",
                   "--temperature", "--json", str(out))
    assert r.returncode == 0, r.stderr
    assert "temperature scaling" in r.stdout
    doc = json.loads(out.read_text())
    t = doc["temperature"]["RandomForestClassifier"]
    assert len(t["t_per_fold"]) == 5 and t["t_mean"] > 0
    assert isinstance(t["ece_before"], float) and isinstance(t["ece_after"], float)


# --- P2: misc CLI ---------------------------------------------------------------

def test_e2e_gate_metric_brier_and_mass_binning(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--candidate", "sklearn.ensemble.RandomForestClassifier",
                   "--candidate-args", '{"n_estimators": 30}', "--epsilon", "1.0",
                   "--gate-metric", "brier", "--ece-binning", "mass", "--calibration")
    assert r.returncode == 0, r.stderr
    assert "[gate] brier:" in r.stdout
    assert "classwise-ECE mean=" in r.stdout
    assert "mass bins" in r.stdout


def test_e2e_ece_bins_too_small_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label", "--ece-bins", "1")
    assert r.returncode == 2
    assert "--ece-bins" in r.stderr


def test_e2e_plot_without_matplotlib_exit2(tmp_path):
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label",
                   "--plot", str(tmp_path / "rel.png"), block=("tabicl", "matplotlib"))
    assert r.returncode == 2
    assert "matplotlib" in r.stderr


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib not installed")
def test_e2e_plot_written(tmp_path):
    png = tmp_path / "rel.png"
    r = run_script("--csv", binary_csv(tmp_path), "--target", "label", "--plot", str(png))
    assert r.returncode == 0, r.stderr
    assert png.exists() and png.stat().st_size > 1000
