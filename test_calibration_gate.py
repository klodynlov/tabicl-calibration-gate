"""Tests for calibration_gate. Run: python -m pytest test_calibration_gate.py -q

TabICL is NOT required: end-to-end tests force the baseline-only path (tabicl
import blocked in the subprocess), and the gate logic is tested directly with
synthetic per-fold results.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import calibration_gate as cg

SCRIPT = Path(__file__).with_name("calibration_gate.py")


def run_script(*argv: str) -> subprocess.CompletedProcess:
    """Run the gate in a subprocess with the tabicl import blocked (baseline-only,
    deterministic even in an env where tabicl is installed)."""
    code = (f"import sys; sys.modules['tabicl'] = None; "
            f"exec(compile(open(r'{SCRIPT}').read(), r'{SCRIPT}', 'exec'))")
    return subprocess.run([sys.executable, "-c", code, *argv],
                          capture_output=True, text=True, timeout=120)


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
