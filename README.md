# tabicl-calibration-gate

A small, reusable **calibration gate** for tabular classifiers — TabICL-first, model-agnostic, 100 % local.

Before you trust a model's probabilities, you should check one thing: **is its confidence calibrated?** A model can be accurate yet pathologically over-confident — its predicted probabilities then mean nothing, and any confidence threshold you set downstream (auto-label above 0.8, escalate below 0.5…) is noise.

This gate runs stratified cross-validation and reports, per model:

- accuracy, balanced accuracy, **f1-macro**, f1-weighted, log-loss
- **Expected Calibration Error (ECE)** + a text **reliability curve**
- a saturation flag (only raised when over-confidence comes *with* miscalibration)

and **gates** a candidate model against a baseline (`HistGradientBoosting`) on a chosen metric, returning a non-zero exit code on regression — so it drops straight into CI.

It uses [TabICL](https://github.com/soda-inria/tabicl) (a SOTA tabular foundation model) as the candidate model when installed; otherwise it runs baseline-only, which is still a valid calibration check. Everything runs offline.

## Why calibration, not just accuracy

Two models can hit the same accuracy while one is honest about its uncertainty and the other isn't:

```
=== calibration: baseline_HGB ===   ECE=0.0197
  [0.9-1.0) n=545  conf=0.998  acc=0.985  gap=-0.012     # mildly over-confident

=== calibration: TabICL ===         ECE=0.0090
  [0.9-1.0) n=525  conf=0.995  acc=0.994  gap=-0.001     # confidence ≈ accuracy
```

Lower ECE = the displayed confidence matches the real accuracy. That makes the confidence a usable decision variable — essential for selective automation, risk scoring, and human-in-the-loop triage.

## Install

```bash
pip install numpy "scikit-learn>=1.3" pandas
pip install tabicl   # optional — enables the TabICL candidate model
```

Python 3.10+. Feature columns must be numeric (encode categoricals first); the gate fails loud with a clear message otherwise.

## Usage

```bash
# sklearn-bundled dataset + calibration diagnostics
python calibration_gate.py --dataset breast_cancer --calibration

# your own data
python calibration_gate.py --csv data.csv --target label --calibration

# high-dimensional inputs (e.g. embeddings) get auto-PCA before the model
python calibration_gate.py --csv embeddings.csv --target y --n-pca 64

# Apple Silicon
python calibration_gate.py --dataset wine --device mps --calibration
```

Bundled datasets: `breast_cancer`, `wine`, `iris`, `digits`. The gate is multiclass-safe.

### Example output

```
dataset: n=569, features=30, classes=2 | folds=5

              accuracy  bal_acc  f1_macro  f1_weighted  log_loss  wall_s
model
baseline_HGB    0.9701   0.9646    0.9677       0.9700    0.1047  2.71
TabICL          0.9824   0.9802    0.9812       0.9824    0.0613  3.91

=== calibration: TabICL ===
mean confidence=0.976 | accuracy=0.982 | ECE=0.0090
share conf>0.99 = 82.6%
reliability curve (confidence vs accuracy per bin):
  [0.6-0.7) n=   15  conf=0.645  acc=0.733  gap=+0.088
  [0.8-0.9) n=   17  conf=0.848  acc=1.000  gap=+0.152
  [0.9-1.0) n=  525  conf=0.995  acc=0.994  gap=-0.001

[gate] f1_macro: TabICL=0.9812 vs baseline=0.9677 (delta=+0.0135, epsilon=0.0) -> PASS
```

## In CI

The process exits `1` if the candidate (TabICL) fails the gate, `0` otherwise:

```yaml
- run: python calibration_gate.py --dataset breast_cancer --gate-metric f1_weighted --epsilon 0.0
```

> If TabICL is **not** installed, the gate prints baseline-only and exits `0` (no verdict) — a CI job would pass green with no gate ever applied. Install `tabicl` in CI if you want the gate enforced.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--dataset` / `--csv` + `--target` | `breast_cancer` | data source (bundled or your CSV) |
| `--n-pca` | `64` | PCA components when `n_features` exceeds it (`0` disables) |
| `--folds` | `5` | stratified CV folds |
| `--gate-metric` | `f1_macro` | metric the PASS/FAIL is decided on |
| `--epsilon` | `0.0` | tolerance: candidate must be `>= baseline - epsilon` |
| `--calibration` | off | print ECE + reliability curve |
| `--n-estimators` | `8` | TabICL ensemble size |
| `--device` | auto | TabICL device (`mps`, `cpu`) |
| `--random-state` | `42` | seed for reproducibility |

## Background

Built while testing whether a local tabular foundation model is genuinely usable for data that can't leave the machine. Write-up: <https://karaibart.fr/tabicl.html>.

Credit to [soda-inria/tabicl](https://github.com/soda-inria/tabicl) for the model.

## License

MIT — see [LICENSE](LICENSE).
