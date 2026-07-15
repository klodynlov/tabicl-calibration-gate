# tabicl-calibration-gate

[![ci](https://github.com/klodynlov/tabicl-calibration-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/klodynlov/tabicl-calibration-gate/actions/workflows/ci.yml)

A small, reusable **calibration gate** for tabular classifiers — TabICL-first, model-agnostic, 100 % local.

Before you trust a model's probabilities, you should check one thing: **is its confidence calibrated?** A model can be accurate yet pathologically over-confident — its predicted probabilities then mean nothing, and any confidence threshold you set downstream (auto-label above 0.8, escalate below 0.5…) is noise.

This gate computes out-of-fold predictions **once** per model (stratified CV) and derives everything from that single pass:

- accuracy, balanced accuracy, **f1-macro**, f1-weighted, log-loss, **Expected Calibration Error (ECE)** — all in the main table
- a text **reliability curve** (`--calibration`, no extra compute)
- a saturation flag (only raised when over-confidence comes *with* miscalibration)

and **gates** a candidate model against a baseline (`HistGradientBoosting`) on a chosen metric — **including `ece` and `log_loss`** — using the **paired per-fold delta**: both models are evaluated on identical folds, so the fold-wise difference is a paired sample, far less noisy than comparing two means. An optional `--max-ece` adds an absolute calibration ceiling. Non-zero exit code on regression — so it drops straight into CI.

It uses [TabICL](https://github.com/soda-inria/tabicl) (a SOTA tabular foundation model) as the candidate model when installed — but **any sklearn-compatible classifier** can be the candidate via `--candidate module.path:ClassName`. Without a candidate it runs baseline-only, which is still a valid calibration check. Everything runs offline.

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
pip install .            # installs the `tabgate` command
pip install ".[tabicl]"  # + the TabICL candidate model (optional)
```

(Or dependency-only, no install: `pip install numpy "scikit-learn>=1.3" pandas` and run `python calibration_gate.py …` — same CLI.)

Python 3.10+. Feature columns must be numeric (encode categoricals first); the gate fails loud (exit `2`, clear message) otherwise — same for classes rarer than `--folds`, infinite values, and NaN when PCA would apply.

## Usage

```bash
# sklearn-bundled dataset + calibration diagnostics
tabgate --dataset breast_cancer --calibration

# your own data
tabgate --csv data.csv --target label --calibration

# high-dimensional inputs (e.g. embeddings) get auto-PCA before the model
tabgate --csv embeddings.csv --target y --n-pca 64

# Apple Silicon
tabgate --dataset wine --device mps --calibration

# any sklearn-compatible classifier as the candidate (replaces TabICL)
tabgate --dataset breast_cancer \
    --candidate sklearn.ensemble.RandomForestClassifier \
    --candidate-args '{"n_estimators": 200}' --gate-metric ece

# machine-readable report (written for PASS and FAIL alike)
tabgate --csv data.csv --target y --gate-metric ece --max-ece 0.05 --json report.json
```

Bundled datasets: `breast_cancer`, `wine`, `iris`, `digits`. The gate is multiclass-safe.

### Example output

```
dataset: n=569, features=30, classes=2 | folds=5

              accuracy  bal_acc  f1_macro  f1_weighted  log_loss    ece  wall_s
model
baseline_HGB    0.9701   0.9646    0.9677       0.9700    0.1047 0.0288  1.3178
TabICL          0.9824   0.9802    0.9812       0.9824    0.0613 0.0197  2.5185

=== calibration: TabICL ===
mean confidence=0.976 | accuracy=0.982 | ECE=0.0090 (pooled)
share conf>0.99 = 82.6%
reliability curve (confidence vs accuracy per bin):
  [0.6-0.7) n=   15  conf=0.645  acc=0.733  gap=+0.088
  [0.8-0.9) n=   17  conf=0.848  acc=1.000  gap=+0.152
  [0.9-1.0) n=  525  conf=0.995  acc=0.994  gap=-0.001

[gate] f1_macro: TabICL=0.9812 vs baseline_HGB=0.9677 (epsilon=0.0)
[gate] paired delta per fold ('+' = candidate better): +0.0135 ± 0.0158 (min +0.0000, 5 folds) -> PASS
```

(The table's `ece` column is the per-fold mean — the value the gate compares; the `--calibration` section shows the pooled-OOF ECE alongside the curve. They differ slightly by construction.)

## In CI

Exit codes: **`0`** gate PASS (or baseline-only, no verdict) · **`1`** gate FAIL · **`2`** invalid input (non-numeric features, class rarer than `--folds`, NaN where PCA would apply, infinite values, all-NaN report) — bad data can never turn a CI job green.

```yaml
- run: tabgate --csv data.csv --target label --gate-metric ece --max-ece 0.05 --require-candidate --json report.json
- uses: actions/upload-artifact@v4
  with: { name: calibration-report, path: report.json }
```

> If TabICL is **not** installed, the gate prints baseline-only and exits `0` (no verdict). Pass `--require-candidate` to make that case fail loud (exit `2`) instead — recommended in CI so a broken install can't pass silently. (`--candidate` needs no such guard: a missing plugin class is already exit `2`.)

`--json` writes the full report — per-model metrics, per-fold values, pooled ECE, the gate verdict with paired deltas, and the exit code — for trend tracking across runs. This repo's [own workflow](.github/workflows/ci.yml) dogfoods all three exit codes on every push.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--dataset` / `--csv` + `--target` | `breast_cancer` | data source (bundled or your CSV) |
| `--n-pca` | `64` | PCA components when `n_features` exceeds it (`0` disables) |
| `--folds` | `5` | stratified CV folds |
| `--gate-metric` | `f1_macro` | metric the PASS/FAIL is decided on: `f1_macro`, `f1_weighted`, `bal_acc`, `accuracy`, `log_loss`, `ece` (the last two gate lower-is-better) |
| `--epsilon` | `0.0` | tolerance on the paired mean delta: candidate may be worse by up to epsilon |
| `--max-ece` | off | absolute ceiling on the candidate's pooled ECE (second gate condition) |
| `--candidate` | TabICL | import path of a plugin candidate (`module.path:ClassName`, sklearn-compatible, must have `predict_proba`) |
| `--candidate-args` | `{}` | constructor kwargs for `--candidate`, as a JSON object |
| `--json` | off | write a machine-readable JSON report (PASS and FAIL alike) |
| `--require-candidate` | off | exit `2` when `tabicl` is missing instead of green baseline-only |
| `--calibration` | off | print reliability curves (ECE is always computed and gateable) |
| `--n-estimators` | `8` | TabICL ensemble size |
| `--device` | auto | TabICL device (`mps`, `cpu`) |
| `--random-state` | `42` | seed for reproducibility |

## Tests

```bash
python -m pytest test_calibration_gate.py -q   # no tabicl needed (baseline-only path is forced)
```

Covers: ECE properties (perfectly calibrated → 0, over-confident → known value, bin weighting), gate direction for lower-is-better metrics, epsilon tolerance, `--max-ece` ceiling, the `--candidate` plugin path (verdict, bad import, missing `predict_proba`, bad JSON args), the `--json` report schema, and end-to-end exit codes (`0`/`2`) on the edge cases above. The GitHub Actions workflow additionally installs the package and exercises all three exit codes through the `tabgate` entry point.

## Background

Built while testing whether a local tabular foundation model is genuinely usable for data that can't leave the machine. Write-up: <https://karaibart.fr/tabicl.html>.

Credit to [soda-inria/tabicl](https://github.com/soda-inria/tabicl) for the model.

## License

MIT — see [LICENSE](LICENSE).
