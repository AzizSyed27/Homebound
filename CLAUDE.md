# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Homebound** ("Will my bike come home?") — a supervised ML pipeline predicting
whether a stolen bicycle is recovered/returned
(Toronto Police data, `data/bike_thefts.csv`, ~37,928 rows). Pipeline:
preprocess → train (3 calibrated models) → evaluate → Flask serving UI. There is
no test suite, build step, or linter — it's run as scripts.

## Commands

Run everything from the **project root** with the miniconda Python (no venv exists):

```bash
pip install -r requirements.txt          # deps install into miniconda site-packages

# Train: writes models/*.joblib, models/best_model.txt, reports/summary_metrics.csv
python -c "import sys; sys.path.insert(0,'src'); import train_models as t; t.DATA_PATH='data/bike_thefts.csv'; t.main()"

# Evaluate: writes reports/summary_metrics_eval.csv, cm_/roc_/pr_*.png, what_if_random_forest.csv
python -c "import sys; sys.path.insert(0,'src'); import evaluate_models as e; e.DATA_PATH='data/bike_thefts.csv'; e.main()"

# Summary plots: reports/model_comparison.png, reports/what_if_plot.png
python -c "import sys; sys.path.insert(0,'src'); import make_extra_plots as m; m.DATA_PATH='data/bike_thefts.csv'; m.main()"

# Serve the prediction UI + API on http://127.0.0.1:5001/
python src/app.py        # PORT env overrides 5001
```

**Critical gotcha:** `src/train_models.py` (line ~22) and `src/evaluate_models.py`
(line ~20) **hardcode a machine-specific absolute `DATA_PATH`** that does not
exist in this repo. Running them directly (`python src/train_models.py`) fails.
Always override `DATA_PATH` as shown above (the commented-out
`os.environ.get("DATA_PATH", "data/bike_thefts.csv")` line shows the intended
default but is not active). Do not "fix" this by editing the scripts unless asked.

## Architecture

**Pipeline data contract (the core thing to understand).** `src/preprocess.py`
is the single source of truth for feature engineering. It splits *before* fitting
the scaler/encoder (no leakage), then serializes four coupled artifacts to
`models/`: `encoder.joblib`, `scaler.joblib`, `cat_keepers.joblib`,
`feature_names.json`. The final feature vector is **strictly ordered**: 10 scaled
numeric features (`NUMERIC_ORDER`) followed by one-hot categoricals
(`CAT_ORDER`). Any code producing a feature row for the model must reproduce this
exact order and the same transforms (cyclical sin/cos time encoding, `log1p`
bike cost clipped 0–20000, top-k=12 category keep-lists folding rare values to
`"Other"`). `train_models.py` selects the best model by AUC → AP → accuracy and
writes the name to `models/best_model.txt`; `app.py` independently re-derives the
best model by reading `reports/summary_metrics.csv`.

**`src/app.py` re-implements preprocess's feature engineering by hand**
(`build_feature_row`, `parse_occ_fields`, etc.) rather than importing
`preprocess`. These two implementations must stay in sync — if you change
feature engineering in `preprocess.py`, mirror it in `app.py` or predictions
will silently diverge from training.

**Frontend.** UI lives in `src/templates/index.html` + `src/static/{app.js,styles.css}`
(Flask default folders; `app = Flask(__name__)` in `src/`). `app.js` fetches
`GET /meta` on load to populate the categorical dropdowns with the model's real
known values (from `cat_keepers.joblib`) plus human-readable labels from the
`CODE_LABELS` map in `app.py`. This dropdown approach exists specifically to
prevent users entering values that silently fold to `"Other"`. API routes:
`/` (UI), `/health`, `/meta`, `/predict` (POST, returns `{prediction, proba,
model_name, features}` or `400 {error}`). `styles.css` is a high-fidelity theme
from a Claude Design handoff (cobalt & coral, Plus Jakarta Sans + JetBrains
Mono). It also carries unused-but-intentional design-system options the
prototype offered as "Tweaks": alternate meter styles (`.meter-seg`,
`.meter-grad`) and the compact-density values — kept as swap-in options, not
dead code to prune. Only `BIKE_TYPE`/`BIKE_COLOUR`/`DIVISION` get friendly
labels via `CODE_LABELS` in `app.py`; `LOCATION_TYPE`/`PREMISES_TYPE`/
`PRIMARY_OFFENCE` show raw strings (extend `CODE_LABELS` if nicer labels are
wanted). The page calls the real Flask endpoints — the prototype's mock APIs
are not shipped.

## Environment constraints

This repo was written against older pandas/sklearn; compatibility fixes are
already applied and must be preserved:
- `preprocess.py`: no `infer_datetime_format=` (removed in pandas ≥2.0).
- `train_models.py` `_calibrate_prefit`: uses `FrozenEstimator` (sklearn ≥1.6
  removed `CalibratedClassifierCV(cv="prefit")`), with a fallback to the old API.

## Known issues (documented, not yet fixed)

- `src/client.py` has a hard `SyntaxError` (line 29: `args.with-report-delay`
  should be `args.with_report_delay`) — it cannot run at all.
- `requirements.txt` lists `xgboost`, `imbalanced-learn`, `seaborn` but only
  Logistic Regression, Decision Tree, Random Forest are trained.

## Interpreting results

Severe class imbalance (~99% not returned) makes **accuracy misleading** — a
trivial baseline scores ~99%. Judge models by ROC AUC and Average Precision.
Decision Tree and Logistic Regression predict zero positives at the 0.5
threshold; Random Forest is the only model with positive-class signal and is the
selected best (AUC ≈ 0.68, AP ≈ 0.17). See `README.md` for the full results
narrative and `reports/` for the generated charts.
