# evaluate_models.py
# Loads saved models and preprocessing artifacts. Evaluates on the held-out test fold.
# Saves confusion matrices, ROC, and PR curves, plus a small "what-if" sweep to show variability.

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score
)

from preprocess import transform_for_eval


DATA_PATH="C:\\Users\\azizs\\OneDrive\\Desktop\\bicycle-theft-prediction-main-final\\bicycle-theft-prediction-main\\data\\bike_thefts.csv"
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
RANDOM_STATE = 42


def _ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _plot_confusion(cm: np.ndarray, title: str, path: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_curves(y_true, y_prob, name: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC: {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"roc_{name}.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.title(f"Precision–Recall: {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"pr_{name}.png"))
    plt.close()


def main():
    _ensure_dirs()

    # Load artifacts
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.joblib"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    keepers = joblib.load(os.path.join(MODELS_DIR, "cat_keepers.joblib"))

    # Prepare test fold using the saved preprocessor
    X_test, y_test = transform_for_eval(
        DATA_PATH, encoder, scaler, keepers, test_size=0.2, random_state=RANDOM_STATE
    )

    # Evaluate all available models
    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib") and f.split(".")[0] in {"logistic_regression", "decision_tree", "random_forest"}
    ]
    rows = []

    for mf in sorted(model_files):
        name = mf.replace(".joblib", "")
        model = joblib.load(os.path.join(MODELS_DIR, mf))

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
        ap = average_precision_score(y_test, y_prob)
        rows.append({"model": name, "accuracy": acc, "auc": roc_auc, "avg_precision": ap})

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        _plot_confusion(cm, f"Confusion Matrix: {name}", os.path.join(REPORTS_DIR, f"cm_{name}.png"))
        _plot_curves(y_test, y_prob, name)

    # Summary CSV
    df = pd.DataFrame(rows).sort_values(by=["auc", "avg_precision", "accuracy"], ascending=False)
    df.to_csv(os.path.join(REPORTS_DIR, "summary_metrics_eval.csv"), index=False)
    print("\n=== Summary (test) ===")
    print(df.to_string(index=False))

    # "Showy" variability sweep (BIKE_COST and OCC_HOUR)
    # We perturb two key features while keeping others at modal/median encoded by feeding through the model that won during training.
    best_name_path = os.path.join(MODELS_DIR, "best_model.txt")
    if os.path.exists(best_name_path):
        best_name = open(best_name_path).read().strip()
    else:
        best_name = df.iloc[0]["model"]
    best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_name}.joblib"))

    # Build a small grid directly in feature space:
    # For a coarse demonstration, vary cost_log and occ_hour while leaving other standardized features at 0.
    # This is not exact human-readable space, but it shows probability shape.
    means = np.zeros(X_test.shape[1], dtype=float)
    probs = []
    for hour in [0, 6, 12, 18, 23]:
        row = means.copy()
        # Indices: by construction, numeric features precede OHE in this order (see preprocess.py):
        # ["occ_month","occ_hour","occ_dow","occ_month_sin","occ_month_cos","occ_hour_sin","occ_hour_cos","is_weekend","report_delay_hours","bike_cost_log"]
        # Then OHE begins.
        H = 1   # index of occ_hour (standardized)
        C = 9   # index of bike_cost_log (standardized)
        for cost_log in [np.log1p(v) for v in [200, 800, 2000, 5000]]:
            tmp = row.copy()
            # Place approximate standardized values: 0 for others; push hour and cost in ±1..2 std ranges for a visible effect.
            tmp[H] = (hour - 12) / 6.0      # rough scaling into std units
            tmp[C] = (cost_log - np.log1p(800)) / 0.7
            p = best_model.predict_proba(tmp.reshape(1, -1))[:, 1][0]
            probs.append({"model": best_name, "hour": hour, "cost": np.exp(cost_log) - 1, "prob": p})

    pd.DataFrame(probs).to_csv(os.path.join(REPORTS_DIR, f"what_if_{best_name}.csv"), index=False)
    print(f"\nSaved variability sweep to reports/what_if_{best_name}.csv")


if __name__ == "__main__":
    main()
