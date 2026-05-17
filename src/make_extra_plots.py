# make_extra_plots.py
# Produces two summary visualizations for the README:
#   1) reports/model_comparison.png  - ROC + PR overlay for all three models
#   2) reports/what_if_plot.png      - what-if sweep: P(returned) vs bike cost, one line per hour
#
# Reuses the saved preprocessing artifacts and the same stratified test fold as
# evaluate_models.py, so the curves are consistent with reports/summary_metrics_eval.csv.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from preprocess import transform_for_eval

DATA_PATH = os.environ.get("DATA_PATH", "data/bike_thefts.csv")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
RANDOM_STATE = 42
MODELS = ["random_forest", "decision_tree", "logistic_regression"]


def _model_comparison():
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    keepers = joblib.load(os.path.join(MODELS_DIR, "cat_keepers.joblib"))
    X_test, y_test = transform_for_eval(
        DATA_PATH, encoder, scaler, keepers, test_size=0.2, random_state=RANDOM_STATE
    )

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    for name in MODELS:
        model = joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC - all models")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall - all models")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")

    fig.suptitle("Model comparison on the held-out test fold")
    fig.tight_layout()
    out = os.path.join(REPORTS_DIR, "model_comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def _what_if_plot():
    df = pd.read_csv(os.path.join(REPORTS_DIR, "what_if_random_forest.csv"))
    plt.figure(figsize=(8, 5))
    for hour, sub in df.groupby("hour"):
        sub = sub.sort_values("cost")
        plt.plot(sub["cost"], sub["prob"] * 100.0, marker="o", label=f"hour {hour}")
    plt.title("What-if: P(returned) vs bike cost, by occurrence hour\n(Random Forest, other features at mean)")
    plt.xlabel("Bike cost (CAD)")
    plt.ylabel("Predicted probability of return (%)")
    plt.legend(title="Occurrence hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "what_if_plot.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    _model_comparison()
    _what_if_plot()


if __name__ == "__main__":
    main()
