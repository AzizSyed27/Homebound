# train_models.py
# Trains Logistic Regression, Decision Tree, and Random Forest.
# Calibrates probabilities for varied, realistic outputs.
# Saves each model and the best model by ROC AUC.

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from preprocess import prepare_train_test, save_artifacts


#DATA_PATH = os.environ.get("DATA_PATH", "data/bike_thefts.csv")
DATA_PATH="C:\\Users\\azizs\\OneDrive\\Desktop\\bicycle-theft-prediction-main-final\\bicycle-theft-prediction-main\\data\\bike_thefts.csv"
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
RANDOM_STATE = 42

def _calibrate_prefit(model, X_cal, y_cal, method="sigmoid"):
    # sklearn >= 1.6 removed cv="prefit"; the supported way to calibrate an
    # already-fitted estimator is to wrap it in FrozenEstimator. Fall back to
    # the old cv="prefit" API on older sklearn versions.
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Prepare data (fit encoder/scaler on train only)
    X_train, X_test, y_train, y_test, encoder, scaler, keepers, feature_names = prepare_train_test(
        DATA_PATH, test_size=0.2, random_state=RANDOM_STATE, top_k=12
    )

    # Create a small calibration split from the training fold
    from sklearn.model_selection import train_test_split
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    models = {}

    # 1) Logistic Regression (balanced)
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    lr.fit(X_tr, y_tr)
    lr = _calibrate_prefit(lr, X_cal, y_cal, method="sigmoid")
    models["logistic_regression"] = lr

    # 2) Decision Tree (shallow to avoid overfitting; balanced)
    dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    dt.fit(X_tr, y_tr)
    dt = _calibrate_prefit(dt, X_cal, y_cal, method="sigmoid")
    models["decision_tree"] = dt

    # 3) Random Forest (more expressive; balanced_subsample)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_tr, y_tr)
    rf = _calibrate_prefit(rf, X_cal, y_cal, method="sigmoid")
    models["random_forest"] = rf

    # Evaluate and save
    rows = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)
        rows.append({"model": name, "accuracy": acc, "auc": auc, "avg_precision": ap})

        # Persist model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    # Save preprocessing artifacts
    save_artifacts(MODELS_DIR, encoder, scaler, keepers, feature_names)

    # Pick best by AUC, then AP, then ACC
    df = pd.DataFrame(rows).sort_values(by=["auc", "avg_precision", "accuracy"], ascending=False)
    df.to_csv(os.path.join(REPORTS_DIR, "summary_metrics.csv"), index=False)

    best_name = df.iloc[0]["model"]
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(str(best_name))

    print("=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nBest model: {best_name}")


if __name__ == "__main__":
    main()
