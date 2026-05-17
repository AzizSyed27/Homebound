import os
import json
import math
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import numpy as np
import joblib
from flask import Flask, request, jsonify, make_response, render_template

# ----------------------------
# Configuration
# ----------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_CSV = os.environ.get("REPORTS_CSV", "reports/summary_metrics.csv")
MODEL_FILE = os.environ.get("MODEL_FILE")  # optional explicit model, e.g. "random_forest.joblib"
# If a client does not send report_dt, use this fallback (hours)
DEFAULT_REPORT_DELAY_H = float(os.environ.get("DEFAULT_REPORT_DELAY_H", "0"))

# You trained 3 models; keep order accordingly
PREFERRED_ORDER = ["random_forest", "logistic_regression", "decision_tree"]

# Numeric feature order expected by the scaler (must match training)
NUMERIC_ORDER = [
    "occ_month", "occ_hour", "occ_dow",
    "occ_month_sin", "occ_month_cos",
    "occ_hour_sin", "occ_hour_cos",
    "is_weekend", "report_delay_hours",
    "bike_cost_log"
]

# Categorical columns order expected by the encoder (must match training)
CAT_ORDER = [
    "BIKE_TYPE", "BIKE_COLOUR", "DIVISION",
    "LOCATION_TYPE", "PREMISES_TYPE", "PRIMARY_OFFENCE"
]

# ----------------------------
# Human-readable labels for the opaque Toronto Police category codes.
# Used only for display in /meta; the model still receives the raw codes.
# Best-effort: unknown codes fall back to the raw value.
# ----------------------------
CODE_LABELS = {
    "BIKE_TYPE": {
        "BM": "BMX", "EL": "Electric", "FO": "Folding", "MT": "Mountain",
        "OT": "Other type", "RC": "Racer / Road", "RE": "Recumbent",
        "RG": "Regular / City", "SC": "Scooter", "TA": "Tandem",
        "TO": "Touring", "TR": "Tricycle",
    },
    "BIKE_COLOUR": {
        "BLK": "Black", "BLU": "Blue", "DBL": "Dark Blue", "GRN": "Green",
        "GRY": "Grey", "LBL": "Light Blue", "ONG": "Orange", "PLE": "Purple",
        "RED": "Red", "SIL": "Silver", "WHI": "White",
        "nan": "Unknown / not specified",
    },
}


def _label_for(column: str, value: str) -> str:
    """Return a friendly label for a category value, with sensible fallbacks."""
    overrides = CODE_LABELS.get(column, {})
    if value in overrides:
        return overrides[value]
    if value == "Other":
        return "Other / not listed"
    if column == "DIVISION" and value.startswith("D") and value[1:].isdigit():
        return f"Division {value[1:]}"
    return value


def _clean_value(v) -> str:
    """Coerce a keeper value to a JSON-safe string ('nan' for NaN/None)."""
    if v is None:
        return "nan"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return str(v)


def build_meta() -> dict:
    """Option lists + numeric bounds the frontend needs to render the form."""
    categories = {}
    for col in CAT_ORDER:
        keep = [_clean_value(v) for v in KEEPERS.get(col, [])]
        if "Other" not in keep:
            keep.append("Other")
        # Stable order: training order, de-duplicated, "Other" pushed to the end.
        seen = set()
        ordered = []
        for v in keep:
            if v != "Other" and v not in seen:
                seen.add(v)
                ordered.append(v)
        ordered.append("Other")
        categories[col] = [
            {"value": v, "label": _label_for(col, v)} for v in ordered
        ]
    return {
        "model": MODEL_NAME,
        "categories": categories,
        "numeric_bounds": {"BIKE_COST": [0, 20000]},
        "dow_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }

# ----------------------------
# Load preprocessing artifacts
# ----------------------------
def load_artifacts(models_dir: str):
    enc_path = os.path.join(models_dir, "encoder.joblib")
    sc_path  = os.path.join(models_dir, "scaler.joblib")
    keep_path = os.path.join(models_dir, "cat_keepers.joblib")

    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encoder at {enc_path}")
    if not os.path.exists(sc_path):
        raise FileNotFoundError(f"Missing scaler at {sc_path}")
    if not os.path.exists(keep_path):
        raise FileNotFoundError(f"Missing category keepers at {keep_path}")

    encoder = joblib.load(enc_path)
    scaler  = joblib.load(sc_path)
    keepers: Dict[str, List[str]] = joblib.load(keep_path)
    return encoder, scaler, keepers

# ----------------------------
# Choose and load model
# ----------------------------
def choose_model_file(models_dir: str, reports_csv: str, model_file_env: Optional[str]) -> str:
    if model_file_env:
        path = os.path.join(models_dir, model_file_env)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MODEL_FILE set to '{model_file_env}' but file not found at {path}")
        return path

    # Try reading the summary_metrics.csv produced by training
    try:
        import pandas as pd
        if os.path.exists(reports_csv):
            df = pd.read_csv(reports_csv)
            cols = {c.lower(): c for c in df.columns}
            def col(name): return cols.get(name, None)
            sort_cols = [c for c in [col("auc"), col("avg_precision"), col("accuracy")] if c]
            df_sorted = df.sort_values(by=sort_cols, ascending=False, kind="mergesort")
            best_model_name = str(df_sorted.iloc[0]["model"])
            candidate = os.path.join(models_dir, f"{best_model_name}.joblib")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    # Fallback order
    for name in PREFERRED_ORDER:
        candidate = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"No model file found in {models_dir} matching {PREFERRED_ORDER}")

def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Failed to load model at {model_path}. "
            f"Ensure the runtime has the same libraries as training. Original error: {e}"
        )

# ----------------------------
# Inference helpers
# ----------------------------
def parse_occ_fields(payload: dict) -> Tuple[int, int, int]:
    """Return (occ_month, occ_hour, occ_dow), dow Monday=0..Sunday=6."""
    if all(k in payload for k in ["occ_month", "occ_hour", "occ_dow"]):
        return int(payload["occ_month"]), int(payload["occ_hour"]), int(payload["occ_dow"])
    if "occ_dt" in payload:
        try:
            dt = datetime.fromisoformat(str(payload["occ_dt"]))
        except Exception:
            raise ValueError("Invalid 'occ_dt' format. Use ISO like 'YYYY-MM-DDTHH:MM'.")
        return dt.month, dt.hour, dt.weekday()
    raise ValueError("Provide either occ_month/occ_hour/occ_dow or a single 'occ_dt'.")

def parse_report_delay_hours(payload: dict, occ_year:int, occ_month:int, occ_day:int, occ_hour:int) -> float:
    """
    If client provides 'report_dt' (ISO), compute delay; else return DEFAULT_REPORT_DELAY_H.
    """
    if "report_dt" in payload and payload["report_dt"]:
        try:
            rep = datetime.fromisoformat(str(payload["report_dt"]))
        except Exception:
            raise ValueError("Invalid 'report_dt' format. Use ISO like 'YYYY-MM-DDTHH:MM'.")
        # Build occ datetime (assume local time) with provided components; use day=1 if not provided
        try:
            occ = datetime(occ_year, occ_month, max(1, occ_day), occ_hour, 0)
        except Exception:
            # Fallback if day unknown
            occ = datetime(occ_year, occ_month, 1, occ_hour, 0)
        delay = (rep - occ).total_seconds() / 3600.0
        if delay < 0:
            delay = 0.0
        if delay > 24 * 30:
            delay = 24.0 * 30
        return float(delay)
    return float(DEFAULT_REPORT_DELAY_H)

def map_keepers(value: str, column: str, keepers: Dict[str, List[str]]) -> str:
    keep = keepers.get(column, None)
    if not keep:
        return value
    return value if value in keep else "Other"

def build_feature_row(payload: dict, encoder, scaler, keepers) -> np.ndarray:
    # --- Numeric block (10 features) ---
    occ_month, occ_hour, occ_dow = parse_occ_fields(payload)
    # Occ day is unknown from inputs; use 1 to compute report delay reference
    occ_day = 1
    occ_year = 2000  # not used in model; only for building a datetime for delay baseline
    # If client passed 'occ_dt', parse real year/day for better delay computation
    if "occ_dt" in payload:
        try:
            dt = datetime.fromisoformat(str(payload["occ_dt"]))
            occ_year, occ_month, occ_day, occ_hour, occ_dow = dt.year, dt.month, dt.day, dt.hour, dt.weekday()
        except Exception:
            pass

    # Cyclic encodings
    occ_month_sin = np.sin(2 * np.pi * (occ_month - 1) / 12.0)
    occ_month_cos = np.cos(2 * np.pi * (occ_month - 1) / 12.0)
    occ_hour_sin  = np.sin(2 * np.pi * occ_hour / 24.0)
    occ_hour_cos  = np.cos(2 * np.pi * occ_hour / 24.0)

    is_weekend = 1 if occ_dow >= 5 else 0

    # Report delay
    report_delay_hours = parse_report_delay_hours(payload, occ_year, occ_month, occ_day, occ_hour)

    # Cost -> log1p
    try:
        bike_cost = float(payload["BIKE_COST"])
    except Exception:
        raise ValueError("'BIKE_COST' must be numeric.")
    if bike_cost < 0:
        bike_cost = 0.0
    if bike_cost > 20000:
        bike_cost = 20000.0
    bike_cost_log = float(np.log1p(bike_cost))

    X_num_dict = {
        "occ_month": occ_month, "occ_hour": occ_hour, "occ_dow": occ_dow,
        "occ_month_sin": occ_month_sin, "occ_month_cos": occ_month_cos,
        "occ_hour_sin": occ_hour_sin, "occ_hour_cos": occ_hour_cos,
        "is_weekend": is_weekend,
        "report_delay_hours": report_delay_hours,
        "bike_cost_log": bike_cost_log
    }
    # Order strictly by NUMERIC_ORDER
    X_num = np.array([[X_num_dict[k] for k in NUMERIC_ORDER]], dtype=float)
    X_num_scaled = scaler.transform(X_num)

    # --- Categorical block with keepers mapping ---
    # Missing optional categoricals are set to "Unknown" then mapped through keepers to "Other" if needed
    def get_cat(name, default="Unknown"):
        v = payload.get(name, default)
        v = "Unknown" if v is None else str(v)
        return map_keepers(v, name, keepers)

    cat_vals = [[
        get_cat("BIKE_TYPE"),
        get_cat("BIKE_COLOUR"),
        get_cat("DIVISION"),
        get_cat("LOCATION_TYPE"),
        get_cat("PREMISES_TYPE"),
        get_cat("PRIMARY_OFFENCE"),
    ]]
    X_cat = encoder.transform(cat_vals)

    # Concatenate to match training order: [scaled numeric | one-hot categorical]
    return np.hstack([X_num_scaled, X_cat]), X_num_dict, {c: cat_vals[0][i] for i, c in enumerate(CAT_ORDER)}

def predict_one(model, X_row: np.ndarray) -> Tuple[int, Optional[float]]:
    y_pred = int(model.predict(X_row)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X_row)[0, 1])
        except Exception:
            proba = None
    return y_pred, proba

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# Load everything at startup
ENCODER, SCALER, KEEPERS = load_artifacts(MODELS_DIR)
MODEL_PATH = choose_model_file(MODELS_DIR, REPORTS_CSV, MODEL_FILE)
MODEL = load_model(MODEL_PATH)
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

@app.route("/health", methods=["GET"])
def health():
    payload = {
        "status": "ok",
        "model": MODEL_NAME,
        "encoder_loaded": True,
        "scaler_loaded": True,
        "keepers_loaded": True,
        "numeric_order": NUMERIC_ORDER,
        "cat_order": CAT_ORDER
    }
    return cors(jsonify(payload))

@app.route("/", methods=["GET"])
def form():
    # UI now lives in src/templates/index.html + src/static/ (Flask defaults).
    return cors(make_response(render_template("index.html"), 200))

@app.route("/meta", methods=["GET"])
def meta():
    # Powers the dropdowns: real model categories with human-readable labels.
    return cors(jsonify(build_meta()))

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return cors(make_response("", 204))

    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")

        X_row, num_feats, cat_feats = build_feature_row(payload, ENCODER, SCALER, KEEPERS)
        y_pred, proba = predict_one(MODEL, X_row)

        # Echo normalized feature values we actually used (pre-scaling)
        occ_month, occ_hour, occ_dow = parse_occ_fields(payload)
        response = {
            "prediction": y_pred,
            "proba": proba,
            "model_name": MODEL_NAME,
            "features": {
                "occ_month": occ_month,
                "occ_hour":  occ_hour,
                "occ_dow":   occ_dow,
                "report_delay_hours": float(num_feats["report_delay_hours"]),
                "BIKE_COST": float(payload["BIKE_COST"]),
                **cat_feats
            }
        }
        return cors(jsonify(response))
    except Exception as e:
        msg = {"error": str(e)}
        return cors(make_response(jsonify(msg), 400))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
