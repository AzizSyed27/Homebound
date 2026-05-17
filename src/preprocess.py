# preprocess.py
# Simple, reproducible preprocessing with saved encoder/scaler and category keep-lists.

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- helpers ----------

def _read_csv_auto(path: str) -> pd.DataFrame:
    # Try comma; if it fails to parse dates later we still proceed.
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce to datetime; rows with missing critical dates will be dropped later.
    for col in ("OCC_DATE", "REPORT_DATE"):
        if col in df.columns:
            # `infer_datetime_format` was deprecated in pandas 2.0 and removed in
            # later versions; modern pandas infers the format automatically.
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. Returns a new DataFrame with only the columns we need."""
    if "STATUS" not in df.columns:
        raise ValueError("Missing STATUS column in dataset.")

    # Target
    y = df["STATUS"].astype(str).str.upper().isin(["RECOVERED", "RETURNED"]).astype(int)

    # Dates
    if "OCC_DATE" in df.columns and "REPORT_DATE" in df.columns:
        occ = df["OCC_DATE"]
        rep = df["REPORT_DATE"]
    else:
        # Fallback to provided integer columns if present; else fail
        if not {"OCC_YEAR", "OCC_MONTH", "OCC_HOUR", "OCC_DOW"}.issubset(df.columns):
            raise ValueError("Dataset must contain OCC_DATE/REPORT_DATE or OCC_* fields.")
        # Build a pseudo datetime for features only (set day=1, minute=0)
        occ = pd.to_datetime(
            dict(year=df["OCC_YEAR"], month=df["OCC_MONTH"], day=1, hour=df["OCC_HOUR"], minute=0),
            errors="coerce",
        )
        rep = None

    # Temporal features
    occ_month = occ.dt.month
    occ_hour  = occ.dt.hour
    occ_dow   = occ.dt.dayofweek  # Mon=0..Sun=6

    # Cyclical encodings
    occ_month_sin = np.sin(2 * np.pi * (occ_month.fillna(0) - 1) / 12.0)
    occ_month_cos = np.cos(2 * np.pi * (occ_month.fillna(0) - 1) / 12.0)
    occ_hour_sin  = np.sin(2 * np.pi * occ_hour.fillna(0) / 24.0)
    occ_hour_cos  = np.cos(2 * np.pi * occ_hour.fillna(0) / 24.0)

    # Weekend
    is_weekend = (occ_dow >= 5).astype(int)

    # Report delay (hours)
    if rep is not None:
        delay_hours = (rep - occ).dt.total_seconds() / 3600.0
    else:
        delay_hours = pd.Series(np.nan, index=df.index)
    # Clip unreasonably large/small values and impute later
    delay_hours = delay_hours.clip(lower=0, upper=24 * 30)

    # Cost (use log1p to stabilize)
    cost = pd.to_numeric(df.get("BIKE_COST", pd.Series(np.nan, index=df.index)), errors="coerce")
    cost = cost.clip(lower=0, upper=20000)
    cost_log = np.log1p(cost)

    out = pd.DataFrame({
        "occ_month": occ_month,
        "occ_hour":  occ_hour,
        "occ_dow":   occ_dow,
        "occ_month_sin": occ_month_sin,
        "occ_month_cos": occ_month_cos,
        "occ_hour_sin":  occ_hour_sin,
        "occ_hour_cos":  occ_hour_cos,
        "is_weekend": is_weekend,
        "report_delay_hours": delay_hours,
        "bike_cost_log": cost_log,
        "BIKE_TYPE": df.get("BIKE_TYPE"),
        "BIKE_COLOUR": df.get("BIKE_COLOUR"),
        "DIVISION": df.get("DIVISION"),
        "LOCATION_TYPE": df.get("LOCATION_TYPE"),
        "PREMISES_TYPE": df.get("PREMISES_TYPE"),
        "PRIMARY_OFFENCE": df.get("PRIMARY_OFFENCE"),
        "target": y,
    })
    return out


def _select_and_impute(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Keep rows with target defined
    df_feat = df_feat[df_feat["target"].notna()].copy()

    # Numeric / categorical split
    num_cols = [
        "occ_month", "occ_hour", "occ_dow",
        "occ_month_sin", "occ_month_cos", "occ_hour_sin", "occ_hour_cos",
        "is_weekend", "report_delay_hours", "bike_cost_log"
    ]
    cat_cols = [
        "BIKE_TYPE", "BIKE_COLOUR", "DIVISION",
        "LOCATION_TYPE", "PREMISES_TYPE", "PRIMARY_OFFENCE"
    ]

    # Impute numeric with median
    for c in num_cols:
        if c not in df_feat.columns:
            raise ValueError(f"Missing expected numeric feature: {c}")
        med = df_feat[c].median()
        df_feat[c] = df_feat[c].fillna(med)

    # Coerce categoricals to string with NaN -> "Unknown"
    for c in cat_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
        else:
            df_feat[c] = "Unknown"

    X_num = df_feat[num_cols].copy()
    X_cat = df_feat[cat_cols].copy()
    y = df_feat["target"].astype(int)
    return (X_num, X_cat, y)


def _keep_topk_and_other(X_cat: pd.DataFrame, keepers: Dict[str, List[str]]) -> pd.DataFrame:
    Xc = X_cat.copy()
    for c, keep in keepers.items():
        Xc[c] = Xc[c].where(Xc[c].isin(keep), other="Other")
    return Xc


def _make_keepers_from_train(X_cat_train: pd.DataFrame, top_k: int = 12) -> Dict[str, List[str]]:
    keepers = {}
    for c in X_cat_train.columns:
        vals = X_cat_train[c].value_counts(dropna=False)
        keep = list(vals.head(top_k).index.astype(str))
        if "Other" not in keep:
            keep.append("Other")
        keepers[c] = keep
    return keepers


def _onehot_encoder():
    # Compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------- public API ----------

def prepare_train_test(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    top_k: int = 12
):
    """
    Returns:
      X_train, X_test, y_train, y_test, encoder, scaler, keepers, feature_names
    """
    raw = _read_csv_auto(path)
    raw = _parse_dates(raw)
    feat = _build_basic_features(raw)
    X_num, X_cat, y = _select_and_impute(feat)

    # Split BEFORE fitting scaler/encoder (no leakage)
    Xn_tr, Xn_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        X_num, X_cat, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Map rare categories to "Other" using train frequencies
    keepers = _make_keepers_from_train(Xc_tr, top_k=top_k)
    Xc_tr = _keep_topk_and_other(Xc_tr, keepers)
    Xc_te = _keep_topk_and_other(Xc_te, keepers)

    # Fit encoder on train cats; transform train/test
    enc = _onehot_encoder()
    Xc_tr_ohe = enc.fit_transform(Xc_tr)
    Xc_te_ohe = enc.transform(Xc_te)

    # Fit scaler on train nums; transform train/test
    scaler = StandardScaler()
    Xn_tr_sc = scaler.fit_transform(Xn_tr.values)
    Xn_te_sc = scaler.transform(Xn_te.values)

    # Concatenate
    X_train = np.hstack([Xn_tr_sc, Xc_tr_ohe])
    X_test  = np.hstack([Xn_te_sc, Xc_te_ohe])

    # Construct feature names
    num_names = list(Xn_tr.columns)
    cat_names = list(enc.get_feature_names_out(Xc_tr.columns))
    feature_names = num_names + cat_names

    return X_train, X_test, y_tr.values, y_te.values, enc, scaler, keepers, feature_names


def transform_for_eval(
    path: str,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    keepers: Dict[str, List[str]],
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Rebuild features from raw CSV, split with same random_state, and transform the test fold
    using the provided encoder/scaler/keepers (fit on the training fold earlier).
    Returns: X_test, y_test
    """
    raw = _read_csv_auto(path)
    raw = _parse_dates(raw)
    feat = _build_basic_features(raw)
    X_num, X_cat, y = _select_and_impute(feat)

    # Split the SAME way
    Xn_tr, Xn_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(
        X_num, X_cat, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Map using saved keepers
    Xc_te = _keep_topk_and_other(Xc_te, keepers)

    # Transform with saved artifacts
    Xc_te_ohe = encoder.transform(Xc_te)
    Xn_te_sc  = scaler.transform(Xn_te.values)
    X_test = np.hstack([Xn_te_sc, Xc_te_ohe])
    return X_test, y_te.values


def save_artifacts(models_dir: str, encoder, scaler, keepers: Dict[str, List[str]], feature_names: List[str]):
    os.makedirs(models_dir, exist_ok=True)
    import joblib
    joblib.dump(encoder, os.path.join(models_dir, "encoder.joblib"))
    joblib.dump(scaler,  os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(keepers, os.path.join(models_dir, "cat_keepers.joblib"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
