"""
modules/anomaly.py
──────────────────
Anomaly detection engine — two complementary methods:

  1. Isolation Forest (sklearn)
     ML-based: builds random trees and measures how quickly a point
     gets isolated. Anomalies isolate faster → lower score.
     Good at: multi-dimensional anomalies (unusual combinations of values)

  2. IQR (Inter-Quartile Range) per column
     Statistical: flags values outside [Q1 - k*IQR, Q3 + k*IQR].
     Good at: per-column extreme outliers with a clear human-readable threshold.

Both methods run on numeric columns only.
Returns a clean result dict that the Dashboard page renders.
Pure Python — no Streamlit imports here (keeps it testable and reusable).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    iqr_multiplier: float = 2.5,
    random_state: int = 42,
) -> dict:
    """
    Run both Isolation Forest and IQR anomaly detection on *df*.

    Parameters
    ----------
    df              : Input DataFrame (any mix of column types).
    contamination   : Fraction of rows expected to be anomalous (Isolation Forest).
    iqr_multiplier  : How many IQR units past Q1/Q3 to flag (IQR method).
    random_state    : For reproducible Isolation Forest results.

    Returns
    -------
    dict with keys:
        anomaly_rows    : DataFrame subset — rows flagged by Isolation Forest
        normal_rows     : DataFrame subset — rows not flagged
        if_scores       : Series — raw Isolation Forest anomaly scores (lower = more anomalous)
        if_labels       : Series — 1 (normal) or -1 (anomaly) per row
        iqr_by_column   : dict[col_name -> {count, upper, lower, max_outlier, min_outlier}]
        numeric_columns : list of numeric column names used
        contamination   : the contamination value used
        iqr_multiplier  : the IQR multiplier used
    """
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Drop columns that are entirely NaN
    num_df = num_df.dropna(axis=1, how="all")

    # Fill remaining NaNs with median (IF can't handle NaN)
    num_df = num_df.fillna(num_df.median())

    if num_df.empty or num_df.shape[1] == 0:
        return _empty_result(df)

    # ── 1. Isolation Forest ───────────────────────────────────────────
    # RobustScaler: scale using median + IQR so outliers don't distort scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(num_df)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,       # more trees = more stable results
        max_samples="auto",
    )
    if_labels = iso.fit_predict(X_scaled)          # 1=normal, -1=anomaly
    if_scores = iso.score_samples(X_scaled)        # lower = more anomalous

    labels_series = pd.Series(if_labels, index=df.index, name="if_label")
    scores_series = pd.Series(if_scores, index=df.index, name="if_score")

    anomaly_mask = labels_series == -1
    anomaly_rows = df[anomaly_mask].copy()
    normal_rows = df[~anomaly_mask].copy()

    # Attach scores to anomaly rows for sorting
    anomaly_rows["_if_score"] = scores_series[anomaly_mask]
    anomaly_rows = anomaly_rows.sort_values(
        "_if_score")    # most anomalous first
    anomaly_rows = anomaly_rows.drop(columns=["_if_score"])

    # ── 2. IQR per column ─────────────────────────────────────────────
    iqr_by_column: dict = {}
    for col in num_df.columns:
        q1 = num_df[col].quantile(0.25)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue        # constant column — skip
        upper = q3 + iqr_multiplier * iqr
        lower = q1 - iqr_multiplier * iqr
        outlier_mask = (df[col] > upper) | (df[col] < lower)
        outliers = df.loc[outlier_mask, col]
        if outliers.empty:
            continue
        iqr_by_column[col] = {
            "count":       int(outliers.count()),
            "upper":       float(upper),
            "lower":       float(lower),
            "max_outlier": float(outliers.max()),
            "min_outlier": float(outliers.min()),
            "q1":          float(q1),
            "q3":          float(q3),
            "iqr":         float(iqr),
        }

    return {
        "anomaly_rows":    anomaly_rows,
        "normal_rows":     normal_rows,
        "if_scores":       scores_series,
        "if_labels":       labels_series,
        "iqr_by_column":   iqr_by_column,
        "numeric_columns": list(num_df.columns),
        "contamination":   contamination,
        "iqr_multiplier":  iqr_multiplier,
    }


def anomaly_summary_stats(result: dict) -> dict:
    """
    Compute KPI numbers from a detect_anomalies() result dict.
    Safe to call even if result has empty DataFrames.
    """
    anomaly_rows = result.get("anomaly_rows", pd.DataFrame())
    normal_rows = result.get("normal_rows",  pd.DataFrame())
    iqr_cols = result.get("iqr_by_column", {})

    total = len(anomaly_rows) + len(normal_rows)
    n_anom = len(anomaly_rows)
    pct = round(n_anom / total * 100, 1) if total else 0.0

    # IQR total across all columns (may count same row in multiple columns)
    iqr_total = sum(info["count"] for info in iqr_cols.values())

    return {
        "total_rows":       total,
        "total_anomalies":  n_anom,
        "anomaly_pct":      pct,
        "normal_count":     len(normal_rows),
        "columns_affected": len(iqr_cols),
        "if_count":         n_anom,
        "iqr_count":        iqr_total,
    }


# ────────────────────────────────────────────────────────────────────────────
# INTERNAL
# ────────────────────────────────────────────────────────────────────────────

def _empty_result(df: pd.DataFrame) -> dict:
    """Return a well-structured empty result when no numeric columns exist."""
    return {
        "anomaly_rows":    pd.DataFrame(),
        "normal_rows":     df.copy(),
        "if_scores":       pd.Series(dtype=float),
        "if_labels":       pd.Series(dtype=int),
        "iqr_by_column":   {},
        "numeric_columns": [],
        "contamination":   0.05,
        "iqr_multiplier":  2.5,
    }
