"""
modules/evidence.py
───────────────────
Pre-compute all statistical facts about a DataFrame BEFORE the LLM
sees any data. The LLM receives only these pre-calculated numbers —
it cannot hallucinate statistics it was never given.

BUGS FIXED vs original app.py
──────────────────────────────
BUG (moderate) ✅  compute_evidence() called st.session_state.get("dtype_fixes")
                   directly — mixing Streamlit UI state into a pure maths
                   function broke testability and separation of concerns.
                   dtype_fixes is now passed as an explicit parameter.

IMPROVEMENT ✅     Function decorated with @st.cache_data so it only
                   re-runs when the DataFrame actually changes, not on
                   every chat message / rerender.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st


# Keywords used to auto-detect a "target" column for correlation analysis
_TARGET_KEYWORDS = [
    "price", "salary", "target", "value", "revenue", "cost", "output",
    "score", "amount", "sales", "income", "total", "result", "profit",
    "loss", "rate", "return", "yield", "demand", "spend", "wage",
    "gross", "net", "fee", "charge", "payment", "purchase",
]


@st.cache_data(show_spinner=False)
def compute_evidence(df: pd.DataFrame, dtype_fixes: tuple[str, ...] = ()) -> dict:
    """
    Compute all statistical facts about *df*.

    Parameters
    ----------
    df          : The DataFrame to analyse.
    dtype_fixes : Columns that were auto-converted on file load
                  (passed explicitly — NOT read from session state).

    Returns a plain dict that is JSON-serialisable and safe to embed
    directly into an LLM prompt.

    NOTE: dtype_fixes is a tuple (not list) so st.cache_data can hash it.
    """
    num_df = df.select_dtypes(include=[np.number])

    # ── 1. Shape & basics ─────────────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    null_series = df.isnull().sum()
    null_dict = {c: int(v) for c, v in null_series.items() if v > 0}
    missing_pct = round(null_series.sum() / total_cells *
                        100, 2) if total_cells else 0.0
    dup_count = int(df.duplicated().sum())

    # ── 2. Descriptive statistics ─────────────────────────────────────
    desc: dict = {}
    if not num_df.empty:
        desc = num_df.describe().round(4).to_dict()

    # ── 3. Smart target column detection ─────────────────────────────
    target_col: str | None = None
    if not num_df.empty:
        for kw in _TARGET_KEYWORDS:
            hits = [c for c in num_df.columns if kw in c.lower()]
            if hits:
                target_col = hits[0]
                break

    # ── 4. Correlations with target ───────────────────────────────────
    corr_with_target: dict = {}
    if target_col and target_col in num_df.columns:
        cs = (
            num_df.corr()[target_col]
            .drop(target_col)
            .sort_values(ascending=False, key=abs)
            .round(3)
        )
        corr_with_target = {c: float(v)
                            for c, v in cs.items() if not pd.isna(v)}

    # ── 5. Categorical distributions ──────────────────────────────────
    cat_distributions: dict = {}
    for col in df.select_dtypes(include=["object", "category"]).columns[:7]:
        vc = df[col].value_counts(normalize=True).head(5).round(4)
        cat_distributions[col] = {
            str(k): f"{v*100:.1f}%" for k, v in vc.items()}

    # ── 6. Extreme outliers (IQR × 3) ────────────────────────────────
    extreme_outliers: list[dict] = []
    for col in num_df.columns:
        q1 = num_df[col].quantile(0.25)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper = q3 + 3 * iqr
        lower = q1 - 3 * iqr
        ext = num_df[(num_df[col] > upper) | (num_df[col] < lower)]
        if not ext.empty:
            extreme_outliers.append({
                "column":          col,
                "outlier_count":   int(len(ext)),
                "max_value":       round(float(ext[col].max()), 4),
                "min_value":       round(float(ext[col].min()), 4),
                "upper_threshold": round(float(upper), 4),
                "lower_threshold": round(float(lower), 4),
            })

    # ── 7. Data types summary ─────────────────────────────────────────
    dtype_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── 8. Suspicious string columns (partially numeric) ─────────────
    suspicious_string_cols: list[dict] = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).str.strip().head(100)
        if sample.empty:
            continue
        cleaned = sample.str.replace(r"[$€£¥₹,%,]", "", regex=True)
        trial = pd.to_numeric(cleaned, errors="coerce")
        rate = trial.notna().sum() / max(len(trial), 1)
        if 0.40 <= rate < 0.80:
            suspicious_string_cols.append({
                "column":       col,
                "sample_values": df[col].dropna().astype(str).head(3).tolist(),
                "note":         f"{rate*100:.0f}% of values look numeric but stored as text",
            })

    return {
        "shape":                    {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names":             list(df.columns),
        "dtypes":                   dtype_summary,
        "missing_values":           null_dict,
        "missing_pct":              missing_pct,
        "duplicate_rows":           dup_count,
        "descriptive_stats":        desc,
        "target_column":            target_col,
        "correlations_by_target":   corr_with_target,
        "categorical_distributions": cat_distributions,
        "extreme_outliers":         extreme_outliers,
        "suspicious_string_columns": suspicious_string_cols,
        # ← explicit param, not session_state
        "auto_converted_columns":   list(dtype_fixes),
    }
