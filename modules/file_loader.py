"""
modules/file_loader.py
──────────────────────
File ingestion: CSV / XLSX / PDF loading + dtype auto-correction.

BUGS FIXED vs original app.py
──────────────────────────────
BUG 4 ✅  fix_dtypes() used regex r"[$€£¥₹, s]" which stripped ALL
          whitespace from every string column. This silently mangled
          text columns — e.g. "New York" → "NewYork", "Ice Cream" →
          "IceCream". The fix: only apply numeric cleaning to columns
          we've already determined are numeric candidates, and use a
          dedicated cleaning helper that never touches pure-text columns.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st


# ── Internal: clean a single Series for numeric conversion ────────────

def _clean_for_numeric(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols, commas, percent signs, and surrounding
    whitespace from a string Series — WITHOUT touching internal spaces
    in words (that was the original bug).

    Only call this on columns already identified as numeric candidates.
    """
    return (
        series.astype(str)
        .str.strip()                                   # leading/trailing space only
        .str.replace(r"^[$€£¥₹]", "", regex=True)   # leading currency symbol
        .str.replace(r",", "", regex=False)            # thousands separators
        .str.replace(r"%$", "", regex=True)            # trailing percent
        .str.replace(r"^\((.+)\)$", r"-\1", regex=True)  # (123) → -123
    )


# ── Public: fix column dtypes after load ─────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Auto-convert columns that look numeric but were loaded as strings.
    Returns (fixed_df, list_of_converted_column_names).

    FIX vs original: numeric-cleaning regex is ONLY applied to columns
    that pass the ≥80% numeric sample test — never to pure-text columns.
    Internal whitespace ("New York") is now preserved.
    """
    fixed_cols: list[str] = []
    df = df.copy()

    for col in df.columns:
        if df[col].dtype != object:
            continue

        sample = df[col].dropna().astype(str).str.strip()
        if sample.empty:
            continue

        # Test on a cleaned sample (≤200 rows for speed)
        test_sample = _clean_for_numeric(sample.head(200))
        converted = pd.to_numeric(test_sample, errors="coerce")
        success_rate = converted.notna().sum() / max(len(converted), 1)

        if success_rate >= 0.80:
            # Apply cleaning only to THIS column (confirmed numeric candidate)
            full_cleaned = _clean_for_numeric(df[col].dropna().rename(col))
            # Reindex to match full df length (NaN rows stay NaN)
            full_series = _clean_for_numeric(
                df[col].astype(str).where(df[col].notna()))
            df[col] = pd.to_numeric(full_series, errors="coerce")
            fixed_cols.append(col)

    return df, fixed_cols


# ── Public: load uploaded file ────────────────────────────────────────

def load_uploaded_file(f) -> tuple[pd.DataFrame | None, str, list[str]]:
    """
    Load CSV / XLSX / PDF.
    Returns (dataframe_or_None, pdf_text_or_empty, dtype_fixes_list).
    """
    name = f.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f)
            df, fixed = fix_dtypes(df)
            _notify_fixes(fixed)
            return df, "", fixed

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f, engine="openpyxl")
            df, fixed = fix_dtypes(df)
            _notify_fixes(fixed)
            return df, "", fixed

        elif name.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(f)
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()
            if not text:
                st.sidebar.warning(
                    "⚠️ Scanned PDF — no selectable text found. "
                    "Text extraction not possible."
                )
                return None, "", []
            total_chars = len(text)
            if total_chars > 15_000:
                text = text[:15_000]
                st.sidebar.info(
                    f"📄 Large PDF ({total_chars:,} chars). "
                    "Using first 15,000 characters."
                )
            return None, text, []

        else:
            st.error(f"Unsupported file type: `{name}`")
            return None, "", []

    except Exception as exc:
        st.error(f"❌ File load error: {exc}")
        return None, "", []


def _notify_fixes(fixed: list[str]) -> None:
    if fixed:
        st.sidebar.info(
            f"🔧 Auto-fixed {len(fixed)} column(s) from text → numeric: "
            + ", ".join(f"`{c}`" for c in fixed)
        )
