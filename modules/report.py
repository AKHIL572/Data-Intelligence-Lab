"""
modules/report.py
─────────────────
Generates a professional, downloadable PDF analyst report from:
  - Dataset overview stats
  - Anomaly detection results (Isolation Forest + IQR)
  - AI-generated business explanations (if available)

Uses fpdf2 (pip install fpdf2) -- zero external calls, runs 100% offline.

Public API
──────────
  generate_report(df, result, stats, explanations, filename) -> bytes
      Returns the PDF as raw bytes → pass directly to st.download_button().
"""

from __future__ import annotations
from datetime import datetime
import io
import pandas as pd
import numpy as np
from fpdf import FPDF, XPos, YPos


# ── Colour palette (matches app dark theme for brand consistency) ──────
C_BLACK = (7,   11,  16)    # #070b10 -- page background (not used directly)
C_NAVY = (11,  15,  22)    # #0b0f16
C_HEADER_BG = (13,  46,  82)    # #0d2e52 -- header bar
C_ACCENT = (51,  204, 255)   # #33ccff -- cyan accent
C_WHITE = (255, 255, 255)
C_LIGHT_GREY = (200, 214, 229)   # #c8d6e5
C_MID_GREY = (61,  90,  115)   # #3d5a73
C_RED = (255, 107, 107)   # #ff6b6b -- anomaly highlight
C_GREEN = (127, 255, 178)   # #7fffb2
C_ORANGE = (255, 179, 71)    # #ffb347


# ────────────────────────────────────────────────────────────────────────────
# INTERNAL PDF CLASS
# ────────────────────────────────────────────────────────────────────────────

class _AnalystReport(FPDF):
    """
    Custom FPDF subclass with branded header, footer, and helper methods.
    """

    def __init__(self, dataset_name: str = "Dataset"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.dataset_name = dataset_name
        self.generated_at = datetime.now().strftime("%Y-%m-%d  %H:%M")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(left=18, top=18, right=18)

    # ── fpdf2 overrides ──────────────────────────────────────────────

    def header(self):
        # Dark header bar
        self.set_fill_color(*C_HEADER_BG)
        self.rect(0, 0, 210, 14, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_ACCENT)
        self.set_xy(8, 3)
        self.cell(0, 8, "[LAB]  PRIVATE DATA INTELLIGENCE LAB  ·  ANOMALY REPORT",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_MID_GREY)
        self.cell(0, 8,
                  f"Generated {self.generated_at}  ·  {self.dataset_name}  ·  "
                  f"Page {self.page_no()} / {{nb}}",
                  align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Helpers ──────────────────────────────────────────────────────

    def section_title(self, text: str) -> None:
        """Dark background section title bar."""
        self.ln(4)
        self.set_fill_color(*C_NAVY)
        self.set_draw_color(*C_ACCENT)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 9, f"  {text}", border="L", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(*C_LIGHT_GREY)

    def body_text(self, text: str, size: int = 9) -> None:
        self.set_font("Helvetica", "", size)
        self.set_text_color(*C_LIGHT_GREY)
        # fpdf2 multi_cell handles long lines & newlines safely
        safe = _safe_str(text)
        self.multi_cell(0, 5.5, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def kpi_row(self, items: list[tuple[str, str]]) -> None:
        """Render a row of KPI boxes -- (label, value) pairs."""
        n = len(items)
        width = (self.w - self.l_margin - self.r_margin - (n - 1) * 3) / n
        x0 = self.l_margin

        for i, (label, value) in enumerate(items):
            x = x0 + i * (width + 3)
            y = self.get_y()

            # Box
            self.set_fill_color(*C_NAVY)
            self.set_draw_color(*C_HEADER_BG)
            self.rect(x, y, width, 16, style="FD")

            # Label
            self.set_xy(x + 2, y + 2)
            self.set_font("Helvetica", "", 6)
            self.set_text_color(*C_MID_GREY)
            self.cell(width - 4, 4, label.upper(),
                      new_x=XPos.RIGHT, new_y=YPos.TOP)

            # Value
            self.set_xy(x + 2, y + 7)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*C_ACCENT)
            self.cell(width - 4, 7, str(value),
                      new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.ln(20)

    def anomaly_block(self, col: str, info: dict,
                      explanation: str | None = None) -> None:
        """One anomaly column block with stats + optional AI explanation."""
        y0 = self.get_y()

        # Left accent bar
        self.set_fill_color(*C_RED)
        self.rect(self.l_margin, y0, 2,
                  28 if not explanation else 42, style="F")

        # Column name
        self.set_xy(self.l_margin + 5, y0 + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_RED)
        self.cell(0, 5, col.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Stats inline
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_LIGHT_GREY)
        stats_line = (
            f"Outliers: {info['count']}  |  "
            f"Upper threshold: {info['upper']:.2f}  |  "
            f"Lower threshold: {info['lower']:.2f}  |  "
            f"Max outlier: {info['max_outlier']:.2f}  |  "
            f"Min outlier: {info['min_outlier']:.2f}"
        )
        self.multi_cell(0, 5, _safe_str(stats_line),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # AI explanation (if available)
        if explanation:
            self.set_x(self.l_margin + 5)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*C_MID_GREY)
            # Strip markdown formatting for clean PDF text
            clean_exp = _strip_markdown(explanation)
            self.multi_cell(0, 5, _safe_str(clean_exp),
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(4)

    def simple_table(self, headers: list[str], rows: list[list],
                     col_widths: list[float] | None = None) -> None:
        """Render a bordered table."""
        usable = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)

        # Header row
        self.set_fill_color(*C_HEADER_BG)
        self.set_text_color(*C_ACCENT)
        self.set_font("Helvetica", "B", 7.5)
        self.set_draw_color(*C_NAVY)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True,
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln()

        # Data rows (alternating shades)
        self.set_font("Helvetica", "", 7.5)
        for ri, row in enumerate(rows):
            self.set_fill_color(*(C_NAVY if ri % 2 == 0 else C_BLACK))
            self.set_text_color(*C_LIGHT_GREY)
            for i, cell_val in enumerate(row):
                self.cell(col_widths[i], 6, _safe_str(str(cell_val)),
                          border=1, fill=True,
                          new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.ln()
        self.ln(3)


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ────────────────────────────────────────────────────────────────────────────

def generate_report(
    df: pd.DataFrame,
    result: dict,
    stats: dict,
    explanations: dict[str, str] | None = None,
    filename: str = "anomaly_report",
) -> bytes:
    """
    Build a full PDF analyst report and return it as raw bytes.

    Parameters
    ----------
    df            : The original DataFrame that was analysed.
    result        : Output of modules.anomaly.detect_anomalies()
    stats         : Output of modules.anomaly.anomaly_summary_stats()
    explanations  : dict mapping column_name → AI explanation string (optional)
    filename      : Used only in the footer dataset label.

    Returns
    -------
    bytes  -- PDF file content, ready for st.download_button()
    """
    explanations = explanations or {}
    pdf = _AnalystReport(dataset_name=filename)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_page_background((7, 11, 16))  # #070b10

    # ── Cover / Title ─────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(0, 12, "Anomaly Detection Report", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*C_MID_GREY)
    pdf.cell(0, 6, f"Dataset: {filename}   ·   Generated: {pdf.generated_at}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # Cyan divider line
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    # ── Section 1: Executive Summary KPIs ─────────────────────────────
    pdf.section_title("1.  Executive Summary")
    pdf.kpi_row([
        ("Total Rows",         f"{stats['total_rows']:,}"),
        ("Anomalies Found",    f"{stats['total_anomalies']:,}"),
        ("Anomaly Rate",       f"{stats['anomaly_pct']:.1f}%"),
        ("Columns Affected",   str(stats['columns_affected'])),
    ])

    pdf.body_text(
        f"This report summarises the results of an automated anomaly detection "
        f"analysis performed on the dataset '{filename}'. The analysis used two "
        f"complementary methods: Isolation Forest (a machine-learning approach that "
        f"detects multi-dimensional anomalies) and IQR (a per-column statistical "
        f"method that flags values outside {result.get('iqr_multiplier', 2.5)} × "
        f"the inter-quartile range). "
        f"A total of {stats['total_anomalies']} rows ({stats['anomaly_pct']:.1f}% "
        f"of the data) were flagged as anomalous, spanning "
        f"{stats['columns_affected']} column(s)."
    )

    # ── Section 2: Dataset Overview ───────────────────────────────────
    pdf.section_title("2.  Dataset Overview")

    num_df = df.select_dtypes(include="number")
    tc = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    quality = round(100 - missing / tc * 100, 1) if tc else 0.0
    dupes = int(df.duplicated().sum())

    pdf.kpi_row([
        ("Rows",            f"{df.shape[0]:,}"),
        ("Columns",         str(df.shape[1])),
        ("Numeric cols",    str(len(num_df.columns))),
        ("Data quality",    f"{quality}%"),
        ("Missing values",  f"{missing:,}"),
        ("Duplicate rows",  str(dupes)),
    ])

    # Column dtype table
    dtype_rows = [[col, str(df[col].dtype),
                   str(df[col].dropna().iloc[0]) if df[col].notna().any() else "--"]
                  for col in df.columns[:20]]   # cap at 20 rows
    pdf.simple_table(
        headers=["Column", "Type", "Sample Value"],
        rows=dtype_rows,
        col_widths=[80, 30, 64],
    )
    if len(df.columns) > 20:
        pdf.body_text(f"… and {len(df.columns) - 20} more columns not shown.")

    # ── Section 3: Descriptive Statistics ─────────────────────────────
    pdf.section_title("3.  Descriptive Statistics (Numeric Columns)")
    if num_df.empty:
        pdf.body_text("No numeric columns found.")
    else:
        desc = num_df.describe().round(2)
        stat_headers = ["Column"] + list(desc.index)
        stat_rows = []
        for col in desc.columns:
            row = [col] + [str(v) for v in desc[col].values]
            stat_rows.append(row)

        # Dynamic column widths
        n_stats = len(stat_headers)
        usable = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = [38] + [(usable - 38) / (n_stats - 1)] * (n_stats - 1)
        pdf.simple_table(stat_headers, stat_rows, col_widths=col_w)

    # ── Section 4: Anomaly Detection Results ──────────────────────────
    pdf.section_title("4.  Anomaly Detection Results")

    pdf.body_text(
        f"Isolation Forest flagged {stats['if_count']} rows as anomalous "
        f"(contamination parameter: {result.get('contamination', 0.05):.0%}). "
        f"IQR analysis identified {stats['iqr_count']} per-column outliers across "
        f"{stats['columns_affected']} column(s)."
    )

    iqr_cols = result.get("iqr_by_column", {})

    if not iqr_cols:
        pdf.body_text(
            "No IQR outliers detected with the current sensitivity settings.")
    else:
        # Summary IQR table
        iqr_rows = []
        for col, info in iqr_cols.items():
            iqr_rows.append([
                col,
                str(info["count"]),
                f"{info['upper']:.2f}",
                f"{info['lower']:.2f}",
                f"{info['max_outlier']:.2f}",
                f"{info['min_outlier']:.2f}",
            ])
        pdf.simple_table(
            headers=["Column", "Outliers", "Upper Threshold",
                     "Lower Threshold", "Max Outlier", "Min Outlier"],
            rows=iqr_rows,
            col_widths=[45, 20, 32, 32, 27, 27],
        )

    # ── Section 5: Column-Level Anomaly Details ────────────────────────
    if iqr_cols:
        pdf.section_title("5.  Column-Level Anomaly Details & AI Explanations")

        for col, info in iqr_cols.items():
            # Page break if not enough room
            if pdf.get_y() > pdf.h - 60:
                pdf.add_page()

            exp = explanations.get(col)
            pdf.anomaly_block(col, info, explanation=exp)

    # ── Section 6: Top Anomalous Rows ─────────────────────────────────
    pdf.section_title("6.  Top Anomalous Rows (Isolation Forest)")
    anomaly_rows = result.get("anomaly_rows", pd.DataFrame())
    if anomaly_rows.empty:
        pdf.body_text("No anomalous rows detected.")
    else:
        # Show first 15 rows, numeric cols only (to fit page)
        show_df = anomaly_rows.head(15)
        num_cols = [
            c for c in show_df.columns if show_df[c].dtype != object][:8]
        show_df = show_df[num_cols].round(2)

        usable = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = [usable / len(show_df.columns)] * len(show_df.columns)
        table_rows = [list(row) for _, row in show_df.iterrows()]
        pdf.simple_table(list(show_df.columns), table_rows, col_widths=col_w)

        if len(anomaly_rows) > 15:
            pdf.body_text(
                f"… and {len(anomaly_rows) - 15} more anomalous rows not shown.")

    # ── Section 7: Recommendations ────────────────────────────────────
    pdf.section_title("7.  Recommended Next Steps")
    recs = _build_recommendations(stats, iqr_cols, df)
    for i, rec in enumerate(recs, 1):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(8, 6, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*C_LIGHT_GREY)
        pdf.multi_cell(0, 5.5, _safe_str(
            rec), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    # ── Final divider + disclaimer ─────────────────────────────────────
    pdf.ln(4)
    pdf.set_draw_color(*C_HEADER_BG)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*C_MID_GREY)
    pdf.multi_cell(
        0, 4.5,
        "This report was generated automatically by Private Data Intelligence Lab. "
        "All anomaly detection was performed locally -- no data was sent to external servers. "
        "AI-generated explanations (where present) are based solely on statistical evidence "
        "computed from the dataset and should be reviewed by a qualified analyst before action.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Output as bytes
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ────────────────────────────────────────────────────────────────────────────

def _safe_str(text: str) -> str:
    """Replace characters fpdf2 can't encode in latin-1 with ASCII equivalents."""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2022": "*", "\u2026": "...",
        "\u00a0": " ", "\u20ac": "EUR", "\u00b0": "deg",
        "🔬": "[LAB]", "📊": "[CHART]", "📄": "[DOC]",
        "✅": "[OK]",  "⚠️": "[WARN]", "❌": "[ERR]",
    }
    for char, sub in replacements.items():
        text = text.replace(char, sub)
    # Final safety: encode to latin-1, replacing anything that can't be represented
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_markdown(text: str) -> str:
    """Remove basic markdown formatting for clean PDF prose."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)   # *italic*
    text = re.sub(r"^#+\s+",        "",    text,
                  flags=re.MULTILINE)  # headings
    text = re.sub(r"^[-*]\s+",      "",    text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"`(.+?)`",       r"\1", text)   # `code`
    return text.strip()


def _build_recommendations(stats: dict, iqr_cols: dict, df: pd.DataFrame) -> list[str]:
    """Generate context-aware next-step recommendations based on findings."""
    recs = []

    # Data quality
    tc = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    miss_pct = missing / tc * 100 if tc else 0
    if miss_pct > 5:
        recs.append(
            f"Address missing values -- {miss_pct:.1f}% of cells are null. "
            "Consider median/mode imputation for numeric columns and 'Unknown' "
            "for categoricals before further modelling."
        )

    # Anomaly rate
    if stats["anomaly_pct"] > 10:
        recs.append(
            f"Investigate the high anomaly rate ({stats['anomaly_pct']:.1f}%). "
            "This may indicate systematic data quality issues, sensor drift, or "
            "a data pipeline problem upstream. Validate source data before analysis."
        )
    elif stats["anomaly_pct"] > 0:
        recs.append(
            f"Review the {stats['total_anomalies']} flagged rows individually to "
            "determine whether they represent genuine business events (e.g. a sales "
            "spike), data entry errors, or sensor/system failures."
        )

    # Per-column recommendations
    for col, info in list(iqr_cols.items())[:3]:
        recs.append(
            f"Column '{col}': {info['count']} values exceed the expected range "
            f"[{info['lower']:.2f}, {info['upper']:.2f}]. "
            "Verify these records against source systems and consider capping or "
            "removing them before using this column in predictive models."
        )

    # General
    recs.append(
        "Re-run anomaly detection after data cleaning to confirm issues are resolved "
        "and to establish a clean baseline for ongoing monitoring."
    )

    if not recs:
        recs.append(
            "No significant anomalies were detected. The dataset appears clean "
            "with the current sensitivity settings. Consider lowering the IQR "
            "multiplier to surface subtler outliers if needed."
        )

    return recs
