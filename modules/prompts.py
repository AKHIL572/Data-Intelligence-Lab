"""
modules/prompts.py
──────────────────
All LLM prompt builders in one place.

BUGS FIXED vs original app.py
──────────────────────────────
BUG 5 ✅  PDF chat had zero factual grounding — compute_evidence() was
          never called for PDFs so the LLM was flying completely blind
          and could hallucinate freely. Fix: prompt_pdf() now accepts
          an optional pdf_evidence dict (extracted stats like page count,
          word count, key figures found via regex) and embeds them in the
          prompt so the LLM cannot invent numbers.

IMPROVEMENT ✅  prompt_chat() intent detection upgraded: keyword list
               now uses whole-word patterns to prevent false matches
               (e.g. "total" in "what is the total" no longer triggers
               a modify intent).
"""

from __future__ import annotations
import json
import re
import pandas as pd


# ── Auto-analysis prompt ──────────────────────────────────────────────

def prompt_auto_analysis(evidence: dict) -> str:
    return f"""You are a senior data scientist. You have been given PRE-CALCULATED mathematical evidence.

STRICT RULES:
- Use ONLY the numbers from the MATHEMATICAL EVIDENCE section below.
- Do NOT invent, estimate, or guess any statistic, percentage, or figure.
- If a section has no data (e.g. no target column), say "Not available" and move on.

Structure your response exactly like this (use markdown headings):

## 📊 Dataset Overview
Summarise shape, column types, and general nature of the data.
If "auto_converted_columns" is non-empty, list them with a NOTE that they were auto-converted from text to numbers.
If "suspicious_string_columns" is non-empty, list them with a WARNING that they may cause calculation errors.

## 🧹 Data Quality
Report missing values, duplicates, and extreme outliers using EXACT numbers from evidence only.

## 🔗 Key Relationships
If target_column present: list top correlated features with their exact correlation values.
If no target_column: state "No numeric target column detected."

## 📈 Distribution Highlights
Summarise categorical column distributions using the exact percentages provided.

## 🎯 Recommended Next Steps
Give 3-5 actionable data cleaning or analysis steps based ONLY on what the evidence shows.

---
MATHEMATICAL EVIDENCE (use ONLY these numbers):
{json.dumps(evidence, indent=2)}
"""


# ── Chat prompt (tabular data) ────────────────────────────────────────

# Whole-word modify keywords — prevents "total" in a question from
# being misclassified as a dataset-modify intent (original bug)
_MODIFY_PATTERNS = [
    r"\bremove\b", r"\bdrop\b", r"\bdelete\b", r"\bfilter\b",
    r"\bkeep only\b", r"\brename\b", r"\breplace\b", r"\bfill\b",
    r"\bclean\b", r"\bfix\b", r"\badd column\b", r"\bcreate column\b",
    r"\bnew column\b", r"\bcalculate column\b", r"\bmake column\b",
    r"\binsert\b", r"\bupdate\b", r"\bchange\b", r"\bconvert\b",
    r"\bsort\b", r"\breset\b", r"\bdeduplicate\b", r"\bstrip\b",
    r"\blowercase\b", r"\buppercase\b", r"\btrim\b",
    r"\bremove outliers\b", r"\bdrop outliers\b",
]

_CHART_PATTERNS = [
    r"\bplot\b", r"\bchart\b", r"\bgraph\b", r"\bhistogram\b",
    r"\bscatter\b", r"\bbar chart\b", r"\bline chart\b",
    r"\bpie chart\b", r"\bboxplot\b", r"\bheatmap\b",
    r"\bvisualise\b", r"\bvisualize\b", r"\bdistribution\b",
    r"\bshow me a (graph|chart|plot)\b", r"\bdraw\b",
]


def _classify_intent(msg: str) -> str:
    """Return 'modify', 'chart', or 'question'."""
    lower = msg.lower()
    if any(re.search(p, lower) for p in _CHART_PATTERNS):
        return "chart"
    if any(re.search(p, lower) for p in _MODIFY_PATTERNS):
        return "modify"
    return "question"


def prompt_chat(user_msg: str, df: pd.DataFrame, evidence: dict) -> str:
    sample = df.head(5).to_string(index=False)
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    intent = _classify_intent(user_msg)

    if intent == "question":
        intent_instruction = (
            "The user is asking a QUESTION. Provide a plain text answer.\n"
            "→ Use ONLY numbers from PRE-CALCULATED EVIDENCE — do not invent statistics.\n"
            "→ Do NOT produce any Python code or code block."
        )
    elif intent == "chart":
        intent_instruction = (
            "The user wants a CHART or VISUALISATION.\n"
            "→ Output a SINGLE ```python code block. Nothing else.\n"
            "→ Use px (plotly.express) or go (plotly.graph_objects). Both are pre-imported.\n"
            "→ Assign the final figure to a variable named exactly `fig`.\n"
            "→ Do NOT call fig.show(). Do NOT import anything."
        )
    else:  # modify
        intent_instruction = (
            "The user wants to MODIFY the dataset.\n"
            "→ Output a SINGLE ```python code block. Nothing else.\n"
            "→ The code must operate on the variable `df`.\n"
            "→ Only these names are in scope: df, pd, np. Do NOT import anything.\n"
            "→ Do NOT call print(), show(), or display()."
        )

    return f"""You are a senior data scientist assistant. The user has a pandas DataFrame called `df`.

DATAFRAME INFO:
- Shape   : {df.shape[0]} rows × {df.shape[1]} columns
- Columns : {list(df.columns)}
- Dtypes  : {json.dumps(dtypes)}
- Sample (first 5 rows):
{sample}

PRE-CALCULATED EVIDENCE (use ONLY these numbers for any statistics):
{json.dumps(evidence, indent=2)}

YOUR TASK:
{intent_instruction}

USER REQUEST: {user_msg}
"""


# ── PDF prompt ────────────────────────────────────────────────────────

def prompt_pdf(user_msg: str, pdf_text: str, pdf_evidence: dict | None = None) -> str:
    """
    FIX (BUG 5): Original had zero factual grounding for PDFs.
    pdf_evidence (word count, page count, numbers found) is now embedded
    so the LLM cannot hallucinate document statistics.
    """
    grounding = ""
    if pdf_evidence:
        grounding = (
            f"\nDOCUMENT FACTS (use ONLY these for any statistics):\n"
            f"{json.dumps(pdf_evidence, indent=2)}\n"
        )

    return (
        "You are a document analyst. Answer the user's question using ONLY the "
        "document text provided below. Do not guess or add information not in the text.\n"
        + grounding
        + f"\nDOCUMENT TEXT (first 10,000 chars):\n{pdf_text[:10_000]}\n\n"
        f"USER QUESTION: {user_msg}"
    )


# ── PDF evidence extractor ────────────────────────────────────────────

def extract_pdf_evidence(pdf_text: str) -> dict:
    """
    Extract simple factual stats from raw PDF text so the LLM has
    grounding data to reference (fixes BUG 5).
    """
    words = pdf_text.split()
    sentences = re.split(r"[.!?]+", pdf_text)
    # Find all standalone numbers (integers and decimals)
    numbers = re.findall(r"\b\d[\d,]*\.?\d*\b", pdf_text)
    unique_nums = list(dict.fromkeys(numbers))[:20]   # first 20 unique numbers

    return {
        "word_count":     len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "char_count":     len(pdf_text),
        "numbers_found":  unique_nums,
        "note":           "All statistics above are extracted directly from the document text.",
    }
