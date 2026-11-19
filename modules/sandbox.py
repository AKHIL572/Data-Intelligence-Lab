"""
modules/sandbox.py
──────────────────
Hardened exec() sandbox for running LLM-generated Python code.

BUGS FIXED vs original app.py
──────────────────────────────
BUG 3 ✅  extract_code_block() fallback regex matched ANY triple-backtick
          block — including markdown tables, bash snippets, and JSON.
          If an LLM explanation contained a fenced code block of any kind,
          safe_exec() would try to run it as Python.
          Fix: fallback is now much stricter — requires at least one of a
          specific set of DataFrame/Plotly operation tokens to match, and
          explicitly rejects blocks that start with shell/bash indicators.

BUG 4 ✅  safe_exec() pre-coerced df into df_clean, ran exec() on df_clean,
          but returned namespace.get("df", df) — the *original* uncoerced df.
          If the LLM code operated on df_clean's corrected values and saved
          results back to "df", those results were returned. BUT if the LLM
          code modified df in-place without reassignment, nothing was captured
          because the namespace "df" still pointed to df_clean.
          Fix: namespace["df"] is now initialised to df_clean (the coerced
          copy), so any modification or reassignment to "df" inside exec()
          is captured correctly. The coercion also now happens unconditionally
          so the caller always gets back consistent data.
"""

from __future__ import annotations
import re
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# ── Blocked imports ───────────────────────────────────────────────────

_BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "requests", "urllib", "httpx", "http", "ftplib", "smtplib",
    "pickle", "shelve", "sqlite3", "builtins", "importlib",
    "ctypes", "threading", "multiprocessing", "asyncio",
    "paramiko", "boto3", "google", "openai", "anthropic", "ollama",
    "glob", "tempfile", "io",
})

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "bytes": bytes, "callable": callable, "chr": chr, "dict": dict,
    "dir": dir, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
    "hash": hash, "hex": hex, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "oct": oct,
    "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "setattr": setattr, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "vars": vars, "zip": zip,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "Exception": Exception,
    "NotImplementedError": NotImplementedError,
    "StopIteration": StopIteration,
}


class _ImportBlocker:
    """sys.meta_path finder that blocks dangerous module imports."""

    def find_spec(self, name, path, target=None):       # ← Python 3.12+ API
        if name.split(".")[0] in _BLOCKED_IMPORTS:
            raise ImportError(
                f"🔒 Security: import of '{name}' is blocked in generated code."
            )
        return None

    # Keep find_module for Python < 3.12 compatibility
    def find_module(self, name, path=None):
        if name.split(".")[0] in _BLOCKED_IMPORTS:
            return self
        return None

    def load_module(self, name):
        raise ImportError(
            f"🔒 Security: import of '{name}' is blocked in generated code."
        )


# ── Code extraction ───────────────────────────────────────────────────

# Tokens that strongly suggest Python DataFrame / Plotly code
_PY_TOKENS = frozenset({
    "df[", "df.", "df =", "df=",
    "px.", "go.", "fig =", "fig=",
    ".groupby", ".merge", ".apply", ".assign",
    ".dropna", ".fillna", ".rename", ".sort_values",
    "pd.", "np.",
})

# Indicators that a block is NOT Python (shell / bash / json / markdown)
_NON_PY_PREFIXES = ("$", "#!", "curl", "pip", "apt", "brew", "npm", "{", "[{")


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract the first ```python ... ``` block from LLM output.

    FIX vs original:
    - Primary: only match ```python fenced blocks (not generic ```)
    - Fallback: generic fence now requires ≥1 known Python token AND
      must not start with a shell/JSON indicator.
    This prevents markdown tables, bash commands, or JSON blobs from
    being passed to safe_exec().
    """
    # Primary: explicit python fence
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: generic fence — only accept if it looks like Python
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        first_line = candidate.split("\n")[0].strip()

        # Reject if it looks like a shell command or non-Python block
        if any(first_line.startswith(p) for p in _NON_PY_PREFIXES):
            return None

        # Accept only if at least one strong Python token is present
        if any(tok in candidate for tok in _PY_TOKENS):
            return candidate

    return None


# ── Numeric pre-coercion helper ───────────────────────────────────────

def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silently convert remaining string-numeric columns so LLM-generated
    arithmetic never hits TypeError. Returns a copy.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        trial = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$€£¥₹,%,]", "", regex=True),
            errors="coerce",
        )
        if trial.notna().sum() / max(len(trial.dropna()), 1) >= 0.80:
            df[col] = trial
    return df


# ── Main sandbox ──────────────────────────────────────────────────────

def safe_exec(
    code: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, Optional[object], Optional[str]]:
    """
    Execute LLM-generated code in a sandboxed namespace.

    FIX (BUG 4): Previously, exec() ran on df_clean but the namespace was
    seeded with the original df. Now namespace["df"] is seeded with df_clean
    so any in-place mutation OR reassignment of "df" inside exec() is
    correctly captured when we read namespace["df"] afterwards.

    Returns (updated_df, fig_or_None, error_str_or_None).
    """
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)

    # Coerce numeric columns BEFORE exec so arithmetic doesn't blow up
    df_clean = _coerce_numeric_columns(df)

    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df_clean,          # ← FIX: seed with df_clean, not original df
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }

    try:
        exec(compile(code, "<llm_generated>", "exec"), namespace)

        result_df = namespace.get("df", df_clean)
        fig = namespace.get("fig", None)

        if not isinstance(result_df, pd.DataFrame):
            return df, None, (
                "Generated code replaced `df` with a non-DataFrame. "
                "Reverting to the previous state."
            )
        return result_df, fig, None

    except ImportError as exc:
        return df, None, f"🔒 Blocked import: {exc}"

    except TypeError as exc:
        msg = str(exc)
        if any(kw in msg for kw in ("str", "int", "float", "concatenate")):
            return df, None, (
                "**Data type mismatch.**\n\n"
                "A column used in this calculation is stored as **text** instead of a number.\n\n"
                "Type in the chat: *'Convert `column_name` to numeric'* and retry.\n\n"
                f"*Technical detail: {exc}*"
            )
        return df, None, f"**TypeError:** {exc}"

    except Exception as exc:
        return df, None, (
            f"**Execution error:** {exc}\n\n"
            "Try rephrasing with explicit column names."
        )

    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
