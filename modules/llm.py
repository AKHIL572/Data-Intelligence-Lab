"""
modules/llm.py
──────────────
Ollama helpers: status check (cached) + text generation.

BUGS FIXED vs original app.py
──────────────────────────────
BUG 2 ✅  ollama_generate() used isinstance(r, dict) guard then r["response"].
          Ollama SDK >= 0.2 always returns a GenerateResponse object, never a
          plain dict → caused silent KeyError. Now always uses r.response.

BUG 3 ✅  ollama_status() was called on every Streamlit rerender (sidebar
          re-executes on every interaction) with no caching → 1-2 s freeze
          on every button click / chat message. Now cached 30 s via
          st.cache_data so Ollama is only pinged when the cache expires.
"""

from __future__ import annotations
import streamlit as st


# ── Status check (cached 30 s) ────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def ollama_status() -> tuple[bool, list[str]]:
    """
    Return (is_online, [model_name, ...]).
    Cached for 30 seconds — won't re-ping Ollama on every rerender.
    Never raises; returns (False, []) on any failure.
    """
    try:
        import ollama as _ol
        resp = _ol.list()
        models = resp.models if hasattr(
            resp, "models") else resp.get("models", [])
        names: list[str] = []
        for m in models:
            n = getattr(m, "model", None) or m.get("model", m.get("name", ""))
            if n:
                names.append(n)
        return True, names
    except Exception:
        return False, []


# ── Generate ──────────────────────────────────────────────────────────

def ollama_generate(prompt: str, model: str) -> str:
    """
    Send prompt to Ollama and return the response text.

    FIX: original code did `r["response"] if isinstance(r, dict) else r.response`
    The SDK never returns a plain dict → the dict branch KeyError'd silently.
    We always use attribute access now which works on all SDK versions.

    On any failure returns a user-friendly markdown error string so the
    Streamlit UI never crashes.
    """
    try:
        import ollama as _ol
        result = _ol.generate(model=model, prompt=prompt)
        return result.response          # ← always attribute, never dict key
    except Exception as exc:
        return (
            f"⚠️ **Ollama error:** `{exc}`\n\n"
            f"**Fix:**\n"
            f"```bash\n"
            f"ollama serve\n"
            f"ollama pull {model}\n"
            f"```"
        )


# ── Stream ────────────────────────────────────────────────────────────

def ollama_generate_stream(prompt: str, model: str):
    """
    Stream response from Ollama, yielding text chunks as they arrive.

    Provides real-time token display in chat interfaces so the user
    sees text appearing immediately instead of waiting behind a spinner.
    Falls back to a single error chunk on any failure.
    """
    try:
        import ollama as _ol
        for chunk in _ol.generate(model=model, prompt=prompt, stream=True):
            text = getattr(chunk, "response", "") or (
                chunk.get("response", "") if isinstance(chunk, dict) else ""
            )
            if text:
                yield text
    except Exception as exc:
        yield (
            f"⚠️ **Ollama error:** `{exc}`\n\n"
            f"**Fix:**\n"
            f"```bash\n"
            f"ollama serve\n"
            f"ollama pull {model}\n"
            f"```"
        )
