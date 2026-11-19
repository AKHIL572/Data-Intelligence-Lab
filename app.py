import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import pathlib
import warnings
import re
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.llm import ollama_status, ollama_generate
from modules.file_loader import load_uploaded_file
from modules.evidence import compute_evidence
from modules.sandbox import safe_exec, extract_code_block
from modules.prompts import (
    prompt_auto_analysis,
    prompt_chat,
    prompt_pdf,
    extract_pdf_evidence,
)

SAMPLE_DIR = pathlib.Path(__file__).parent / "assets" / "sample_data"

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Intelligence Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  —  zero external calls, OS system fonts only
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Trebuchet MS', 'Century Gothic', Futura, sans-serif;
    background-color: #070b10;
    color: #c8d6e5;
}
section[data-testid="stSidebar"] {
    background: #0b0f16;
    border-right: 1px solid #19283a;
}
section[data-testid="stSidebar"] * { color: #7a9db8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #33ccff !important; }
h1 { font-weight:800; color:#ffffff !important; letter-spacing:-1px; }
h2, h3 { color:#33ccff !important; }
[data-testid="metric-container"] {
    background:#0b0f16; border:1px solid #19283a;
    border-radius:8px; padding:14px 12px;
}
[data-testid="metric-container"] label {
    color:#3d5a73 !important; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#33ccff !important;
    font-family: Consolas,'Cascadia Code','Fira Code',monospace;
    font-size:1.4rem !important;
}
[data-testid="stDataFrame"] { border:1px solid #19283a; border-radius:8px; }
[data-testid="stChatMessage"] {
    background:#0b0f16; border:1px solid #19283a;
    border-radius:10px; margin-bottom:10px; padding:6px 10px;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li { color:#c8d6e5; line-height:1.7; }
code, pre {
    font-family: Consolas,'Cascadia Code','Fira Code','Courier New',monospace !important;
    background:#0e1620 !important; border:1px solid #19283a !important;
    color:#7fffb2 !important; font-size:0.8rem; border-radius:6px;
}
.stButton > button {
    background:linear-gradient(135deg,#0d2e52,#111d2e);
    color:#33ccff; border:1px solid #33ccff44;
    border-radius:6px; font-family:Consolas,monospace; font-size:0.78rem;
    padding:6px 16px; transition:all .2s;
}
.stButton > button:hover {
    background:#33ccff18; border-color:#33ccff; color:#fff; box-shadow:0 0 16px #33ccff33;
}
[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#0d3d2a,#0a1f15);
    color:#7fffb2; border:1px solid #7fffb244;
    border-radius:6px; font-family:Consolas,monospace; font-size:0.78rem;
    padding:6px 16px; transition:all .2s; width:100%;
}
[data-testid="stDownloadButton"] > button:hover {
    background:#7fffb218; border-color:#7fffb2; color:#fff; box-shadow:0 0 16px #7fffb233;
}
[data-testid="stFileUploader"] { background:#0b0f16; border:1px dashed #19283a; border-radius:8px; }
[data-testid="stChatInput"] textarea {
    background:#0b0f16 !important; border:1px solid #19283a !important;
    color:#c8d6e5 !important; font-family:Consolas,monospace !important;
}
details { background:#0b0f16; border:1px solid #19283a !important; border-radius:8px; }
summary { color:#33ccff; font-size:0.83rem; padding:4px 0; }
hr { border-color:#19283a !important; }
[data-testid="stSelectbox"] > div > div {
    background:#0b0f16 !important; border:1px solid #19283a !important; color:#c8d6e5 !important;
}
[data-testid="stAlert"] { background:#0b0f16; border:1px solid #19283a; border-radius:8px; }
.badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-family:Consolas,monospace; font-size:0.68rem; border:1px solid; margin:2px;
}
.badge-green  { color:#7fffb2; border-color:#7fffb244; background:#7fffb20d; }
.badge-red    { color:#ff6b6b; border-color:#ff6b6b44; background:#ff6b6b0d; }
.badge-blue   { color:#33ccff; border-color:#33ccff44; background:#33ccff0d; }
.badge-orange { color:#ffb347; border-color:#ffb34744; background:#ffb3470d; }
.badge-purple { color:#c792ff; border-color:#c792ff44; background:#c792ff0d; }
.section-label {
    font-family:Consolas,monospace; font-size:0.6rem; letter-spacing:2.5px;
    text-transform:uppercase; color:#33ccff66; margin-bottom:6px; margin-top:2px;
}
.welcome-card {
    background:linear-gradient(135deg,#0b0f16 0%,#0d1e30 100%);
    border:1px solid #19283a; border-radius:14px;
    padding:2.5rem 2rem; text-align:center; margin:3rem auto; max-width:680px;
}
.welcome-card h2 { color:#ffffff !important; font-size:1.6rem; margin-bottom:1rem; }
.welcome-card p  { color:#3d5a73; font-size:0.9rem; line-height:1.8; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "df":            None,
    "df_original":   None,
    "df_modified":   False,
    "pdf_text":      "",
    # ← NEW: grounding facts for PDF chat (fixes Bug 5)
    "pdf_evidence":  {},
    "file_name":     "",
    # NOTE: fig objects NOT stored here (fixes memory bug)
    "messages":      [],
    "analysis_done": False,
    "ollama_model":  "llama3",
    "dtype_fixes":   [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _apply_plotly_theme(fig) -> None:
    fig.update_layout(
        paper_bgcolor="#070b10",
        plot_bgcolor="#070b10",
        font_color="#c8d6e5",
        template="plotly_dark",
    )


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.2rem;line-height:1'>🔬</div>
        <div style='font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-.3px;margin-top:6px'>
            DATA INTELLIGENCE LAB
        </div>
        <div style='font-family:Consolas,monospace;font-size:.58rem;color:#33ccff66;
                    letter-spacing:2.5px;margin-top:3px'>
            PRIVATE · OFFLINE · v2.1
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Ollama status (cached — no freeze on every rerender) ──────────
    ollama_ok, available_models = ollama_status()

    if ollama_ok:
        st.markdown(
            '<div class="badge badge-green">● OLLAMA ONLINE</div>', unsafe_allow_html=True)
        st.write("")
        _PREF = ["llama3", "mistral", "gemma",
                 "phi", "llama2", "qwen", "deepseek"]
        sorted_models = sorted(
            available_models,
            key=lambda m: next(
                (i for i, p in enumerate(_PREF) if p in m.lower()), 99),
        )
        chosen_model = st.selectbox(
            "Active model",
            options=sorted_models or _PREF,
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.ollama_model = chosen_model
    else:
        st.markdown(
            '<div class="badge badge-red">✗ OLLAMA OFFLINE</div>', unsafe_allow_html=True)
        st.caption(
            "**Start Ollama:**\n```\nollama serve\n```\n**Pull a model:**\n```\nollama pull llama3\n```\nThen refresh.")
        st.session_state.ollama_model = "llama3"

    st.divider()

    # ── Active Dataset Info (Upload is in main working area) ───────────
    if st.session_state.df is not None or st.session_state.pdf_text:
        fname = st.session_state.get("file_name", "Active Document")
        st.markdown(f'<div style="font-family:Consolas,monospace;font-size:0.75rem;color:#7fffb2;word-break:break-all">📄 <b>{fname}</b></div>', unsafe_allow_html=True)
        if st.button("🔄 Change Dataset", use_container_width=True, key="app_sidebar_change_file"):
            st.session_state.df = None
            st.session_state.df_original = None
            st.session_state.df_modified = False
            st.session_state.pdf_text = ""
            st.session_state.pdf_evidence = {}
            st.session_state.file_name = ""
            st.session_state.dash_df = None
            st.session_state.dash_file = ""
            st.session_state.chat_df = None
            st.session_state.chat_file = ""
            st.session_state.messages = []
            st.session_state.analysis_done = False
            st.rerun()

    # ── Data Health panel ─────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        total_cells = df.shape[0] * df.shape[1]
        n_missing = int(df.isnull().sum().sum())
        miss_pct = round(n_missing / total_cells *
                         100, 1) if total_cells else 0
        quality_pct = round(100 - miss_pct, 1)
        n_dupes = int(df.duplicated().sum())
        n_numeric = len(df.select_dtypes(include="number").columns)
        n_outlier_cols = 0
        for c in df.select_dtypes(include="number").columns:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and not df[(df[c] > q3 + 3*iqr) | (df[c] < q1 - 3*iqr)].empty:
                n_outlier_cols += 1

        st.divider()
        st.markdown('<div class="section-label">📊 Data Health</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows",    f"{df.shape[0]:,}")
        c2.metric("Cols",    df.shape[1])
        c3, c4 = st.columns(2)
        c3.metric("Quality", f"{quality_pct}%")
        c4.metric("Numeric", n_numeric)
        c5, c6 = st.columns(2)
        c5.metric("Missing", f"{n_missing:,}", delta_color="inverse")
        c6.metric("Dupes",   n_dupes,          delta_color="inverse")

        if n_outlier_cols:
            st.markdown(
                f'<div class="badge badge-orange">⚠ {n_outlier_cols} col(s) have extreme outliers</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        if st.button("↺  Reset to Original", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.df_modified = False
            st.session_state.messages = []
            st.session_state.analysis_done = False
            st.rerun()

        if st.session_state.df_modified:
            st.markdown(
                '<div class="badge badge-orange">● Dataset has been modified</div>', unsafe_allow_html=True)
            st.write("")
            base = st.session_state.file_name.rsplit(".", 1)[0]
            st.download_button(
                label="⬇  Download Updated CSV",
                data=to_csv_bytes(df),
                file_name=f"updated_{base}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.markdown('<div class="section-label">🔒 Privacy</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="badge badge-green">✓ Zero external calls</div>&nbsp;'
            '<div class="badge badge-green">✓ Data in RAM only</div>',
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px'>🔬 Private Data Intelligence Lab</h1>"
    "<p style='color:#2d4a63;font-size:.78rem;margin-top:2px;"
    "font-family:Consolas,monospace;letter-spacing:.5px'>"
    "100% OFFLINE &nbsp;·&nbsp; OLLAMA &nbsp;·&nbsp; ZERO TELEMETRY &nbsp;·&nbsp; v2.1 (Bug-Fixed)"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Welcome screen ────────────────────────────────────────────────────
if st.session_state.df is None and not st.session_state.pdf_text:
    st.markdown("""
    <div class="welcome-card">
        <h2>Get Started: Upload a Dataset or Choose Sample Data</h2>
        <p>
            Drop a <strong>CSV</strong>, <strong>Excel (XLSX)</strong>, or
            <strong>PDF</strong> file below or in the sidebar.<br>
            All statistics are computed locally before the LLM sees data — ensuring 100% privacy and zero hallucinations.
        </p>
        <div>
            <span class="badge badge-blue">🔒 100% Private</span>
            <span class="badge badge-green">⚡ Any Ollama Model</span>
            <span class="badge badge-blue">📊 Plotly Charts</span>
            <span class="badge badge-green">🛡 Sandboxed exec()</span>
            <span class="badge badge-blue">↺ Reset Anytime</span>
            <span class="badge badge-purple">📄 PDF Support</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col_up, col_sample = st.columns([1.2, 1], gap="large")

    with col_up:
        st.markdown("### 📁 Upload File")
        main_uploaded = st.file_uploader(
            "Upload CSV, Excel, or PDF",
            type=["csv", "xlsx", "xls", "pdf"],
            key="main_file_uploader",
        )
        if main_uploaded:
            with st.spinner("Processing uploaded file…"):
                new_df, new_pdf, dtype_fixes = load_uploaded_file(main_uploaded)
            if new_df is not None or new_pdf:
                st.session_state.df = new_df
                st.session_state.dash_df = new_df
                st.session_state.chat_df = new_df
                st.session_state.df_original = new_df.copy() if new_df is not None else None
                st.session_state.df_modified = False
                st.session_state.pdf_text = new_pdf
                st.session_state.pdf_evidence = extract_pdf_evidence(new_pdf) if new_pdf else {}
                st.session_state.file_name = main_uploaded.name
                st.session_state.dash_file = main_uploaded.name
                st.session_state.chat_file = main_uploaded.name
                st.session_state.messages = []
                st.session_state.analysis_done = False
                st.session_state.dtype_fixes = dtype_fixes
                st.rerun()

    with col_sample:
        st.markdown("### ⚡ Or Try Sample Data")
        samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
        if samples:
            sample_names = [s.name for s in samples]
            chosen_sample = st.selectbox("Select a sample dataset:", sample_names, index=0)
            if st.button("🚀 Load Sample Dataset", use_container_width=True):
                sample_path = SAMPLE_DIR / chosen_sample
                sample_df = pd.read_csv(sample_path)
                sample_df, fixes = load_uploaded_file(open(sample_path, "rb")) if hasattr(open(sample_path, "rb"), "name") else (sample_df, "", [])
                st.session_state.df = sample_df
                st.session_state.dash_df = sample_df
                st.session_state.chat_df = sample_df
                st.session_state.df_original = sample_df.copy()
                st.session_state.df_modified = False
                st.session_state.file_name = chosen_sample
                st.session_state.dash_file = chosen_sample
                st.session_state.chat_file = chosen_sample
                st.session_state.messages = []
                st.session_state.analysis_done = False
                st.session_state.dtype_fixes = fixes
                st.rerun()
        else:
            st.info("No sample datasets found in `assets/sample_data`.")

    st.stop()


# ── Dataset overview metrics ──────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    num_df = df.select_dtypes(include="number")
    tc = df.shape[0] * df.shape[1]
    miss = int(df.isnull().sum().sum())

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Rows",    f"{df.shape[0]:,}")
    oc2.metric("Columns", df.shape[1])
    oc3.metric("Numeric", len(num_df.columns))
    oc4.metric("Quality", f"{100 - miss/tc*100:.1f}%" if tc else "—")

    with st.expander("📋  Dataset Preview & Statistics", expanded=False):
        st.dataframe(df.head(50), use_container_width=True, height=280)

        tab_stats, tab_corr, tab_nulls, tab_dtypes = st.tabs(
            ["Descriptive Stats", "Correlation Matrix",
                "Missing Values", "⚠️ Data Types"]
        )
        with tab_stats:
            st.dataframe(df.describe(include="all").round(3),
                         use_container_width=True)

        with tab_corr:
            if not num_df.empty and num_df.shape[1] >= 2:
                fig_corr = px.imshow(
                    num_df.corr(), color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, template="plotly_dark",
                    title="Pearson Correlation Matrix",
                )
                _apply_plotly_theme(fig_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation matrix.")

        with tab_nulls:
            null_counts = df.isnull().sum()
            null_df = pd.DataFrame({
                "Column":  null_counts.index,
                "Missing": null_counts.values,
                "Percent": (null_counts.values / len(df) * 100).round(2),
            }).sort_values("Missing", ascending=False)
            null_df = null_df[null_df["Missing"] > 0]
            if null_df.empty:
                st.success("✅ No missing values found.")
            else:
                st.dataframe(null_df, use_container_width=True,
                             hide_index=True)
                fig_null = px.bar(
                    null_df, x="Column", y="Percent",
                    title="Missing Values (%)", template="plotly_dark",
                    color="Percent", color_continuous_scale="OrRd",
                )
                _apply_plotly_theme(fig_null)
                st.plotly_chart(fig_null, use_container_width=True)

        with tab_dtypes:
            dtype_fixes = st.session_state.get("dtype_fixes", [])
            all_obj_cols = df.select_dtypes(include="object").columns.tolist()
            suspicious = []
            for col in all_obj_cols:
                samp = df[col].dropna().astype(str).str.strip().head(100)
                cleaned = samp.str.replace(r"[$€£¥₹,%,]", "", regex=True)
                trial = pd.to_numeric(cleaned, errors="coerce")
                rate = trial.notna().sum() / max(len(trial), 1)
                if rate >= 0.40:
                    suspicious.append(
                        (col, rate, df[col].dropna().astype(str).head(3).tolist()))

            if dtype_fixes:
                st.success(
                    f"✅ {len(dtype_fixes)} column(s) auto-converted from text → number.")
                for c in dtype_fixes:
                    st.markdown(f"- `{c}` → **{df[c].dtype}**")
                st.divider()

            if suspicious:
                st.warning(
                    f"⚠️ {len(suspicious)} column(s) look numeric but are stored as text.")
                rows = [{"Column": col, "% Numeric": f"{r*100:.0f}%",
                         "Samples": " | ".join(s)} for col, r, s in suspicious]
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)
            elif not dtype_fixes:
                st.success("✅ All columns have correct data types.")

            st.divider()
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type":   [str(df[c].dtype) for c in df.columns],
                "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                           for c in df.columns],
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.divider()


# ────────────────────────────────────────────────────────────────────────────
# AUTO-ANALYSIS — fires once per new file upload
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    if not ollama_ok:
        st.warning(
            "⚠️ **Ollama is not running.** Start it with:\n```bash\nollama serve\n```")
    else:
        model = st.session_state.ollama_model

        if st.session_state.df is not None:
            with st.spinner("🔬 Calculating statistical evidence…"):
                # BUG 1b FIX: dtype_fixes passed explicitly — not read from session_state inside function
                evidence = compute_evidence(
                    st.session_state.df,
                    tuple(st.session_state.dtype_fixes),
                )
            with st.spinner(f"🤖 Generating data report with **{model}**…"):
                report = ollama_generate(prompt_auto_analysis(evidence), model)

            st.session_state.messages.append({
                "role": "assistant", "content": report,
                "code": None, "error": None,
                # NOTE: fig NOT stored in session_state to avoid memory bloat
            })
            st.session_state.analysis_done = True
            st.rerun()

        elif st.session_state.pdf_text:
            with st.spinner(f"📄 Analysing PDF with **{model}**…"):
                pdf_prompt = (
                    "Summarise this document using ONLY the text and facts provided. "
                    "Do not add or invent any information.\n"
                    "Structure:\n"
                    "## 📄 What is this document?\n"
                    "## 🔑 Key Points\n"
                    "## 🔢 Notable Data or Figures\n"
                    "## 💡 Summary\n\n"
                    f"DOCUMENT FACTS:\n{json.dumps(st.session_state.pdf_evidence, indent=2)}\n\n"
                    f"TEXT:\n{st.session_state.pdf_text}"
                )
                report = ollama_generate(pdf_prompt, model)

            st.session_state.messages.append({
                "role": "assistant", "content": report,
                "code": None, "error": None,
            })
            st.session_state.analysis_done = True
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">💬 Chat with your Data</div>',
            unsafe_allow_html=True)

# ── Render history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🔬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("code"):
            with st.expander("🧬 Generated Code", expanded=False):
                st.code(msg["code"], language="python")
        if msg.get("error"):
            with st.expander("⚠️ Execution Error", expanded=True):
                st.code(msg["error"], language="text")

# ── Input bar ─────────────────────────────────────────────────────────
_hint = (
    "Ask a question, request a change, or ask for a chart…"
    if ollama_ok else "Ollama is offline — start it with: ollama serve"
)

# Input length guard — prevents OOM from massive prompts
user_input = st.chat_input(_hint)
if user_input and len(user_input) > 2000:
    st.warning(
        "⚠️ Message too long (max 2000 characters). Please shorten your request.")
    user_input = None

if user_input:
    if not ollama_ok:
        st.error("❌ Ollama is offline. Run `ollama serve` in a terminal.")
        st.stop()

    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "code": None, "error": None,
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🔬"):
        df_now = st.session_state.df
        model = st.session_state.ollama_model

        if df_now is not None:
            evidence = compute_evidence(
                df_now, tuple(st.session_state.dtype_fixes))
            prompt = prompt_chat(user_input, df_now, evidence)
        elif st.session_state.pdf_text:
            prompt = prompt_pdf(
                user_input,
                st.session_state.pdf_text,
                st.session_state.pdf_evidence,   # ← BUG 5 FIX: grounding evidence
            )
        else:
            prompt = f"The user asked: {user_input}. No data has been loaded yet."

        with st.spinner(f"Thinking with {model}…"):
            raw_response = ollama_generate(prompt, model)

        code_block = extract_code_block(raw_response)
        exec_error = None
        fig_result = None

        # Strip code fences from displayed text
        display_text = re.sub(r"```python.*?```", "",
                              raw_response, flags=re.DOTALL)
        display_text = re.sub(r"```.*?```",        "",
                              display_text, flags=re.DOTALL).strip()
        if display_text:
            st.markdown(display_text)

        if code_block and df_now is not None:
            updated_df, fig_result, exec_error = safe_exec(code_block, df_now)

            if exec_error is None:
                shape_changed = updated_df.shape != df_now.shape
                cols_changed = list(updated_df.columns) != list(df_now.columns)
                try:
                    data_changed = not updated_df.equals(df_now)
                except Exception:
                    data_changed = True

                if shape_changed or cols_changed or data_changed:
                    st.session_state.df = updated_df
                    st.session_state.df_modified = True
                    st.success(
                        f"✅ Dataset updated — "
                        f"{updated_df.shape[0]:,} rows × {updated_df.shape[1]} cols"
                    )
            else:
                st.error("Code execution failed — dataset unchanged.")

            if fig_result is not None:
                _apply_plotly_theme(fig_result)
                st.plotly_chart(fig_result, use_container_width=True)
                # NOTE: fig NOT saved to session_state — prevents memory bloat

            with st.expander("🧬 Generated Code", expanded=False):
                st.code(code_block, language="python")

            if exec_error:
                with st.expander("⚠️ Execution Error", expanded=True):
                    st.code(exec_error, language="text")

    # Save assistant message — WITHOUT fig object
    st.session_state.messages.append({
        "role":    "assistant",
        "content": display_text if code_block else raw_response,
        "code":    code_block,
        "error":   exec_error,
        # fig intentionally NOT stored here
    })


# ────────────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#19283a;font-family:Consolas,monospace;"
    "font-size:.6rem;letter-spacing:1.5px;padding:8px 0'>"
    "PRIVATE DATA INTELLIGENCE LAB &nbsp;·&nbsp; ALL COMPUTATION LOCAL "
    "&nbsp;·&nbsp; ZERO TELEMETRY &nbsp;·&nbsp; v2.1"
    "</div>",
    unsafe_allow_html=True,
)
