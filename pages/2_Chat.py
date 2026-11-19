import streamlit as st
from modules.prompts import prompt_auto_analysis, prompt_chat
from modules.sandbox import safe_exec, extract_code_block
from modules.evidence import compute_evidence
from modules.file_loader import load_uploaded_file
from modules.llm import ollama_status, ollama_generate
import pathlib
import sys
import json
import os
import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(
    page_title="Data Chat",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
html,body,[class*="css"]{
    font-family:'Trebuchet MS','Century Gothic',Futura,sans-serif;
    background:#070b10;color:#c8d6e5;
}
h1{font-weight:800;color:#fff!important;letter-spacing:-1px;}
h2,h3{color:#33ccff!important;}
[data-testid="stChatMessage"]{
    background:#0b0f16;border:1px solid #19283a;
    border-radius:10px;margin-bottom:10px;padding:6px 10px;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li{color:#c8d6e5;line-height:1.7;}
code,pre{
    font-family:Consolas,'Fira Code',monospace!important;
    background:#0e1620!important;border:1px solid #19283a!important;
    color:#7fffb2!important;font-size:.8rem;border-radius:6px;
}
.stButton>button{
    background:linear-gradient(135deg,#0d2e52,#111d2e);
    color:#33ccff;border:1px solid #33ccff44;border-radius:6px;
    font-family:Consolas,monospace;font-size:.78rem;padding:6px 16px;transition:all .2s;
}
.stButton>button:hover{background:#33ccff18;border-color:#33ccff;color:#fff;}
[data-testid="stDownloadButton"]>button{
    background:linear-gradient(135deg,#0d3d2a,#0a1f15);
    color:#7fffb2;border:1px solid #7fffb244;border-radius:6px;
    font-family:Consolas,monospace;font-size:.78rem;
    padding:6px 16px;transition:all .2s;width:100%;
}
[data-testid="metric-container"]{
    background:#0b0f16;border:1px solid #19283a;border-radius:8px;padding:14px 12px;
}
[data-testid="metric-container"] label{color:#3d5a73!important;font-size:.7rem;
    text-transform:uppercase;letter-spacing:1px;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:#33ccff!important;font-family:Consolas,monospace;font-size:1.4rem!important;}
[data-testid="stChatInput"] textarea{
    background:#0b0f16!important;border:1px solid #19283a!important;
    color:#c8d6e5!important;font-family:Consolas,monospace!important;
}
details{background:#0b0f16;border:1px solid #19283a!important;border-radius:8px;}
summary{color:#33ccff;font-size:.83rem;padding:4px 0;}
hr{border-color:#19283a!important;}
.badge{
    display:inline-block;padding:3px 12px;border-radius:20px;
    font-family:Consolas,monospace;font-size:.68rem;border:1px solid;margin:2px;
}
.badge-green{color:#7fffb2;border-color:#7fffb244;background:#7fffb20d;}
.badge-red  {color:#ff6b6b;border-color:#ff6b6b44;background:#ff6b6b0d;}
.badge-blue {color:#33ccff;border-color:#33ccff44;background:#33ccff0d;}
.badge-orange{color:#ffb347;border-color:#ffb34744;background:#ffb3470d;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "df" in st.session_state and st.session_state.df is not None:
    default_df = st.session_state.df
    default_df_orig = st.session_state.get("df_original", default_df.copy())
    default_file = st.session_state.get("file_name", "Active Dataset")
else:
    default_df = None
    default_df_orig = None
    default_file = ""

for k, v in {
    "chat_df": default_df, "chat_df_original": default_df_orig, "chat_df_modified": False,
    "chat_file": default_file, "chat_messages": [], "chat_analysis_done": False,
    "chat_dtype_fixes": [], "chat_model": "llama3",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.chat_df is None and st.session_state.get("df") is not None:
    st.session_state.chat_df = st.session_state.df
    st.session_state.chat_df_original = st.session_state.get("df_original", st.session_state.df.copy())
    st.session_state.chat_file = st.session_state.get("file_name", "Active Dataset")

SAMPLE_DIR = pathlib.Path(__file__).parent.parent / "assets" / "sample_data"
from modules.file_loader import fix_dtypes


def _theme(fig):
    fig.update_layout(paper_bgcolor="#070b10", plot_bgcolor="#070b10",
                      font_color="#c8d6e5", template="plotly_dark")
    return fig


def to_csv(df): return df.to_csv(index=False).encode("utf-8")


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px'>
        <div style='font-size:1.8rem'>💬</div>
        <div style='font-size:1rem;font-weight:800;color:#fff;margin-top:4px'>DATA CHAT</div>
        <div style='font-family:Consolas,monospace;font-size:.55rem;
                    color:#33ccff66;letter-spacing:2px;margin-top:2px'>
            CLEAN · TRANSFORM · VISUALISE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    ollama_ok, models = ollama_status()
    if ollama_ok:
        st.markdown('<div class="badge badge-green">● OLLAMA ONLINE</div>',
                    unsafe_allow_html=True)
        st.write("")
        _PREF = ["llama3", "mistral", "gemma", "phi"]
        sorted_m = sorted(models, key=lambda m: next(
            (i for i, p in enumerate(_PREF) if p in m.lower()), 99))
        st.session_state.chat_model = st.selectbox(
            "Model", sorted_m or _PREF, index=0, label_visibility="collapsed")
    else:
        st.markdown('<div class="badge badge-red">✗ OLLAMA OFFLINE</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Active Dataset Info (Upload is in main working area) ───────────
    if st.session_state.chat_df is not None:
        fname = st.session_state.get("chat_file", "Active Dataset")
        st.markdown(f'<div style="font-family:Consolas,monospace;font-size:0.75rem;color:#7fffb2;word-break:break-all">📄 <b>{fname}</b></div>', unsafe_allow_html=True)
        if st.button("🔄 Change Dataset", use_container_width=True, key="chat_sidebar_change_file"):
            st.session_state.chat_df = None
            st.session_state.chat_df_original = None
            st.session_state.chat_df_modified = False
            st.session_state.chat_file = ""
            st.session_state.chat_messages = []
            st.session_state.chat_analysis_done = False
            st.rerun()

    if st.session_state.chat_df is not None:
        df = st.session_state.chat_df
        tc = df.shape[0] * df.shape[1]
        miss = int(df.isnull().sum().sum())
        st.divider()
        st.markdown('<div style="font-family:Consolas;font-size:.6rem;letter-spacing:2px;'
                    'text-transform:uppercase;color:#33ccff66;margin-bottom:6px">'
                    '📊 Data Health</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Cols", df.shape[1])
        c3, c4 = st.columns(2)
        c3.metric("Quality", f"{100-miss/tc*100:.1f}%" if tc else "—")
        c4.metric("Numeric", len(df.select_dtypes(include="number").columns))
        st.divider()

        if st.button("↺ Reset to Original", use_container_width=True):
            st.session_state.chat_df = st.session_state.chat_df_original.copy()
            st.session_state.chat_df_modified = False
            st.session_state.chat_messages = []
            st.session_state.chat_analysis_done = False
            st.rerun()

        if st.session_state.chat_df_modified:
            st.markdown('<div class="badge badge-orange">● Dataset modified</div>',
                        unsafe_allow_html=True)
            st.write("")
            base = st.session_state.chat_file.rsplit(".", 1)[0]
            st.download_button("⬇ Download Updated CSV", to_csv(df),
                               f"updated_{base}.csv", "text/csv",
                               use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px'>💬 Data Chat</h1>"
    "<p style='color:#2d4a63;font-size:.78rem;margin-top:2px;"
    "font-family:Consolas,monospace;letter-spacing:.5px'>"
    "CHAT TO CLEAN · TRANSFORM · VISUALISE YOUR DATA &nbsp;·&nbsp; LOCALLY"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.chat_df is None:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0b0f16,#0d1e30);
                border:1px solid #19283a;border-radius:14px;
                padding:2.5rem 2rem;text-align:center;margin:1.5rem auto;max-width:720px'>
        <div style='font-size:2.5rem'>💬</div>
        <h2 style='color:#fff!important;margin:.6rem 0'>Load a Dataset to Start Chatting</h2>
        <p style='color:#7a9db8;font-size:.9rem;line-height:1.8'>
            Upload a CSV / Excel file below or select a bundled sample to converse with your dataset, run transformations, and generate charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_upload, c_sample = st.columns([1.2, 1], gap="large")
    with c_upload:
        st.markdown("### 📁 Upload File")
        chat_main_up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="chat_main_uploader")
        if chat_main_up:
            new_df, _, fixes = load_uploaded_file(chat_main_up)
            if new_df is not None:
                st.session_state.chat_df = new_df
                st.session_state.df = new_df
                st.session_state.chat_df_original = new_df.copy()
                st.session_state.chat_df_modified = False
                st.session_state.chat_file = chat_main_up.name
                st.session_state.file_name = chat_main_up.name
                st.session_state.chat_messages = []
                st.session_state.chat_analysis_done = False
                st.session_state.chat_dtype_fixes = fixes
                st.rerun()

    with c_sample:
        st.markdown("### ⚡ Or Select Sample Dataset")
        samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
        if samples:
            chosen = st.selectbox("Available sample files:", [s.name for s in samples], key="chat_main_sample_select")
            if st.button("🚀 Load Sample Dataset", use_container_width=True, key="chat_load_sample_main"):
                path = SAMPLE_DIR / chosen
                raw_df = pd.read_csv(path)
                clean_df, fixes = fix_dtypes(raw_df)
                st.session_state.chat_df = clean_df
                st.session_state.df = clean_df
                st.session_state.chat_df_original = clean_df.copy()
                st.session_state.chat_df_modified = False
                st.session_state.chat_file = chosen
                st.session_state.file_name = chosen
                st.session_state.chat_messages = []
                st.session_state.chat_analysis_done = False
                st.session_state.chat_dtype_fixes = fixes
                st.rerun()
        else:
            st.info("No sample datasets found.")

    st.stop()

df = st.session_state.chat_df
num_df = df.select_dtypes(include="number")
tc = df.shape[0] * df.shape[1]
miss = int(df.isnull().sum().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Numeric", len(num_df.columns))
c4.metric("Quality", f"{100-miss/tc*100:.1f}%" if tc else "—")

with st.expander("📋 Dataset Preview", expanded=False):
    st.dataframe(df.head(50), use_container_width=True, height=250)
    t1, t2, t3 = st.tabs(["Stats", "Correlation", "Missing"])
    with t1:
        st.dataframe(df.describe(include="all").round(3),
                     use_container_width=True)
    with t2:
        if not num_df.empty and num_df.shape[1] >= 2:
            fig_c = px.imshow(num_df.corr(), color_continuous_scale="RdBu",
                              zmin=-1, zmax=1, template="plotly_dark",
                              title="Correlation Matrix")
            st.plotly_chart(_theme(fig_c), use_container_width=True)
        else:
            st.info("Need ≥2 numeric columns.")
    with t3:
        nc = df.isnull().sum()
        nd = pd.DataFrame({"Column": nc.index, "Missing": nc.values,
                           "Percent": (nc.values/len(df)*100).round(2)}
                          ).sort_values("Missing", ascending=False)
        nd = nd[nd["Missing"] > 0]
        if nd.empty:
            st.success("✅ No missing values.")
        else:
            st.dataframe(nd, use_container_width=True, hide_index=True)

st.divider()

# ── Auto-analysis ─────────────────────────────────────────────────────
if not st.session_state.chat_analysis_done:
    if not ollama_ok:
        st.warning("⚠️ Ollama offline. Run `ollama serve`.")
    else:
        model = st.session_state.chat_model
        with st.spinner("🔬 Calculating evidence…"):
            evidence = compute_evidence(
                df, tuple(st.session_state.chat_dtype_fixes))
        with st.spinner(f"🤖 Generating report with {model}…"):
            report = ollama_generate(prompt_auto_analysis(evidence), model)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": report, "code": None, "error": None})
        st.session_state.chat_analysis_done = True
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────
st.markdown('<div style="font-family:Consolas;font-size:.6rem;letter-spacing:2px;'
            'text-transform:uppercase;color:#33ccff66;margin-bottom:6px">'
            '💬 Chat with your Data</div>', unsafe_allow_html=True)

for msg in st.session_state.chat_messages:
    av = "👤" if msg["role"] == "user" else "🔬"
    with st.chat_message(msg["role"], avatar=av):
        st.markdown(msg["content"])
        if msg.get("code"):
            with st.expander("🧬 Generated Code", expanded=False):
                st.code(msg["code"], language="python")
        if msg.get("error"):
            with st.expander("⚠️ Error", expanded=True):
                st.code(msg["error"], language="text")

# ── Input ─────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question, request a change, or ask for a chart…"
    if ollama_ok else "Ollama offline — run: ollama serve"
)
if user_input and len(user_input) > 2000:
    st.warning("Message too long (max 2000 chars).")
    user_input = None

if user_input:
    if not ollama_ok:
        st.error("❌ Ollama offline.")
        st.stop()

    st.session_state.chat_messages.append(
        {"role": "user", "content": user_input, "code": None, "error": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🔬"):
        model = st.session_state.chat_model
        evidence = compute_evidence(
            df, tuple(st.session_state.chat_dtype_fixes))
        prompt = prompt_chat(user_input, df, evidence)

        with st.spinner(f"Thinking with {model}…"):
            raw = ollama_generate(prompt, model)

        code_block = extract_code_block(raw)
        exec_error = None
        fig_result = None

        display_text = re.sub(r"```python.*?```", "", raw, flags=re.DOTALL)
        display_text = re.sub(
            r"```.*?```", "", display_text, flags=re.DOTALL).strip()
        if display_text:
            st.markdown(display_text)

        if code_block:
            updated_df, fig_result, exec_error = safe_exec(code_block, df)
            if exec_error is None:
                try:
                    changed = not updated_df.equals(df)
                except:
                    changed = True
                if updated_df.shape != df.shape or list(updated_df.columns) != list(df.columns) or changed:
                    st.session_state.chat_df = updated_df
                    st.session_state.chat_df_modified = True
                    st.success(
                        f"✅ Updated — {updated_df.shape[0]:,} rows × {updated_df.shape[1]} cols")
            else:
                st.error("Code failed — dataset unchanged.")

            if fig_result is not None:
                _theme(fig_result)
                st.plotly_chart(fig_result, use_container_width=True)

            with st.expander("🧬 Generated Code", expanded=False):
                st.code(code_block, language="python")
            if exec_error:
                with st.expander("⚠️ Error", expanded=True):
                    st.code(exec_error, language="text")

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": display_text if code_block else raw,
        "code": code_block, "error": exec_error,
    })

st.divider()
st.markdown(
    "<div style='text-align:center;color:#19283a;font-family:Consolas,monospace;"
    "font-size:.6rem;letter-spacing:1.5px;padding:8px 0'>"
    "DATA CHAT &nbsp;·&nbsp; LOCAL LLM &nbsp;·&nbsp; SANDBOXED EXEC &nbsp;·&nbsp; ZERO TELEMETRY"
    "</div>",
    unsafe_allow_html=True,
)
