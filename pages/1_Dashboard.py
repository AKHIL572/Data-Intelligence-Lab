import streamlit as st
import pathlib
import sys
from modules.report import generate_report
from modules.anomaly import detect_anomalies, anomaly_summary_stats
from modules.llm import ollama_status, ollama_generate
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


st.set_page_config(
    page_title="BI Anomaly Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── shared CSS reset ──────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[class*="css"]{
    font-family:'Trebuchet MS','Century Gothic',Futura,sans-serif;
    background:#070b10; color:#c8d6e5;
}
h1{font-weight:800;color:#fff!important;letter-spacing:-1px;}
h2,h3{color:#33ccff!important;}
[data-testid="metric-container"]{
    background:#0b0f16;border:1px solid #19283a;
    border-radius:8px;padding:14px 12px;
}
[data-testid="metric-container"] label{
    color:#3d5a73!important;font-size:.7rem;
    text-transform:uppercase;letter-spacing:1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:#33ccff!important;font-family:Consolas,monospace;font-size:1.4rem!important;
}
.stButton>button{
    background:linear-gradient(135deg,#0d2e52,#111d2e);
    color:#33ccff;border:1px solid #33ccff44;border-radius:6px;
    font-family:Consolas,monospace;font-size:.78rem;padding:6px 16px;transition:all .2s;
}
.stButton>button:hover{background:#33ccff18;border-color:#33ccff;color:#fff;}
.badge{
    display:inline-block;padding:3px 12px;border-radius:20px;
    font-family:Consolas,monospace;font-size:.68rem;border:1px solid;margin:2px;
}
.badge-red   {color:#ff6b6b;border-color:#ff6b6b44;background:#ff6b6b0d;}
.badge-orange{color:#ffb347;border-color:#ffb34744;background:#ffb3470d;}
.badge-green {color:#7fffb2;border-color:#7fffb244;background:#7fffb20d;}
.badge-blue  {color:#33ccff;border-color:#33ccff44;background:#33ccff0d;}
.anomaly-card{
    background:#12080a;border:1px solid #ff6b6b33;border-radius:10px;
    padding:14px 18px;margin:6px 0;
}
.anomaly-card .col-name{
    font-family:Consolas,monospace;font-size:.72rem;
    color:#ff6b6b;letter-spacing:1px;text-transform:uppercase;
}
.anomaly-card .explanation{color:#c8d6e5;font-size:.88rem;line-height:1.7;margin-top:6px;}
hr{border-color:#19283a!important;}
</style>
""", unsafe_allow_html=True)

# ── imports from modules ──────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from modules.file_loader import load_uploaded_file, fix_dtypes

# ────────────────────────────────────────────────────────────────────────
# SESSION DEFAULTS
# ────────────────────────────────────────────────────────────────────────
if "df" in st.session_state and st.session_state.df is not None:
    default_df = st.session_state.df
    default_file = st.session_state.get("file_name", "Active Dataset")
else:
    default_df = None
    default_file = ""

for k, v in {
    "dash_df": default_df, "dash_file": default_file,
    "dash_anomalies": None, "dash_explanations": {},
    "dash_model": "llama3",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.dash_df is None and st.session_state.get("df") is not None:
    st.session_state.dash_df = st.session_state.df
    st.session_state.dash_file = st.session_state.get("file_name", "Active Dataset")

# ────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────
SAMPLE_DIR = pathlib.Path(__file__).parent.parent / "assets" / "sample_data"


def _theme(fig):
    fig.update_layout(
        paper_bgcolor="#070b10", plot_bgcolor="#070b10",
        font_color="#c8d6e5", template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px'>
        <div style='font-size:1.8rem'>📊</div>
        <div style='font-size:1rem;font-weight:800;color:#fff;margin-top:4px'>
            BI ANOMALY DASHBOARD
        </div>
        <div style='font-family:Consolas,monospace;font-size:.55rem;
                    color:#33ccff66;letter-spacing:2px;margin-top:2px'>
            ISOLATION FOREST · IQR · LLM EXPLAIN
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Ollama
    ollama_ok, models = ollama_status()
    if ollama_ok:
        st.markdown('<div class="badge badge-green">● OLLAMA ONLINE</div>',
                    unsafe_allow_html=True)
        st.write("")
        _PREF = ["llama3", "mistral", "gemma", "phi"]
        sorted_m = sorted(models, key=lambda m: next(
            (i for i, p in enumerate(_PREF) if p in m.lower()), 99))
        st.session_state.dash_model = st.selectbox(
            "Model", sorted_m or _PREF, index=0, label_visibility="collapsed")
    else:
        st.markdown('<div class="badge badge-red">✗ OLLAMA OFFLINE</div>',
                    unsafe_allow_html=True)
        st.caption("```\nollama serve\n```")

    st.divider()

    # ── Active Dataset Info (Upload is in main working area) ───────────
    if st.session_state.dash_df is not None:
        fname = st.session_state.get("dash_file", "Active Dataset")
        st.markdown(f'<div style="font-family:Consolas,monospace;font-size:0.75rem;color:#7fffb2;word-break:break-all">📄 <b>{fname}</b></div>', unsafe_allow_html=True)
        if st.button("🔄 Change Dataset", use_container_width=True, key="dash_sidebar_change_file"):
            st.session_state.dash_df = None
            st.session_state.dash_file = ""
            st.session_state.dash_anomalies = None
            st.session_state.dash_explanations = {}
            st.rerun()

    st.divider()

    # ── Anomaly settings ──────────────────────────────────────────────
    st.markdown('<div style="font-family:Consolas;font-size:.6rem;'
                'letter-spacing:2px;text-transform:uppercase;color:#33ccff66;'
                'margin-bottom:6px">⚙️ Detection Settings</div>', unsafe_allow_html=True)

    contamination = st.slider(
        "Expected anomaly %", min_value=1, max_value=20, value=5,
        help="What % of your data you expect to be anomalous (Isolation Forest contamination)"
    ) / 100

    iqr_multiplier = st.slider(
        "IQR sensitivity", min_value=1.5, max_value=4.0, value=2.5, step=0.5,
        help="Higher = only flag extreme outliers. Lower = flag mild ones too."
    )

    run_btn = st.button("🔍  Run Anomaly Detection", use_container_width=True,
                        disabled=st.session_state.dash_df is None)


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px'>📊 BI Anomaly Dashboard</h1>"
    "<p style='color:#2d4a63;font-size:.78rem;margin-top:2px;"
    "font-family:Consolas,monospace;letter-spacing:.5px'>"
    "ISOLATION FOREST + IQR &nbsp;·&nbsp; AI-POWERED BUSINESS EXPLANATIONS &nbsp;·&nbsp; 100% OFFLINE"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Empty state ───────────────────────────────────────────────────────
if st.session_state.dash_df is None:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0b0f16,#0d1e30);
                border:1px solid #19283a;border-radius:14px;
                padding:2.5rem 2rem;text-align:center;margin:1.5rem auto;max-width:720px'>
        <div style='font-size:2.5rem'>🔍</div>
        <h2 style='color:#fff!important;margin:.6rem 0'>Load a Dataset to Begin Anomaly Detection</h2>
        <p style='color:#7a9db8;font-size:.9rem;line-height:1.8'>
            Upload a CSV / Excel file below or pick a sample dataset to detect statistical & multi-dimensional anomalies with <strong>Isolation Forest</strong> and <strong>IQR</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_upload, c_sample = st.columns([1.2, 1], gap="large")
    with c_upload:
        st.markdown("### 📁 Upload File")
        dash_main_up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="dash_main_uploader")
        if dash_main_up:
            new_df, _, _ = load_uploaded_file(dash_main_up)
            if new_df is not None:
                st.session_state.dash_df = new_df
                st.session_state.df = new_df
                st.session_state.dash_file = dash_main_up.name
                st.session_state.file_name = dash_main_up.name
                st.session_state.dash_anomalies = None
                st.session_state.dash_explanations = {}
                st.rerun()

    with c_sample:
        st.markdown("### ⚡ Or Select Sample Dataset")
        samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
        if samples:
            chosen = st.selectbox("Available sample files:", [s.name for s in samples], key="dash_main_sample_select")
            if st.button("🚀 Load Sample Dataset", use_container_width=True, key="dash_load_sample_main"):
                path = SAMPLE_DIR / chosen
                raw_df = pd.read_csv(path)
                clean_df, _ = fix_dtypes(raw_df)
                st.session_state.dash_df = clean_df
                st.session_state.df = clean_df
                st.session_state.dash_file = chosen
                st.session_state.file_name = chosen
                st.session_state.dash_anomalies = None
                st.session_state.dash_explanations = {}
                st.rerun()
        else:
            st.info("No sample datasets found.")

    st.stop()

df = st.session_state.dash_df
num_df = df.select_dtypes(include="number")

# ── Dataset overview strip ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",          f"{len(df):,}")
c2.metric("Columns",       df.shape[1])
c3.metric("Numeric cols",  len(num_df.columns))
c4.metric("Missing",       f"{int(df.isnull().sum().sum()):,}")
st.divider()

# ── Run detection ─────────────────────────────────────────────────────
if run_btn:
    if num_df.empty:
        st.error("No numeric columns found — cannot run anomaly detection.")
    else:
        with st.spinner("🔬 Running Isolation Forest + IQR analysis…"):
            result = detect_anomalies(df, contamination=contamination,
                                      iqr_multiplier=iqr_multiplier)
        st.session_state.dash_anomalies = result
        st.session_state.dash_explanations = {}   # reset explanations for new run

# ── If results exist, show everything ────────────────────────────────
if st.session_state.dash_anomalies is not None:
    res = st.session_state.dash_anomalies
    stats = anomaly_summary_stats(res)

    # ── KPI row ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Anomalies",    stats["total_anomalies"],
              delta=f"{stats['anomaly_pct']:.1f}% of data", delta_color="inverse")
    k2.metric("Columns Affected",   stats["columns_affected"])
    k3.metric("Isolation Forest",   stats["if_count"],  delta="ML method")
    k4.metric("IQR Outliers",
              stats["iqr_count"], delta="Statistical method")

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────────
    tab_overview, tab_deep, tab_explain = st.tabs([
        "📈 Overview Charts",
        "🔎 Deep Dive by Column",
        "🤖 AI Business Explanations",
    ])

    # ═════════════════════ TAB 1 : OVERVIEW ══════════════════════════
    with tab_overview:
        anomaly_df = res["anomaly_rows"]
        normal_df = res["normal_rows"]

        if anomaly_df.empty:
            st.success("✅ No anomalies detected with current settings. "
                       "Try lowering the IQR sensitivity or increasing anomaly %.")
        else:
            st.markdown(f"### 🚨 {len(anomaly_df):,} anomalous rows detected")

            # Scatter: first two numeric cols coloured by anomaly flag
            if len(num_df.columns) >= 2:
                col_x, col_y = num_df.columns[0], num_df.columns[1]
                plot_df = df.copy()
                plot_df["_anomaly"] = "Normal"
                plot_df.loc[anomaly_df.index, "_anomaly"] = "Anomaly"

                fig_sc = px.scatter(
                    plot_df, x=col_x, y=col_y, color="_anomaly",
                    color_discrete_map={
                        "Anomaly": "#ff6b6b", "Normal": "#33ccff"},
                    title=f"Anomaly Map — {col_x} vs {col_y}",
                    template="plotly_dark", opacity=0.75,
                )
                st.plotly_chart(_theme(fig_sc), use_container_width=True)

            # Bar: anomaly count per column
            col_counts = res.get("iqr_by_column", {})
            if col_counts:
                cc_df = pd.DataFrame(
                    [(col, info["count"]) for col, info in col_counts.items()],
                    columns=["Column", "Anomaly Count"]
                ).sort_values("Anomaly Count", ascending=False)

                fig_bar = px.bar(
                    cc_df, x="Column", y="Anomaly Count",
                    title="Anomalies per Column (IQR Method)",
                    color="Anomaly Count", color_continuous_scale="OrRd",
                    template="plotly_dark",
                )
                st.plotly_chart(_theme(fig_bar), use_container_width=True)

            # Preview table
            st.markdown("#### Anomalous Rows (first 30)")
            st.dataframe(anomaly_df.head(
                30), use_container_width=True, height=260)

    # ═════════════════════ TAB 2 : DEEP DIVE ═════════════════════════
    with tab_deep:
        iqr_cols = res.get("iqr_by_column", {})
        if not iqr_cols:
            st.info("No column-level IQR anomalies detected.")
        else:
            selected_col = st.selectbox(
                "Select column to inspect",
                list(iqr_cols.keys()),
            )
            info = iqr_cols[selected_col]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outlier Count",    info["count"])
            m2.metric("Upper Threshold",  f"{info['upper']:.2f}")
            m3.metric("Lower Threshold",  f"{info['lower']:.2f}")
            m4.metric("Column Mean",      f"{df[selected_col].mean():.2f}")

            # Box plot
            fig_box = px.box(
                df, y=selected_col,
                title=f"Distribution of {selected_col}",
                template="plotly_dark", color_discrete_sequence=["#33ccff"],
            )
            # Mark threshold lines
            fig_box.add_hline(y=info["upper"], line_dash="dash",
                              line_color="#ff6b6b", annotation_text="Upper threshold")
            fig_box.add_hline(y=info["lower"], line_dash="dash",
                              line_color="#ff6b6b", annotation_text="Lower threshold")
            st.plotly_chart(_theme(fig_box), use_container_width=True)

            # Histogram
            fig_hist = px.histogram(
                df, x=selected_col, nbins=40,
                title=f"Histogram — {selected_col}",
                template="plotly_dark", color_discrete_sequence=["#33ccff"],
            )
            fig_hist.add_vline(
                x=info["upper"], line_dash="dash", line_color="#ff6b6b")
            fig_hist.add_vline(
                x=info["lower"], line_dash="dash", line_color="#ff6b6b")
            st.plotly_chart(_theme(fig_hist), use_container_width=True)

    # ═════════════════════ TAB 3 : AI EXPLANATIONS ═══════════════════
    with tab_explain:
        iqr_cols = res.get("iqr_by_column", {})
        if not iqr_cols:
            st.info("Run anomaly detection first to get AI explanations.")
        else:
            st.markdown(
                "Click **Explain** on any column to get a plain-language "
                "business explanation of why it's anomalous."
            )
            if not ollama_ok:
                st.warning(
                    "⚠️ Ollama is offline — start it to generate explanations.")

            for col_name, info in iqr_cols.items():
                with st.container():
                    st.markdown(f"""
                    <div class="anomaly-card">
                        <div class="col-name">📌 {col_name}</div>
                    """, unsafe_allow_html=True)

                    # Stats summary
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Outliers",  info["count"])
                    sc2.metric("Max value", f"{info['max_outlier']:.2f}")
                    sc3.metric("Min value", f"{info['min_outlier']:.2f}")

                    # Cached explanation or generate button
                    if col_name in st.session_state.dash_explanations:
                        st.markdown(
                            f'<div class="explanation">'
                            f'{st.session_state.dash_explanations[col_name]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif ollama_ok:
                        if st.button(f"🤖 Explain anomaly in `{col_name}`",
                                     key=f"btn_{col_name}"):
                            desc = df[col_name].describe().round(2).to_dict()
                            context = json.dumps({
                                "column": col_name,
                                "outlier_count": info["count"],
                                "upper_threshold": round(info["upper"], 2),
                                "lower_threshold": round(info["lower"], 2),
                                "max_outlier_value": round(info["max_outlier"], 2),
                                "min_outlier_value": round(info["min_outlier"], 2),
                                "column_stats": {k: round(v, 2) for k, v in desc.items()},
                                "dataset_columns": list(df.columns),
                                "total_rows": len(df),
                            }, indent=2)

                            prompt = f"""You are a senior business intelligence analyst.
You have detected statistical anomalies in a dataset column.

ANOMALY DATA (these are REAL numbers — do not invent any):
{context}

Write a SHORT (3-4 sentence) plain-language business explanation:
1. What the anomaly IS (what values are unusual and by how much)
2. What BUSINESS IMPACT this could indicate (data quality issue, fraud, genuine spike, etc.)
3. What action a data analyst should take next

Write for a non-technical business audience. Be specific using the numbers above.
Do NOT use bullet points. Write in flowing prose."""

                            with st.spinner(f"🤖 Analysing {col_name}…"):
                                explanation = ollama_generate(
                                    prompt,
                                    st.session_state.dash_model
                                )
                            st.session_state.dash_explanations[col_name] = explanation
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)
                    st.write("")

            # ── Explain ALL button ────────────────────────────────────
            if ollama_ok and len(iqr_cols) > 1:
                st.divider()
                if st.button("🤖 Explain ALL anomalies at once", use_container_width=True):
                    all_context = {
                        col: {
                            "outlier_count": info["count"],
                            "upper_threshold": round(info["upper"], 2),
                            "lower_threshold": round(info["lower"], 2),
                            "max_outlier_value": round(info["max_outlier"], 2),
                        }
                        for col, info in iqr_cols.items()
                    }
                    prompt_all = f"""You are a senior business intelligence analyst.
You have found anomalies in multiple columns of a business dataset.

ANOMALY EVIDENCE (real numbers only):
{json.dumps(all_context, indent=2)}

Dataset columns: {list(df.columns)}
Total rows: {len(df)}

Write a BRIEF executive summary (5-7 sentences total) that:
1. Names which columns have the most concerning anomalies
2. Groups related anomalies if they might share a root cause
3. Recommends 2-3 specific next steps for the analyst team

Write for a non-technical executive audience. Use the actual numbers.
Do NOT use bullet points. Write in clear business prose."""

                    with st.spinner("🤖 Generating executive summary…"):
                        summary = ollama_generate(
                            prompt_all,
                            st.session_state.dash_model
                        )

                    st.markdown("### 📋 Executive Summary")
                    st.markdown(
                        f"<div style='background:#0b1a28;border:1px solid #33ccff33;"
                        f"border-radius:10px;padding:18px 22px;line-height:1.8;"
                        f"color:#c8d6e5'>{summary}</div>",
                        unsafe_allow_html=True,
                    )


# ── Download Report ───────────────────────────────────────────────────
if st.session_state.dash_anomalies is not None:
    st.divider()
    st.markdown("### 📥 Download Analyst Report")
    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        with st.spinner("📄 Building PDF report…"):
            try:
                pdf_bytes = generate_report(
                    df=st.session_state.dash_df,
                    result=st.session_state.dash_anomalies,
                    stats=anomaly_summary_stats(
                        st.session_state.dash_anomalies),
                    explanations=st.session_state.dash_explanations,
                    filename=st.session_state.dash_file or "dataset",
                )
                base_name = st.session_state.dash_file.rsplit(
                    ".", 1)[0] if st.session_state.dash_file else "report"
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"anomaly_report_{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")
    with col_info:
        st.markdown(
            "<div style='background:#0b0f16;border:1px solid #19283a;border-radius:8px;"
            "padding:12px 16px;font-size:.82rem;color:#7a9db8;line-height:1.7'>"
            "The PDF report includes:<br>"
            "• Executive summary KPIs<br>"
            "• Dataset overview & descriptive statistics<br>"
            "• Anomaly detection results (Isolation Forest + IQR)<br>"
            "• Column-level anomaly details<br>"
            "• AI business explanations (if generated)<br>"
            "• Analyst recommendations"
            "</div>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#19283a;font-family:Consolas,monospace;"
    "font-size:.6rem;letter-spacing:1.5px;padding:8px 0'>"
    "BI ANOMALY DASHBOARD &nbsp;·&nbsp; ISOLATION FOREST + IQR "
    "&nbsp;·&nbsp; LOCAL LLM &nbsp;·&nbsp; ZERO TELEMETRY"
    "</div>",
    unsafe_allow_html=True,
)
