import streamlit as st
from modules.prompts import prompt_pdf, extract_pdf_evidence
from modules.llm import ollama_status, ollama_generate
import os
import pathlib
import sys
import json
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(page_title="PDF Analyser", page_icon="📄", layout="wide")

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
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li{
    color:#c8d6e5;line-height:1.7;}
[data-testid="stChatInput"] textarea{
    background:#0b0f16!important;border:1px solid #19283a!important;
    color:#c8d6e5!important;font-family:Consolas,monospace!important;
}
.stButton>button{
    background:linear-gradient(135deg,#0d2e52,#111d2e);
    color:#33ccff;border:1px solid #33ccff44;border-radius:6px;
    font-family:Consolas,monospace;font-size:.78rem;padding:6px 16px;transition:all .2s;
}
.badge{
    display:inline-block;padding:3px 12px;border-radius:20px;
    font-family:Consolas,monospace;font-size:.68rem;border:1px solid;margin:2px;
}
.badge-green{color:#7fffb2;border-color:#7fffb244;background:#7fffb20d;}
.badge-red  {color:#ff6b6b;border-color:#ff6b6b44;background:#ff6b6b0d;}
.badge-blue {color:#33ccff;border-color:#33ccff44;background:#33ccff0d;}
hr{border-color:#19283a!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for k, v in {
    "pdf_text": "", "pdf_evidence": {}, "pdf_file": "",
    "pdf_messages": [], "pdf_summary_done": False, "pdf_model": "llama3",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px'>
        <div style='font-size:1.8rem'>📄</div>
        <div style='font-size:1rem;font-weight:800;color:#fff;margin-top:4px'>PDF ANALYSER</div>
        <div style='font-family:Consolas,monospace;font-size:.55rem;
                    color:#33ccff66;letter-spacing:2px;margin-top:2px'>
            GROUNDED · LOCAL · PRIVATE
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
        st.session_state.pdf_model = st.selectbox(
            "Model", sorted_m or _PREF, index=0, label_visibility="collapsed")
    else:
        st.markdown('<div class="badge badge-red">✗ OLLAMA OFFLINE</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Active Document Info (Upload is in main working area) ──────────
    if st.session_state.pdf_text:
        fname = st.session_state.get("pdf_file", "Active PDF Document")
        st.markdown(f'<div style="font-family:Consolas,monospace;font-size:0.75rem;color:#7fffb2;word-break:break-all">📄 <b>{fname}</b></div>', unsafe_allow_html=True)
        if st.button("🔄 Change PDF", use_container_width=True, key="pdf_sidebar_change_file"):
            st.session_state.pdf_text = ""
            st.session_state.pdf_evidence = {}
            st.session_state.pdf_file = ""
            st.session_state.pdf_messages = []
            st.session_state.pdf_summary_done = False
            st.rerun()

    if st.session_state.pdf_text:
        ev = st.session_state.pdf_evidence
        st.divider()
        st.markdown('<div style="font-family:Consolas;font-size:.6rem;letter-spacing:2px;'
                    'text-transform:uppercase;color:#33ccff66;margin-bottom:6px">'
                    '📊 Document Stats</div>', unsafe_allow_html=True)
        st.metric("Words",     f"{ev.get('word_count', 0):,}")
        st.metric("Sentences", f"{ev.get('sentence_count', 0):,}")
        st.metric("Chars",     f"{ev.get('char_count', 0):,}")

# ── Main ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:2px'>📄 PDF Analyser</h1>"
    "<p style='color:#2d4a63;font-size:.78rem;margin-top:2px;"
    "font-family:Consolas,monospace;letter-spacing:.5px'>"
    "GROUNDED ANALYSIS · NO HALLUCINATED STATS · LOCAL LLM"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

if not st.session_state.pdf_text:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0b0f16,#0d1e30);
                border:1px solid #19283a;border-radius:14px;
                padding:2.5rem 2rem;text-align:center;margin:1.5rem auto;max-width:720px'>
        <div style='font-size:2.5rem'>📄</div>
        <h2 style='color:#fff!important;margin:.6rem 0'>Upload a PDF to Begin Analysis</h2>
        <p style='color:#7a9db8;font-size:.9rem;line-height:1.8'>
            The analyser extracts key statistics, figures, and facts <em>before</em> sending to
            the LLM — ensuring high grounding with zero hallucinated figures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_center = st.container()
    with c_center:
        main_pdf_up = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_main_uploader")
        if main_pdf_up:
            import PyPDF2
            reader = PyPDF2.PdfReader(main_pdf_up)
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(pages).strip()

            if not text:
                st.error("⚠️ Scanned PDF — no extractable text found.")
            else:
                if len(text) > 15_000:
                    st.info(f"Large PDF ({len(text):,} chars) — using first 15,000 chars.")
                    text = text[:15_000]

                evidence = extract_pdf_evidence(text)
                st.session_state.pdf_text = text
                st.session_state.pdf_evidence = evidence
                st.session_state.pdf_file = main_pdf_up.name
                st.session_state.file_name = main_pdf_up.name
                st.session_state.pdf_messages = []
                st.session_state.pdf_summary_done = False
                st.rerun()

    st.stop()

# ── Auto summary ──────────────────────────────────────────────────────
if not st.session_state.pdf_summary_done:
    if not ollama_ok:
        st.warning("⚠️ Ollama offline. Run `ollama serve`.")
    else:
        model = st.session_state.pdf_model
        summary_prompt = (
            "Summarise this document using ONLY the text and facts provided. "
            "Do not add or invent any information.\n\n"
            "## 📄 What is this document?\n"
            "## 🔑 Key Points\n"
            "## 🔢 Notable Data or Figures\n"
            "## 💡 Summary\n\n"
            f"DOCUMENT FACTS:\n{json.dumps(st.session_state.pdf_evidence, indent=2)}\n\n"
            f"TEXT:\n{st.session_state.pdf_text}"
        )
        with st.spinner(f"📄 Summarising with {model}…"):
            summary = ollama_generate(summary_prompt, model)

        st.session_state.pdf_messages.append(
            {"role": "assistant", "content": summary})
        st.session_state.pdf_summary_done = True
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────
st.markdown('<div style="font-family:Consolas;font-size:.6rem;letter-spacing:2px;'
            'text-transform:uppercase;color:#33ccff66;margin-bottom:6px">'
            '💬 Ask about your PDF</div>', unsafe_allow_html=True)

for msg in st.session_state.pdf_messages:
    av = "👤" if msg["role"] == "user" else "📄"
    with st.chat_message(msg["role"], avatar=av):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Ask anything about the document…"
    if ollama_ok else "Ollama offline — run: ollama serve"
)
if user_input and len(user_input) > 2000:
    st.warning("Message too long (max 2000 chars).")
    user_input = None

if user_input:
    if not ollama_ok:
        st.error("❌ Ollama offline.")
        st.stop()

    st.session_state.pdf_messages.append(
        {"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    model = st.session_state.pdf_model
    # BUG 5 FIX: always pass pdf_evidence for grounding
    prompt = prompt_pdf(user_input, st.session_state.pdf_text,
                        st.session_state.pdf_evidence)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner(f"Thinking with {model}…"):
            answer = ollama_generate(prompt, model)
        st.markdown(answer)

    st.session_state.pdf_messages.append(
        {"role": "assistant", "content": answer})

st.divider()
st.markdown(
    "<div style='text-align:center;color:#19283a;font-family:Consolas,monospace;"
    "font-size:.6rem;letter-spacing:1.5px;padding:8px 0'>"
    "PDF ANALYSER &nbsp;·&nbsp; GROUNDED ANALYSIS &nbsp;·&nbsp; LOCAL LLM &nbsp;·&nbsp; ZERO TELEMETRY"
    "</div>",
    unsafe_allow_html=True,
)
