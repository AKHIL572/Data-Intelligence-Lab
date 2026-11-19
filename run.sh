#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  run.sh  —  Secure launcher for Private Data Intelligence Lab
#
#  Usage:
#    chmod +x run.sh
#    ./run.sh
#
#  What this script does:
#    1. Forces Ollama to bind on 127.0.0.1 ONLY (not LAN-exposed 0.0.0.0)
#    2. Kills any existing Ollama instance that might be LAN-exposed
#    3. Disables Streamlit telemetry via environment variables
#    4. Starts Streamlit bound strictly to 127.0.0.1:8501
# ══════════════════════════════════════════════════════════════════════
set -e

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║    PRIVATE DATA INTELLIGENCE LAB  —  SECURE RUN  ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── Check Python dependencies ─────────────────────────────────────────
echo "[0/4] Checking Python dependencies..."
python3 -c "
missing = []
for pkg in ['streamlit','ollama','pandas','openpyxl','PyPDF2','plotly']:
    try:
        __import__(pkg.lower().replace('pypdf2','PyPDF2'))
    except ImportError:
        missing.append(pkg)
if missing:
    print('MISSING:', missing)
    exit(1)
else:
    print('All dependencies found.')
" || {
    echo ""
    echo "Install missing packages with:"
    echo "  pip install streamlit ollama pandas openpyxl PyPDF2 plotly"
    echo ""
    exit 1
}

# ── Force Ollama to bind localhost ONLY ───────────────────────────────
export OLLAMA_HOST="127.0.0.1:11434"

# Kill any existing Ollama that may be bound to 0.0.0.0
echo "[1/4] Restarting Ollama on 127.0.0.1:11434 (localhost only)..."
pkill -f "ollama serve" 2>/dev/null && sleep 1 || true

ollama serve &
OLLAMA_PID=$!
sleep 2

# Verify binding
if command -v ss &>/dev/null; then
    BIND=$(ss -tlnp 2>/dev/null | grep ":11434" | awk '{print $4}' | head -1)
elif command -v netstat &>/dev/null; then
    BIND=$(netstat -tlnp 2>/dev/null | grep ":11434" | awk '{print $4}' | head -1)
else
    BIND="(verification unavailable)"
fi
echo "    Ollama binding: ${BIND:-listening}"

# ── Disable Streamlit telemetry ───────────────────────────────────────
echo "[2/4] Disabling Streamlit telemetry..."
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ADDRESS="127.0.0.1"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Copy config.toml into place if not already there
if [ -f ".streamlit/config.toml" ]; then
    mkdir -p ~/.streamlit
    cp ".streamlit/config.toml" ~/.streamlit/config.toml
    echo "    config.toml applied."
fi

# ── Summary ───────────────────────────────────────────────────────────
echo "[3/4] Security summary:"
echo "    ✅  Ollama  → 127.0.0.1:11434 (localhost only)"
echo "    ✅  Streamlit → 127.0.0.1:8501 (localhost only)"
echo "    ✅  Telemetry → DISABLED"
echo "    ✅  CORS     → DISABLED"
echo "    ✅  Your data stays on this machine"
echo ""

# ── Launch Streamlit ──────────────────────────────────────────────────
echo "[4/4] Opening lab at http://127.0.0.1:8501 ..."
echo ""

streamlit run app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection true

# Cleanup Ollama when Streamlit exits
trap "echo 'Shutting down Ollama...'; kill $OLLAMA_PID 2>/dev/null" EXIT
