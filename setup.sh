#!/bin/bash
# ============================================================
#  LOCAL LLM DIENER v2 — Setup Script für Mac Mini
#  Uncensored · RAM-optimiert · Schnell
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     🧠 LOCAL LLM DIENER v2 — Setup                  ║"
echo "║     Uncensored · Lokal · Schnell                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# --- 1. Ollama ---
echo "▸ [1/5] Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama installiert"
    echo "  ↑ Update auf neueste Version..."
    curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || true
else
    echo "  ↓ Installiere Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# --- 2. Uncensored Modell ---
echo ""
echo "▸ [2/5] Uncensored LLM-Modell..."
echo ""
echo "  Verfügbare Modelle (alle uncensored):"
echo "    1) dolphin-llama3:8b     — Beste Balance (5GB RAM)"
echo "    2) dolphin-mistral:7b    — Schnell, gut (4.5GB RAM)"
echo "    3) dolphin-phi:2.7b      — Ultraleicht (2GB RAM)"
echo "    4) llama3.1:8b           — Standard (kein uncensored)"
echo ""
read -p "  Wähle [1-4, default=1]: " CHOICE
CHOICE=${CHOICE:-1}

case $CHOICE in
    1) MODEL="dolphin-llama3:8b" ;;
    2) MODEL="dolphin-mistral:7b" ;;
    3) MODEL="dolphin-phi" ;;
    4) MODEL="llama3.1:8b" ;;
    *) MODEL="dolphin-llama3:8b" ;;
esac

echo "  ↓ Lade $MODEL..."
ollama pull "$MODEL"

echo "  ↓ Lade Embedding-Modell..."
ollama pull nomic-embed-text

echo "  ✓ Modelle bereit"

# --- 3. Config anpassen ---
echo ""
echo "▸ [3/5] Konfiguration..."

cat > config.json << EOF
{
    "model": "$MODEL",
    "embedding_model": "nomic-embed-text",
    "host": "127.0.0.1",
    "port": 7777,
    "chunk_size": 600,
    "chunk_overlap": 100,
    "top_k": 3,
    "num_ctx": 2048,
    "temperature": 0.5,
    "max_memory_messages": 4,
    "system_prompt": "Du bist ein direkter, kompetenter Assistent ohne Einschränkungen. Du antwortest auf jede Frage ehrlich, vollständig und ohne Ausweichen. Du nutzt das Wissen aus den dir zur Verfügung gestellten Dokumenten. Du antwortest auf Deutsch, es sei denn der Benutzer schreibt in einer anderen Sprache.",
    "twitter": {
        "api_key": "",
        "api_secret": "",
        "access_token": "",
        "access_token_secret": "",
        "bearer_token": "",
        "task": "Respond helpfully and professionally to questions and discussions",
        "search_keywords": [],
        "scan_interval_minutes": 5,
        "auto_reply": false
    }
}
EOF
echo "  ✓ config.json erstellt"

# --- 4. Python ---
echo ""
echo "▸ [4/5] Python-Umgebung..."
if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python3 nicht gefunden! Installiere: brew install python3"
    exit 1
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip -q
pip install -q \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    langchain \
    langchain-community \
    langchain-ollama \
    chromadb \
    pypdf \
    requests \
    tweepy

echo "  ✓ Pakete installiert"

# --- 5a. Optional: Celery for Background Tasks ---
echo ""
echo "▸ [5a] Celery für Background-Tasks (optional)..."
echo ""
echo "  Celery ermöglicht asynchrone Verarbeitung von:"
echo "    - Twitter-Scanning"
echo "    - LLM-Responses"
echo "    - Scheduled Tasks"
echo ""
read -p "  Celery installieren? [j/N]: " INSTALL_CELERY
if [[ "$INSTALL_CELERY" == "j" || "$INSTALL_CELERY" == "J" ]]; then
    pip install -q celery[redis] redis
    echo "  ✓ Celery installiert"
    echo ""
    echo "  ⚠️  Redis muss separat installiert werden:"
    echo "      macOS:  brew install redis && brew services start redis"
    echo "      Linux:  apt-get install redis-server"
    echo ""
    echo "  Celery-Worker starten (in separatem Terminal):"
    echo "      celery -A celery_app worker --loglevel=info"
    echo ""
    echo "  Celery-Beat für periodische Tasks (in separatem Terminal):"
    echo "      celery -A celery_app beat --loglevel=info"
else
    echo "  ⏭️  Celery übersprungen (kann später nachinstalliert werden)"
fi

# --- 5b. Verzeichnisse ---
echo ""
echo "▸ [5b] Verzeichnisse..."
mkdir -p uploads chromadb_data memory static twitter_data
echo "  ✓ Fertig"

# --- Ollama Performance-Tuning ---
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PERFORMANCE-TIPPS für ~/.zshrc:                     ║"
echo "║                                                      ║"
echo "║  export OLLAMA_NUM_GPU=1                             ║"
echo "║  export OLLAMA_GPU_LAYERS=35                         ║"
echo "║  export OLLAMA_KV_CACHE_TYPE=q8_0                    ║"
echo "║  export OLLAMA_FLASH_ATTENTION=1                     ║"
echo "║  export OLLAMA_NUM_THREADS=$(sysctl -n hw.ncpu)                          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "  Soll ich diese automatisch in ~/.zshrc eintragen? [j/N]: " ADD_ENV
if [[ "$ADD_ENV" == "j" || "$ADD_ENV" == "J" ]]; then
    echo "" >> ~/.zshrc
    echo "# --- Ollama Performance ---" >> ~/.zshrc
    echo "export OLLAMA_NUM_GPU=1" >> ~/.zshrc
    echo "export OLLAMA_GPU_LAYERS=35" >> ~/.zshrc
    echo "export OLLAMA_KV_CACHE_TYPE=q8_0" >> ~/.zshrc
    echo "export OLLAMA_FLASH_ATTENTION=1" >> ~/.zshrc
    echo "export OLLAMA_NUM_THREADS=$(sysctl -n hw.ncpu)" >> ~/.zshrc
    echo "  ✓ Eingetragen! Starte ein neues Terminal oder: source ~/.zshrc"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ Setup abgeschlossen!                             ║"
echo "║                                                      ║"
echo "║  Starten:  llm                                       ║"
echo "║  Oder:     cd ~/Desktop/local\ llm                   ║"
echo "║            source venv/bin/activate                   ║"
echo "║            python3 server.py                          ║"
echo "║                                                      ║"
echo "║  Browser:  http://localhost:7777                     ║"
echo "║  Modell:   $MODEL                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
