#!/bin/bash
# ============================================================
#  LOCAL LLM DIENER v2 — Cross-Platform Setup Script
#  Supports: macOS, Linux, Windows (WSL/Git Bash)
#  Uncensored · RAM-optimiert · Schnell
# ============================================================
set -e

# ============================================================
#  OS Detection
# ============================================================
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            OS="macos"
            ;;
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                OS="wsl"
            else
                OS="linux"
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            OS="windows"
            ;;
        *)
            OS="unknown"
            ;;
    esac
    echo "$OS"
}

# Get CPU core count cross-platform
get_cpu_count() {
    case "$OS" in
        macos)
            sysctl -n hw.ncpu 2>/dev/null || echo "4"
            ;;
        linux|wsl)
            nproc 2>/dev/null || grep -c processor /proc/cpuinfo 2>/dev/null || echo "4"
            ;;
        windows)
            echo "${NUMBER_OF_PROCESSORS:-4}"
            ;;
        *)
            echo "4"
            ;;
    esac
}

# Get shell config file path
get_shell_config() {
    case "$OS" in
        macos)
            if [[ "$SHELL" == *"zsh"* ]]; then
                echo "$HOME/.zshrc"
            else
                echo "$HOME/.bash_profile"
            fi
            ;;
        linux|wsl)
            if [[ "$SHELL" == *"zsh"* ]]; then
                echo "$HOME/.zshrc"
            else
                echo "$HOME/.bashrc"
            fi
            ;;
        windows)
            echo "$HOME/.bashrc"
            ;;
        *)
            echo "$HOME/.bashrc"
            ;;
    esac
}

# Helper function for right-padding text in box display
# Usage: pad_right "text" total_width
# Returns padding spaces needed to fill total_width
pad_right() {
    local text="$1"
    local width="$2"
    local len=${#text}
    local padding=$((width - len))
    # Ensure non-negative padding
    if [[ $padding -lt 0 ]]; then
        padding=0
    fi
    printf '%*s' "$padding" ''
}

# Install Ollama based on OS
install_ollama() {
    echo "▸ Ollama Installation/Update..."
    
    case "$OS" in
        macos|linux|wsl)
            if command -v ollama &> /dev/null; then
                echo "  ✓ Ollama installiert"
                echo "  ↑ Update auf neueste Version..."
                curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || true
            else
                echo "  ↓ Installiere Ollama..."
                curl -fsSL https://ollama.com/install.sh | sh
            fi
            ;;
        windows)
            if command -v ollama &> /dev/null; then
                echo "  ✓ Ollama installiert"
                echo "  ⚠️  Windows: Bitte manuell updaten via https://ollama.com/download/windows"
            else
                echo "  ✗ Ollama nicht gefunden!"
                echo "  ⚠️  Windows: Bitte manuell installieren:"
                echo "      https://ollama.com/download/windows"
                echo ""
                read -p "  Enter drücken wenn Ollama installiert ist..." _
            fi
            ;;
        *)
            echo "  ✗ Unbekanntes OS - Ollama Installation nicht möglich"
            echo "  Bitte manuell installieren: https://ollama.com"
            exit 1
            ;;
    esac
}

# Update Ollama based on OS
update_ollama() {
    echo "▸ Ollama Update..."
    
    case "$OS" in
        macos|linux|wsl)
            if command -v ollama &> /dev/null; then
                echo "  ↑ Update Ollama auf neueste Version..."
                curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || {
                    echo "  ⚠️  Ollama Update fehlgeschlagen"
                    return 1
                }
                echo "  ✓ Ollama aktualisiert"
            else
                echo "  ✗ Ollama nicht installiert - führe erst 'Install' aus"
                return 1
            fi
            ;;
        windows)
            if command -v ollama &> /dev/null; then
                echo "  ⚠️  Windows: Bitte Ollama manuell updaten via:"
                echo "      https://ollama.com/download/windows"
            else
                echo "  ✗ Ollama nicht installiert"
            fi
            ;;
        *)
            echo "  ✗ Unbekanntes OS"
            return 1
            ;;
    esac
}

# Check Python installation and provide OS-specific install instructions
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "  ✗ Python3 nicht gefunden!"
        case "$OS" in
            macos)
                echo "  Installiere mit: brew install python3"
                echo "  Oder: https://www.python.org/downloads/macos/"
                ;;
            linux)
                echo "  Installiere mit: sudo apt install python3 python3-venv python3-pip"
                echo "  Oder (Fedora): sudo dnf install python3 python3-pip"
                ;;
            wsl)
                echo "  Installiere mit: sudo apt install python3 python3-venv python3-pip"
                ;;
            windows)
                echo "  Installiere von: https://www.python.org/downloads/windows/"
                echo "  Oder mit: winget install Python.Python.3.11"
                ;;
        esac
        exit 1
    fi
    echo "  ✓ Python3 gefunden: $(python3 --version)"
}

# Show header
show_header() {
    local os_padding
    os_padding=$(pad_right "$OS" 36)
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     🧠 LOCAL LLM DIENER v2 — Setup                  ║"
    echo "║     Uncensored · Lokal · Schnell                     ║"
    echo "║     OS: ${OS}${os_padding}║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================
#  Installation Function
# ============================================================
do_install() {
    # --- 1. Ollama ---
    echo "▸ [1/5] Ollama..."
    install_ollama

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

    if [ ! -f "config.json" ]; then
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
    else
        # Check if model in config.json differs from selected model
        EXISTING_MODEL=$(grep -o '"model": *"[^"]*"' config.json | head -1 | cut -d'"' -f4)
        if [ "$EXISTING_MODEL" != "$MODEL" ]; then
            echo "  ⚠️  config.json existiert mit Modell: $EXISTING_MODEL"
            echo "      Gewähltes Modell: $MODEL"
            read -p "  Modell in config.json aktualisieren? [j/N]: " UPDATE_MODEL
            if [[ "$UPDATE_MODEL" == "j" || "$UPDATE_MODEL" == "J" ]]; then
                # Use sed to update the model - cross-platform compatible
                if [[ "$OS" == "macos" ]]; then
                    sed -i '' "s|\"model\": *\"[^\"]*\"|\"model\": \"$MODEL\"|" config.json
                else
                    sed -i "s|\"model\": *\"[^\"]*\"|\"model\": \"$MODEL\"|" config.json
                fi
                echo "  ✓ Modell in config.json aktualisiert"
            else
                echo "  ⏭️  config.json unverändert gelassen"
            fi
        else
            echo "  ✓ config.json existiert bereits (Modell: $EXISTING_MODEL)"
        fi
    fi

    # --- 4. Python ---
    echo ""
    echo "▸ [4/5] Python-Umgebung..."
    check_python

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "  ✓ Virtual Environment erstellt"
    fi
    
    # Activate venv based on OS
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
    else
        source venv/bin/activate
    fi

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
        tweepy \
        psutil \
        pydantic

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
        case "$OS" in
            macos)
                echo "      brew install redis && brew services start redis"
                ;;
            linux)
                echo "      sudo apt install redis-server && sudo systemctl start redis"
                ;;
            wsl)
                echo "      sudo apt install redis-server && sudo service redis-server start"
                ;;
            windows)
                echo "      Installiere Redis via https://github.com/microsoftarchive/redis/releases"
                echo "      Oder nutze Docker: docker run -d -p 6379:6379 redis"
                ;;
        esac
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
    setup_performance_env

    local model_padding
    model_padding=$(pad_right "$MODEL" 22)
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  ✓ Setup abgeschlossen!                             ║"
    echo "║                                                      ║"
    echo "║  Starten:                                            ║"
    case "$OS" in
        windows)
    echo "║      venv\\Scripts\\activate                          ║"
            ;;
        *)
    echo "║      source venv/bin/activate                        ║"
            ;;
    esac
    echo "║      python3 server.py                               ║"
    echo "║                                                      ║"
    echo "║  Browser:  http://localhost:7777                     ║"
    echo "║  Modell:   ${MODEL}${model_padding}║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================
#  Update Function
# ============================================================
do_update() {
    local os_padding
    os_padding=$(pad_right "$OS" 36)
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     🔄 LOCAL LLM DIENER v2 — Update                 ║"
    echo "║     OS: ${OS}${os_padding}║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    # --- 1. Update Ollama ---
    echo "▸ [1/3] Ollama Update..."
    update_ollama
    echo ""

    # --- 2. Update Ollama Models ---
    echo "▸ [2/3] Ollama Modelle aktualisieren..."
    if command -v ollama &> /dev/null; then
        # Get currently configured model from config.json if it exists
        if [ -f "config.json" ]; then
            CURRENT_MODEL=$(grep -o '"model": *"[^"]*"' config.json | head -1 | cut -d'"' -f4)
            if [ -n "$CURRENT_MODEL" ]; then
                echo "  ↑ Update $CURRENT_MODEL..."
                ollama pull "$CURRENT_MODEL" || echo "  ⚠️  Model Update fehlgeschlagen"
            fi
        fi
        echo "  ↑ Update nomic-embed-text..."
        ollama pull nomic-embed-text || echo "  ⚠️  Embedding Model Update fehlgeschlagen"
        echo "  ✓ Modelle aktualisiert"
    else
        echo "  ⚠️  Ollama nicht verfügbar"
    fi
    echo ""

    # --- 3. Update Python Packages ---
    echo "▸ [3/3] Python-Pakete aktualisieren..."
    
    if [ ! -d "venv" ]; then
        echo "  ✗ Keine venv gefunden - führe erst 'Install' aus"
        return 1
    fi
    
    # Activate venv based on OS
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
    else
        source venv/bin/activate
    fi
    
    echo "  ↑ Update pip..."
    pip install --upgrade pip -q
    
    echo "  ↑ Update alle Pakete..."
    pip install --upgrade -q \
        flask \
        flask-cors \
        langchain \
        langchain-community \
        langchain-ollama \
        chromadb \
        pypdf \
        requests \
        tweepy \
        psutil \
        pydantic
    
    # Check if celery is installed and update it too
    if pip show celery &>/dev/null; then
        echo "  ↑ Update Celery..."
        pip install --upgrade -q celery[redis] redis
    fi
    
    echo "  ✓ Python-Pakete aktualisiert"

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  ✓ Update abgeschlossen!                            ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================
#  Performance Environment Setup
# ============================================================
setup_performance_env() {
    local CPU_COUNT
    CPU_COUNT=$(get_cpu_count)
    local SHELL_CONFIG
    SHELL_CONFIG=$(get_shell_config)
    local SHELL_NAME
    SHELL_NAME=$(basename "$SHELL_CONFIG")
    local shell_padding
    shell_padding=$(pad_right "$SHELL_NAME" 23)
    local cpu_padding
    cpu_padding=$(pad_right "$CPU_COUNT" 25)
    
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  PERFORMANCE-TIPPS für ~/${SHELL_NAME}:${shell_padding}║"
    echo "║                                                      ║"
    echo "║  export OLLAMA_NUM_GPU=1                             ║"
    echo "║  export OLLAMA_GPU_LAYERS=35                         ║"
    echo "║  export OLLAMA_KV_CACHE_TYPE=q8_0                    ║"
    echo "║  export OLLAMA_FLASH_ATTENTION=1                     ║"
    echo "║  export OLLAMA_NUM_THREADS=${CPU_COUNT}${cpu_padding}║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
    read -p "  Soll ich diese automatisch in ~/$SHELL_NAME eintragen? [j/N]: " ADD_ENV
    if [[ "$ADD_ENV" == "j" || "$ADD_ENV" == "J" ]]; then
        # Check if already added
        if grep -q "# --- Ollama Performance ---" "$SHELL_CONFIG" 2>/dev/null; then
            echo "  ⚠️  Ollama Performance bereits in $SHELL_CONFIG vorhanden"
        else
            echo "" >> "$SHELL_CONFIG"
            echo "# --- Ollama Performance ---" >> "$SHELL_CONFIG"
            echo "export OLLAMA_NUM_GPU=1" >> "$SHELL_CONFIG"
            echo "export OLLAMA_GPU_LAYERS=35" >> "$SHELL_CONFIG"
            echo "export OLLAMA_KV_CACHE_TYPE=q8_0" >> "$SHELL_CONFIG"
            echo "export OLLAMA_FLASH_ATTENTION=1" >> "$SHELL_CONFIG"
            echo "export OLLAMA_NUM_THREADS=$CPU_COUNT" >> "$SHELL_CONFIG"
            echo "  ✓ Eingetragen! Starte ein neues Terminal oder: source $SHELL_CONFIG"
        fi
    fi
}

# ============================================================
#  Main Menu
# ============================================================
show_menu() {
    local os_padding
    os_padding=$(pad_right "$OS" 36)
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     🧠 LOCAL LLM DIENER v2                          ║"
    echo "║     Uncensored · Lokal · Schnell                     ║"
    echo "║     OS: ${OS}${os_padding}║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
    echo "  Optionen:"
    echo "    1) Install  — Vollständige Installation"
    echo "    2) Update   — Pip-Pakete & Ollama aktualisieren"
    echo "    3) Exit     — Beenden"
    echo ""
    read -p "  Wähle [1-3, default=1]: " MENU_CHOICE
    MENU_CHOICE=${MENU_CHOICE:-1}
}

# ============================================================
#  Main
# ============================================================
main() {
    # Detect OS first
    OS=$(detect_os)
    
    # Check for command line arguments
    if [[ "$1" == "--install" || "$1" == "-i" ]]; then
        show_header
        do_install
        exit 0
    elif [[ "$1" == "--update" || "$1" == "-u" ]]; then
        do_update
        exit 0
    elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --install, -i    Run full installation"
        echo "  --update, -u     Update Ollama and pip packages"
        echo "  --help, -h       Show this help message"
        echo ""
        echo "Without options, an interactive menu is shown."
        echo ""
        echo "Supported Operating Systems:"
        echo "  - macOS (Intel & Apple Silicon)"
        echo "  - Linux (Debian/Ubuntu, Fedora, etc.)"
        echo "  - Windows (via WSL or Git Bash)"
        echo ""
        exit 0
    fi
    
    # Interactive menu
    show_menu
    
    case $MENU_CHOICE in
        1)
            show_header
            do_install
            ;;
        2)
            do_update
            ;;
        3)
            echo "  Auf Wiedersehen! 👋"
            exit 0
            ;;
        *)
            show_header
            do_install
            ;;
    esac
}

# Run main
main "$@"
