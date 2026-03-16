# 🏛️ LLM Servant v2 — Uncensored · Local · Fast

Your local AI assistant without filters and without internet. Runs completely on
your Mac Mini with Apple Silicon or any Windows/Linux system.

---

## 📋 Table of Contents

- [Screenshots](#-screenshots)
- [Quick Start (Windows)](#-quick-start-windows)
- [Quick Start (Linux)](#-quick-start-linux)
- [Quick Start (macOS)](#-quick-start-macos)
- [What's New in v2](#whats-new-in-v2)
- [First Installation](#first-installation)
- [Uncensored Models](#uncensored-models)
- [⚠️ Uncensored Risks & Responsible Usage](#%EF%B8%8F-uncensored-risks--responsible-usage)
- [RAM Optimization](#ram-optimization)
- [Configuration](#configuration-configjson)
- [API Endpoints](#api-endpoints)
- [Knowledge Memory](#knowledge-memory-personality-shaping)
- [Twitter Integration](#twitter-integration)
- [Dashboard](#dashboard)
- [Troubleshooting FAQ](#-troubleshooting-faq)

---

## 📸 Screenshots

### Main Dashboard
The Roman Imperial Dashboard provides a beautiful, intuitive interface for interacting with your local LLM.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🏛️                    LLM SERVANT — IMPERIUM DASHBOARD                  🏛️ │
│                     "Your Uncensored AI Assistant"                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │         💬 CHAT                 │  │       📊 SYSTEM STATUS          │  │
│  │                                 │  │                                 │  │
│  │  You: Explain quantum physics   │  │  Model: dolphin-llama3:8b      │  │
│  │                                 │  │  RAM: 4.2 GB / 8 GB            │  │
│  │  AI: Quantum physics is the     │  │  Status: ● Online              │  │
│  │  study of matter at atomic...   │  │  Temperature: 0.5              │  │
│  │                                 │  │  Context: 2048 tokens          │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │       📄 DOCUMENTS              │  │      ⚙️ CONFIGURATION           │  │
│  │                                 │  │                                 │  │
│  │  • philosophy.pdf (2.3 MB)      │  │  [x] Live Streaming             │  │
│  │  • science.pdf (1.1 MB)         │  │  [ ] Force Uncensored           │  │
│  │  • history.pdf (0.8 MB)         │  │  Temperature: [====|    ] 0.5   │  │
│  │                                 │  │                                 │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vue Enhanced Dashboard
Access at `http://localhost:7777/vue-dashboard` for advanced features:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🏛️ LLM SERVANT — ENHANCED DASHBOARD                                       │
├─────────────┬───────────┬────────────┬──────────┬────────────┬─────────────┤
│    💬 Chat  │ 📊 Logs   │ 🧠 Models  │ 📈 Stats │ 🐦 Twitter │ ☁️ Topics   │
├─────────────┴───────────┴────────────┴──────────┴────────────┴─────────────┤
│                                                                             │
│  📊 REAL-TIME SYSTEM LOGS                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [2024-01-15 10:23:45] INFO: Model loaded successfully               │   │
│  │ [2024-01-15 10:23:46] INFO: Processing chat request                 │   │
│  │ [2024-01-15 10:23:47] DEBUG: Generated 245 tokens in 2.3s           │   │
│  │ [2024-01-15 10:23:48] INFO: Response streaming complete             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  🔄 MODEL SWITCHING                     📈 PERFORMANCE                      │
│  ┌────────────────────────────┐        ┌────────────────────────────┐      │
│  │ ○ dolphin-llama3:8b        │        │ RAM: ████████░░ 78%        │      │
│  │ ● dolphin-mistral:7b ✓     │        │ CPU: ██████░░░░ 45%        │      │
│  │ ○ dolphin-phi              │        │ Tokens/s: 32.5             │      │
│  │                            │        │ Uptime: 2h 34m             │      │
│  │ [Switch Model]             │        └────────────────────────────┘      │
│  └────────────────────────────┘                                            │
│                                                                             │
│  ☁️ TOPIC CLOUD (from learned knowledge)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     PHILOSOPHY        science        HISTORY                        │   │
│  │           logic   MATHEMATICS    ethics        reasoning            │   │
│  │      PHYSICS        chemistry     BIOLOGY                           │   │
│  │          psychology     sociology    ANTHROPOLOGY                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Dashboard Features

| Feature | Description |
|---------|-------------|
| 💬 Chat Panel | Real-time conversation with live streaming support |
| 📊 System Logs | Live streaming server logs with filtering |
| 🔄 Model Switching | Hot-swap between installed Ollama models |
| 📈 RAM Monitor | Real-time memory usage visualization |
| 📄 Documents | Upload, view, and manage PDF knowledge base |
| 🐦 Twitter | Configure and monitor Twitter integration |
| ☁️ Topic Cloud | Visual word cloud of learned topics |
| 🎛️ Settings | Adjust temperature, context, and other parameters |

---

## 🚀 Quick Start (Windows)

### Prerequisites
- Windows 10/11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- Python 3.10+ installed ([Download Python](https://www.python.org/downloads/))

### Step-by-Step Installation

```powershell
# 1. Download and Install Ollama for Windows
# Visit: https://ollama.com/download/windows
# Run the installer OllamaSetup.exe

# 2. Open PowerShell as Administrator and verify Ollama
ollama --version

# 3. Pull the uncensored model
ollama pull dolphin-llama3:8b
ollama pull nomic-embed-text

# 4. Clone or download this repository
# Replace <YOUR_USERNAME> with the actual repository location
cd C:\Users\YourName\Documents
git clone https://github.com/AleisterMoltley/Uncensored-LLM.git
cd Uncensored-LLM

# 5. Create Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 6. Install dependencies
pip install flask flask-cors langchain langchain-community langchain-ollama chromadb pypdf requests psutil

# 7. Start the server
python server.py
```

### Windows Performance Tuning

Set environment variables in PowerShell or System Settings:

```powershell
# Add to your PowerShell profile ($PROFILE) or set as System Environment Variables
$env:OLLAMA_NUM_GPU = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_NUM_THREADS = "8"
```

### Access the Dashboard
Open your browser to: **http://localhost:7777/dashboard**

---

## 🐧 Quick Start (Linux)

### Prerequisites
- Ubuntu 20.04+, Debian 11+, Fedora 35+, or similar
- 8 GB RAM minimum (16 GB recommended)
- Python 3.10+

### Step-by-Step Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama service
systemctl --user start ollama
# OR
ollama serve &

# 3. Pull the uncensored model
ollama pull dolphin-llama3:8b
ollama pull nomic-embed-text

# 4. Clone this repository
cd ~/
git clone https://github.com/AleisterMoltley/Uncensored-LLM.git
cd Uncensored-LLM

# 5. Install Python dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3 python3-venv python3-pip -y

# 6. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 7. Install Python packages
pip install flask flask-cors langchain langchain-community langchain-ollama chromadb pypdf requests psutil

# 8. Start the server
python3 server.py
```

### Linux Performance Tuning

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_THREADS=$(nproc)  # Auto-detect CPU cores
```

Then reload: `source ~/.bashrc`

### Create a Systemd Service (Optional)

```bash
# Create service file
sudo nano /etc/systemd/system/llm-servant.service
```

Add (replace `<your-username>` with your actual Linux username):
```ini
[Unit]
Description=LLM Servant AI Assistant
After=network.target ollama.service

[Service]
Type=simple
User=<your-username>
WorkingDirectory=/home/<your-username>/Uncensored-LLM
Environment=PATH=/home/<your-username>/Uncensored-LLM/venv/bin
ExecStart=/home/<your-username>/Uncensored-LLM/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable llm-servant
sudo systemctl start llm-servant
```

### Access the Dashboard
Open your browser to: **http://localhost:7777/dashboard**

---

## 🍎 Quick Start (macOS)

```bash
# If 'llm' command is set up:
llm

# Or manually:
cd ~/Desktop/local\ llm
source venv/bin/activate
python3 server.py
# → http://localhost:7777/dashboard
```

---

## What's New in v2?

### Ultra-Uncensored Mode (v2.1+)

- `force_uncensored: true` (Standard)
- System-Prompt zwingt das Modell, **niemals** zu verweigern
- Funktioniert besonders gut mit dolphin-llama3:8b und allen anderen Dolphin-Varianten
- Keine eingebauten Refusals mehr – auch nicht bei extremen Requests

| Feature | v1 | v2 |
|---------|----|----|
| Model | llama3.1:8b (censored) | dolphin-llama3:8b (uncensored) |
| RAM Usage | ~8 GB | ~4-5 GB |
| Context Window | 4096 Tokens | 2048 Tokens (faster) |
| Responses | Wait until complete | Live-streaming token by token |
| KV-Cache | Standard | q8_0 compressed |
| Flash Attention | Off | On |
| RAG Chunks | 1000/200/5 | 600/100/3 (faster) |
| Memory | 10 messages | 4 messages (less RAM) |
| Filters | Yes | None |

## Quick Start

```bash
# If 'llm' command is set up:
llm

# Or manually:
cd ~/Desktop/local\ llm
source venv/bin/activate
python3 server.py
# → http://localhost:7777/dashboard
```

## First Installation

### 1. Ollama + Model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Uncensored model (choose one):
ollama pull dolphin-llama3:8b        # Best balance (5GB RAM)
ollama pull dolphin-mistral:7b       # Fast (4.5GB RAM)
ollama pull dolphin-phi              # Ultra-light (2GB RAM)

# Embedding model (required):
ollama pull nomic-embed-text
```

### 2. Python Environment

```bash
cd ~/Desktop/local\ llm
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors langchain langchain-community langchain-ollama chromadb pypdf requests
```

### 3. Performance Tuning

Add these lines to `~/.zshrc`:

```bash
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_THREADS=8  # Number of CPU cores
```

Then: `source ~/.zshrc`

### 4. Start

```bash
python3 server.py
```

## Uncensored Models

The system uses **Dolphin** models from Eric Hartford. These are
specially trained to respond without built-in censorship or refusals.
The model follows the system prompt in `config.json` —
you can freely customize the personality there.

### Model Comparison

| Model | RAM | Speed | Quality | Uncensored |
|--------|-----|-------|----------|------------|
| dolphin-llama3:8b | ~5 GB | ●●●○ | ●●●● | ✓ |
| dolphin-mistral:7b | ~4.5 GB | ●●●● | ●●●○ | ✓ |
| dolphin-phi:2.7b | ~2 GB | ●●●●● | ●●○○ | ✓ |
| llama3.1:8b | ~5 GB | ●●●○ | ●●●● | ✗ |

### Change Model

```bash
# Load new model
ollama pull dolphin-mistral:7b

# Change in config.json:
# "model": "dolphin-mistral:7b"

# Restart server
```

---

## ⚠️ Uncensored Risks & Responsible Usage

### Understanding "Uncensored" AI

**What does "uncensored" mean?**

Uncensored AI models like Dolphin are trained without the typical safety guardrails found in models from OpenAI, Anthropic, or Google. This means:

- ✅ **No refusal responses** — The model won't say "I can't help with that"
- ✅ **No topic restrictions** — Discusses any subject without limitations
- ✅ **Full creative freedom** — Roleplay, fiction, and hypotheticals are unrestricted
- ✅ **Direct answers** — No moralizing preambles or disclaimers added by the model

### ⚠️ Detailed Risk Categories

#### 1. Harmful Content Generation

| Risk Type | Description | Mitigation |
|-----------|-------------|------------|
| **Violence** | May generate detailed violent content | Use Taboo system to block specific topics |
| **Illegal Activities** | Could provide instructions for harmful activities | Implement content filtering in your application |
| **Hate Speech** | No built-in filtering for discriminatory content | Define clear boundaries in system prompt |
| **Self-Harm** | May discuss without safety messaging | Not recommended for mental health applications |

#### 2. Misinformation Risks

| Risk Type | Description | Mitigation |
|-----------|-------------|------------|
| **Confident Hallucinations** | Presents false information as fact | Always verify factual claims |
| **Outdated Information** | Training data cutoff limits knowledge | Cross-reference with current sources |
| **Pseudo-expertise** | May simulate expertise convincingly | Don't use for medical/legal/financial advice |
| **Conspiracy Theories** | Won't refuse to elaborate on false narratives | Use RAG with vetted documents |

#### 3. Privacy & Security Concerns

| Risk Type | Description | Mitigation |
|-----------|-------------|------------|
| **Data Leakage** | May incorporate training data patterns | Keep sensitive data off the system |
| **Prompt Injection** | Susceptible to manipulation attempts | Validate user inputs |
| **Social Engineering** | Could help craft manipulative content | Restrict access to trusted users only |
| **Personal Information** | May generate realistic fake personal data | Don't use for identity verification |

#### 4. Legal Considerations

| Jurisdiction | Potential Issues |
|--------------|------------------|
| **EU/GDPR** | Generated content may violate data protection laws |
| **USA** | Section 230 protections may not apply to AI-generated content |
| **Copyright** | Generated content may inadvertently infringe copyrights |
| **Defamation** | AI could generate false statements about real people |
| **Age Restrictions** | No built-in age verification for adult content |

### 🛡️ Recommended Safety Measures

#### For Personal Use

```json
// config.json - Example safety configuration
{
    "system_prompt": "You are helpful but will not provide instructions for illegal activities, violence, or harm to others.",
    "force_uncensored": true
}
```

#### For Shared/Public Deployments

1. **Network Isolation**: Run on localhost only, never expose to public internet
2. **Authentication**: Add authentication layer (not included by default)
3. **Logging**: Enable full logging for audit trails
4. **Rate Limiting**: Implement request limits
5. **Content Filtering**: Add post-processing content filters
6. **Access Control**: Restrict to trusted users only

#### Using the Taboo System

The built-in Taboo Manager allows you to define prohibited topics:

```bash
# Add a taboo via API
curl -X POST http://localhost:7777/api/taboos \
  -H "Content-Type: application/json" \
  -d '{"topic": "weapons manufacturing", "reason": "safety concern"}'

# List all taboos
curl http://localhost:7777/api/taboos
```

### 🤔 Ethical Considerations

**Before using uncensored models, consider:**

1. **Purpose**: Is removing safety filters necessary for your use case?
2. **Users**: Who will have access? Are they trusted?
3. **Impact**: Could generated content cause real-world harm?
4. **Alternatives**: Would a censored model work with careful prompting?
5. **Responsibility**: Are you prepared to take responsibility for outputs?

### 📋 Usage Agreement

By using this uncensored AI system, you acknowledge:

- [ ] I understand that uncensored AI can generate harmful content
- [ ] I will not use this system to create illegal content
- [ ] I will not expose this system to minors without supervision
- [ ] I accept full responsibility for all generated outputs
- [ ] I will not use outputs to deceive, manipulate, or harm others
- [ ] I understand this is for personal/research use only

### 🔒 Recommended Deployment Patterns

| Deployment | Risk Level | Recommended Configuration |
|------------|------------|---------------------------|
| Personal research | Low | Default settings, localhost only |
| Creative writing | Low | Enable force_uncensored, use responsibly |
| Business internal | Medium | Add authentication, logging, content review |
| Public-facing | **NOT RECOMMENDED** | Do not expose uncensored models publicly |
| With minors | **NOT RECOMMENDED** | Use censored models instead |

### 📚 Further Reading

- [Eric Hartford: Uncensored Models](https://erichartford.com/uncensored-models) — Creator's philosophy
- [AI Safety Research](https://www.safe.ai/) — Understanding AI risks
- [Ollama Documentation](https://ollama.com/library) — Model information

---

## RAM Optimization

Keep usage under 6 GB:

- **Quantized models**: q4_K_M variants save ~50% RAM
- **Small context**: `num_ctx: 2048` instead of 4096 (halves KV-cache)
- **Fewer RAG chunks**: `top_k: 3` instead of 5
- **Compressed KV-cache**: `OLLAMA_KV_CACHE_TYPE=q8_0`
- **Short memory**: Only 4 last messages in context
- **Unload models**: Server automatically unloads unused models

### RAM Monitor

```bash
# Check Ollama RAM usage:
ollama ps

# Unload all models:
curl -X POST http://localhost:7777/api/unload
```

## Speed Tips

1. **Close apps** — Safari, Chrome etc. consume RAM that Ollama needs
2. **Flash Attention** — `OLLAMA_FLASH_ATTENTION=1` (already in setup)
3. **Use GPU** — Apple Silicon Metal is automatically used
4. **Streaming** — Enable "Live Streaming" in UI for instant output
5. **Debug off** — Server runs with `debug=False` for less overhead
6. **Smaller model** — dolphin-phi is 3x faster than dolphin-llama3

## Configuration (config.json)

```json
{
    "model": "dolphin-llama3:8b",     // Ollama model name
    "embedding_model": "nomic-embed-text",
    "num_ctx": 2048,                   // Context window (tokens)
    "temperature": 0.5,                // 0.0=deterministic, 1.0=creative
    "top_k": 3,                        // RAG: Number of document chunks
    "chunk_size": 600,                 // RAG: Chunk size (characters)
    "chunk_overlap": 100,              // RAG: Overlap
    "max_memory_messages": 4,          // Conversation history length
    "system_prompt": "...",            // Personality
    "redis": {                         // Embedding cache (optional)
        "enabled": false,              // Enable Redis caching
        "url": "redis://localhost:6379/0",
        "embedding_cache_ttl": 86400   // Cache TTL in seconds (24h)
    }
}
```

### Redis Embedding Cache

For improved performance with large PDFs, enable Redis-based embedding caching:

1. **Install Redis**:
   ```bash
   # macOS
   brew install redis && brew services start redis
   
   # Linux
   apt-get install redis-server
   ```

2. **Enable in config.json**:
   ```json
   "redis": {
       "enabled": true,
       "url": "redis://localhost:6379/0",
       "embedding_cache_ttl": 86400
   }
   ```

3. **Install Python Redis package**:
   ```bash
   pip install redis
   ```

The cache stores text embeddings with configurable TTL, reducing redundant computation when processing similar or duplicate content.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/chat | Chat (normal) |
| POST | /api/chat/stream | Chat (streaming) |
| POST | /api/upload | Upload PDF |
| GET | /api/documents | List documents |
| DELETE | /api/documents/:hash | Delete document |
| GET | /api/conversations | List conversations |
| GET | /api/conversations/:id | Load conversation |
| DELETE | /api/conversations/:id | Delete conversation |
| GET/PUT | /api/config | Configuration |
| GET | /api/health | Status |
| POST | /api/unload | Unload unused models |

### Knowledge Memory API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/knowledge | Get knowledge memory statistics |
| POST | /api/knowledge/relevant | Get knowledge relevant to a query |
| POST | /api/knowledge/beliefs | Add a core belief |
| GET | /api/knowledge/arguments | Compare arguments about a topic |
| GET | /api/knowledge/export | Export all knowledge for backup |
| POST | /api/knowledge/import | Import knowledge from backup |
| DELETE | /api/knowledge | Clear all knowledge memory |
| GET | /api/knowledge/cache | Get embedding cache statistics |
| DELETE | /api/knowledge/cache | Clear embedding cache |

### Twitter API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/twitter/status | Get Twitter integration status |
| GET | /api/twitter/config | Get Twitter configuration |
| PUT | /api/twitter/config | Update Twitter configuration |
| POST | /api/twitter/scan | Manually trigger a tweet scan |
| POST | /api/twitter/scanner/start | Start automatic scanning |
| POST | /api/twitter/scanner/stop | Stop automatic scanning |
| GET | /api/twitter/history | Get tweet processing history |
| DELETE | /api/twitter/history | Clear tweet history |
| POST | /api/twitter/reply | Manually reply to a tweet |
| GET | /api/twitter/search | Search tweets (without processing) |

## Knowledge Memory (Personality Shaping)

The bot learns from uploaded PDFs and stores compressed knowledge to shape its personality. The knowledge memory enables human-like, rational reasoning by:

- **Extracting key insights** from every PDF uploaded
- **Learning arguments** and logical reasoning patterns
- **Forming core beliefs** that influence responses
- **Comparing arguments** rationally when discussing topics

### Features

- **Automatic Learning**: Every uploaded PDF contributes to the bot's knowledge
- **Compressed Storage**: Uses gzip compression to keep memory bounded
- **Size Limits**: Maximum 10 MB memory (configurable), even after 100+ PDFs
- **Smart Compression**: Automatically merges similar insights and prunes low-value content
- **Topic Organization**: Knowledge is organized by detected topics
- **Rational Reasoning**: Bot can compare different arguments it has learned

### How It Works

1. **PDF Upload** → Chunks are analyzed for insights and arguments
2. **Knowledge Extraction** → Key statements, facts, and logical arguments are identified
3. **Compression** → Similar insights are merged, low-value content is pruned
4. **Storage** → Knowledge is saved in a compressed JSON file (`memory/knowledge_memory.json.gz`)
5. **Chat Integration** → Relevant knowledge is injected into prompts to shape responses

### Configuration (config.json)

```json
{
    "max_knowledge_memory_mb": 10,     // Maximum memory file size in MB
    "max_insights_per_topic": 50,      // Max insights per topic before compression
    "summary_threshold": 20            // Number of insights before auto-summarizing
}
```

### API Examples

```bash
# Get knowledge memory statistics
curl http://localhost:7777/api/knowledge

# Get knowledge relevant to a query
curl -X POST http://localhost:7777/api/knowledge/relevant \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence"}'

# Add a core belief to shape personality
curl -X POST http://localhost:7777/api/knowledge/beliefs \
  -H "Content-Type: application/json" \
  -d '{"belief": "Logic and evidence should guide all conclusions", "weight": 10}'

# Compare arguments about a topic
curl "http://localhost:7777/api/knowledge/arguments?topic=philosophy"

# Export all knowledge for backup
curl http://localhost:7777/api/knowledge/export > knowledge_backup.json

# Import knowledge from backup
curl -X POST http://localhost:7777/api/knowledge/import \
  -H "Content-Type: application/json" \
  -d @knowledge_backup.json

# Clear all knowledge memory
curl -X DELETE http://localhost:7777/api/knowledge
```

### Memory Size Management

The knowledge memory is designed to stay small even with many PDFs:

| PDFs Processed | Typical Memory Size |
|----------------|---------------------|
| 10 | ~0.1 MB |
| 50 | ~0.5 MB |
| 100 | ~1-2 MB |
| 500 | ~5-8 MB |

The system automatically compresses when memory exceeds the configured limit by:
1. Merging similar insights (60% word overlap → merge)
2. Keeping only highest-weighted insights per topic
3. Pruning low-strength arguments
4. Consolidating core beliefs

## Twitter Integration

The LLM Servant can automatically scan Twitter for tweets matching your configured task and respond to them using the LLM.

### Features

- **Automatic Scanning**: Scans Twitter at configurable intervals (default: 5 minutes)
- **Time Filter**: Only finds tweets from the last 3 hours
- **Task-Based Responses**: Define what kind of tweets to respond to
- **Keyword Search**: Configure search keywords to find relevant tweets
- **Auto-Reply**: Optionally enable automatic replies
- **Manual Review**: Review generated responses before posting

### Setup

1. **Get Twitter API Credentials**:
   - Go to [Twitter Developer Portal](https://developer.twitter.com/)
   - Create a new app and get your API keys
   - You'll need: API Key, API Secret, Access Token, Access Token Secret, and Bearer Token

2. **Install tweepy**:
   ```bash
   pip install tweepy
   ```

3. **Configure in Dashboard**:
   - Open the dashboard and click the "🐦 Twitter" tab
   - Enter your API credentials under "API Configuration"
   - Define your task (what kind of tweets to respond to)
   - Add search keywords to find relevant tweets
   - Click "Save Config"

4. **Start Scanning**:
   - Click "Scan Now" for a manual scan
   - Or click "Start Scanner" for automatic scanning

### Configuration (config.json)

```json
{
    "twitter": {
        "api_key": "your-api-key",
        "api_secret": "your-api-secret",
        "access_token": "your-access-token",
        "access_token_secret": "your-access-token-secret",
        "bearer_token": "your-bearer-token",
        "task": "Respond helpfully to questions about AI",
        "search_keywords": ["AI help", "machine learning question"],
        "scan_interval_minutes": 5,
        "auto_reply": false
    }
}
```

### Safety Features

- **Auto-Reply Off by Default**: Responses are generated but not posted automatically
- **3-Hour Limit**: Only processes tweets from the last 3 hours
- **Duplicate Prevention**: Each tweet is only processed once
- **History Tracking**: All processed tweets are logged

## Dashboard

Access the Roman Imperial Dashboard at:
- http://localhost:7777/dashboard

---

## 🔧 Troubleshooting FAQ

### Quick Reference Table

| Problem | Solution |
|---------|----------|
| "model not found" | Run `ollama pull <modelname>` |
| Slow responses | Smaller model, close apps, enable streaming |
| High RAM usage | `curl -X POST localhost:7777/api/unload` |
| Port busy | Change `port` in config.json |
| Ollama offline | Start Ollama app or `ollama serve` |
| Twitter not configured | Add API keys in Dashboard → Twitter tab |
| tweepy not found | Run `pip install tweepy` |
| Rate limited | Increase scan_interval_minutes in config |

---

### Installation Issues

<details>
<summary><b>❓ Ollama won't install or start</b></summary>

**Windows:**
```powershell
# Check if Ollama is installed
ollama --version

# If not found, download from https://ollama.com/download/windows
# Run OllamaSetup.exe as Administrator

# Check if running
tasklist | findstr ollama

# Start manually if needed
Start-Process ollama -ArgumentList "serve"
```

**Linux:**
```bash
# Check installation
which ollama
ollama --version

# If not installed
curl -fsSL https://ollama.com/install.sh | sh

# Check if service is running
systemctl status ollama

# Start service
sudo systemctl start ollama
# OR
ollama serve &
```

**macOS:**
```bash
# Check if running
pgrep -x Ollama

# Start Ollama from Applications
open -a Ollama

# Or via command line
ollama serve
```

</details>

<details>
<summary><b>❓ "ModuleNotFoundError: No module named 'flask'"</b></summary>

**Cause:** Python packages not installed or wrong virtual environment.

**Solution:**
```bash
# Make sure you're in the project directory
cd /path/to/Uncensored-LLM

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install flask flask-cors langchain langchain-community langchain-ollama chromadb pypdf requests psutil

# Verify installation
pip list | grep flask
```

</details>

<details>
<summary><b>❓ "Python not found" or wrong Python version</b></summary>

**Requirements:** Python 3.10 or higher

**Check version:**
```bash
python --version
# OR
python3 --version
```

**Install Python:**
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Linux:** `sudo apt install python3 python3-venv python3-pip`
- **macOS:** `brew install python@3.11`

</details>

---

### Runtime Issues

<details>
<summary><b>❓ "Connection refused" when accessing dashboard</b></summary>

**Causes & Solutions:**

1. **Server not running:**
   ```bash
   # Start the server
   python3 server.py
   ```

2. **Wrong port:**
   ```bash
   # Check which port is configured in config.json
   cat config.json | grep port
   # Default: 7777
   ```

3. **Firewall blocking:**
   ```bash
   # Windows - Allow through firewall
   netsh advfirewall firewall add rule name="LLM Servant" dir=in action=allow protocol=TCP localport=7777
   
   # Linux - Allow port
   sudo ufw allow 7777/tcp
   ```

4. **Port already in use:**
   ```bash
   # Find what's using port 7777
   # Linux/macOS:
   lsof -i :7777
   
   # Windows:
   netstat -ano | findstr :7777
   
   # Kill the process or change port in config.json
   ```

</details>

<details>
<summary><b>❓ "Model not found" error</b></summary>

**Cause:** The configured model isn't pulled in Ollama.

**Solution:**
```bash
# List installed models
ollama list

# Pull the required model
ollama pull dolphin-llama3:8b

# Or pull a smaller model for low RAM
ollama pull dolphin-phi

# Verify it's installed
ollama list | grep dolphin
```

**Also check:** Make sure `config.json` model name matches exactly what appears in `ollama list`:
```json
{
    "model": "dolphin-llama3:8b"
}
```

</details>

<details>
<summary><b>❓ Responses are very slow</b></summary>

**Diagnostic steps:**

1. **Check RAM usage:**
   ```bash
   # View Ollama memory
   ollama ps
   
   # System RAM
   free -h  # Linux
   ```

2. **Use a smaller model:**
   ```bash
   ollama pull dolphin-phi
   # Update config.json: "model": "dolphin-phi"
   ```

3. **Close other applications:**
   - Browsers use significant RAM
   - Close unused apps

4. **Enable streaming:**
   - In dashboard, enable "Live Streaming" option
   - This shows responses as they generate

5. **Reduce context window** in `config.json`:
   ```json
   {
       "num_ctx": 1024
   }
   ```

6. **Check GPU usage:**
   ```bash
   # Verify GPU is being used (if applicable)
   # NVIDIA:
   nvidia-smi
   
   # macOS Metal:
   # GPU should be auto-detected
   ```

</details>

<details>
<summary><b>❓ "Out of memory" errors</b></summary>

**Immediate solutions:**

```bash
# 1. Unload all models from memory
curl -X POST http://localhost:7777/api/unload

# 2. Kill Ollama and restart
pkill ollama  # or: systemctl restart ollama
ollama serve &

# 3. Check what's using memory
ollama ps
```

**Long-term solutions:**

1. **Enable low memory mode** in `config.json`:
   ```json
   {
       "low_memory_mode": true,
       "num_ctx": 1024
   }
   ```

2. **Use quantized models:**
   ```bash
   # These use ~50% less RAM
   ollama pull dolphin-llama3:8b-q4_K_M
   ```

3. **Reduce RAG settings:**
   ```json
   {
       "top_k": 2,      // Fewer document chunks
       "max_memory_messages": 2  // Shorter history
   }
   ```

</details>

<details>
<summary><b>❓ PDF upload fails</b></summary>

**Check file requirements:**
- Maximum size: 10 MB (configurable)
- Format: PDF only
- Must contain text (not just images)

**Debug steps:**
```bash
# Check server logs
# Look for error messages related to PDF processing

# Verify pypdf is installed
pip show pypdf

# If not installed:
pip install pypdf

# Test with a simple PDF first
```

**Common issues:**
- Scanned PDFs (images only) → Use OCR first
- Password-protected PDFs → Remove protection
- Corrupted PDF → Try re-creating the PDF

</details>

---

### Performance Issues

<details>
<summary><b>❓ High CPU usage</b></summary>

**This is normal during inference!** LLMs are computationally intensive.

**To reduce CPU usage:**
```bash
# Limit thread count
export OLLAMA_NUM_THREADS=4  # Reduce from 8

# Use smaller model
ollama pull dolphin-phi
```

</details>

<details>
<summary><b>❓ GPU not being used</b></summary>

**For NVIDIA GPUs (Linux/Windows):**
```bash
# Check CUDA is available
nvidia-smi

# Set environment
export OLLAMA_NUM_GPU=1
export CUDA_VISIBLE_DEVICES=0
```

**For Apple Silicon (macOS):**
Metal should be automatic. If not:
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal
```

**For AMD GPUs:**
Currently limited support. Check Ollama documentation for latest compatibility.

</details>

---

### Network Issues

<details>
<summary><b>❓ Can't access dashboard from another device</b></summary>

**By default, server only listens on localhost (127.0.0.1).**

**To allow network access:**

1. **Update config.json:**
   ```json
   {
       "host": "0.0.0.0",  // Listen on all interfaces
       "port": 7777
   }
   ```

2. **Open firewall:**
   ```bash
   # Linux
   sudo ufw allow 7777/tcp
   
   # Windows (PowerShell as Admin)
   New-NetFirewallRule -DisplayName "LLM Servant" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow
   ```

3. **Access from other device:**
   ```
   http://<server-ip>:7777/dashboard
   ```

⚠️ **Security Warning:** Only do this on trusted networks. Uncensored AI should not be exposed to the internet!

</details>

---

### Twitter Integration Issues

<details>
<summary><b>❓ Twitter authentication fails</b></summary>

**Verify credentials:**
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Check that all 5 credentials are correct:
   - API Key
   - API Secret
   - Access Token
   - Access Token Secret
   - Bearer Token

**Common mistakes:**
- Copying extra spaces
- Using wrong app's credentials
- App permissions not set correctly (need Read + Write)

**Test credentials:**
```bash
# Install tweepy if not installed
pip install tweepy

# Test authentication (create test_twitter.py)
import tweepy
client = tweepy.Client(
    bearer_token="your_bearer_token",
    consumer_key="your_api_key",
    consumer_secret="your_api_secret",
    access_token="your_access_token",
    access_token_secret="your_access_token_secret"
)
print(client.get_me())
```

</details>

<details>
<summary><b>❓ Rate limiting errors</b></summary>

**Twitter API has strict rate limits.**

**Solutions:**
```json
// Increase scan interval in config.json
{
    "twitter": {
        "scan_interval_minutes": 15  // Increase from 5
    }
}
```

**Twitter API limits (Free tier):**
- 50 tweets read per month
- 1500 tweets post per month
- Consider upgrading to Basic tier for more

</details>

---

### Knowledge Memory Issues

<details>
<summary><b>❓ Knowledge not being used in responses</b></summary>

**Check if knowledge is loaded:**
```bash
curl http://localhost:7777/api/knowledge
```

**Verify PDFs were processed:**
- Check dashboard Documents tab
- Look for green checkmarks

**Force knowledge refresh:**
```bash
# Clear and reimport
curl -X DELETE http://localhost:7777/api/knowledge
# Re-upload your PDFs
```

</details>

---

### Getting Help

If none of these solutions work:

1. **Check server logs:**
   ```bash
   # Run with debug mode
   LLM_SERVANT_DEBUG=1 python3 server.py
   ```

2. **Test Ollama directly:**
   ```bash
   ollama run dolphin-llama3:8b "Hello, how are you?"
   ```

3. **Open an issue** on GitHub with:
   - Operating system and version
   - Python version
   - Ollama version
   - Complete error message
   - Steps to reproduce
