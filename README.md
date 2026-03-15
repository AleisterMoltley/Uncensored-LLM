# 🏛️ LLM Servant v2 — Uncensored · Local · Fast

Your local AI assistant without filters and without internet. Runs completely on
your Mac Mini with Apple Silicon.

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
    "system_prompt": "..."             // Personality
}
```

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

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "model not found" | Run `ollama pull <modelname>` |
| Slow | Smaller model, close apps, streaming on |
| High RAM | `curl -X POST localhost:7777/api/unload` |
| Port busy | Change `port` in config.json |
| Ollama offline | Start Ollama app or `ollama serve` |
| Twitter not configured | Add API keys in Dashboard → Twitter tab |
| tweepy not found | Run `pip install tweepy` |
| Rate limited | Increase scan_interval_minutes in config |
