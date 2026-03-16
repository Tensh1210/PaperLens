# PaperLens

**Agentic RAG-based ML Paper Search & Comparison Engine**

PaperLens is an intelligent research assistant that helps ML researchers find, understand, and compare academic papers using state-of-the-art Agentic RAG architecture.

## Features

- **Semantic Search**: Find papers by meaning using SPECTER2 embeddings, not just keywords
- **Agentic Reasoning**: ReAct agent that plans, searches, and synthesizes autonomously
- **Auto Comparison**: Compare papers on methodology, contributions, and results
- **4-Layer Memory System**: Semantic (Qdrant), Episodic (SQLite), Working (RAM), Belief (SQLite)
- **Personalization**: Learns user preferences via belief memory with confidence decay
- **Multi-Provider LLM**: Supports Groq, Cerebras, and OpenAI via LiteLLM

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           AGENT LOOP (ReAct)            │
│                                         │
│  THOUGHT → ACTION → OBSERVATION → ...   │
│     │                        │          │
│     └────────────────────────┘          │
│           (max 5 iterations)            │
│                 │                       │
│     ┌───────────┴───────────┐           │
│     │    TOOL REGISTRY      │           │
│     │  6 tools: search,     │           │
│     │  get, compare, relate,│           │
│     │  summarize, recall    │           │
│     └───────────┬───────────┘           │
│                 │                       │
│          MEMORY MANAGER                 │
└─────────────────┼───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌────────┐
│SEMANTIC│  │ EPISODIC │  │ BELIEF │
│ Memory │  │  Memory  │  │ Memory │
│(Qdrant)│  │ (SQLite) │  │(SQLite)│
│117k    │  │ history  │  │ prefs  │
│papers  │  │ & recall │  │ decay  │
└────────┘  └──────────┘  └────────┘
```

### Memory System

| Memory Type | Purpose | Storage | Key Feature |
|-------------|---------|---------|-------------|
| **Semantic** | Paper knowledge & embeddings | Qdrant (HNSW) | 117k ML papers, cosine similarity |
| **Episodic** | Past searches & interactions | SQLite (async) | Query history, liked papers |
| **Working** | Current session context | In-memory | Max 20 messages, agent steps |
| **Belief** | User preferences & patterns | SQLite | Confidence decay (0.95), auto-pruning |

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Embedding | SPECTER2 (768d) | Trained on scientific citation graphs |
| Vector DB | Qdrant | Fast HNSW, rich filtering, self-hosted |
| LLM | Groq (Llama 3.3 70B) | Fastest inference for ReAct loops |
| LLM Gateway | LiteLLM | Provider-agnostic, easy switching |
| Agent | Custom ReAct | Full control, no framework lock-in |
| Backend | FastAPI | Async, Pydantic, auto OpenAPI docs |
| Frontend | Streamlit | Pure Python chat UI |
| Data | HuggingFace | CShorten/ML-ArXiv-Papers dataset |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Qdrant)
- API Key: [Groq](https://console.groq.com/) or [Cerebras](https://cloud.cerebras.ai/) (free tier)

### Installation

```bash
# Clone repository
git clone https://github.com/Tensh1210/paperlens.git
cd paperlens

# Copy environment variables
cp .env.example .env
# Edit .env and add your API keys

# Start Qdrant
docker compose up -d qdrant

# Install dependencies
pip install -e ".[dev]"

# Index papers (first time only, ~30 min)
make index

# Start API + Frontend
make dev
# API: http://localhost:8000 | UI: http://localhost:8501
```

## Project Structure

```
PaperLens/
├── src/
│   ├── config.py           # Settings (pydantic-settings)
│   ├── utils.py            # Async/sync bridge utilities
│   ├── models/             # Pydantic data models
│   ├── clients/            # HuggingFace data loader
│   ├── services/           # Embedding, VectorStore, LLM
│   ├── memory/             # 4-layer memory system
│   ├── agent/              # ReAct agent, tools, planner, prompts
│   └── api/                # FastAPI routes (chat, search)
├── frontend/app.py         # Streamlit chat UI
├── scripts/                # Indexing scripts
├── tests/                  # pytest suite
├── docker/Dockerfile       # Multi-stage build
└── docker-compose.yml
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Agentic chat (ReAct loop) |
| `/api/chat/stream` | POST | Streaming chat (SSE) |
| `/api/search` | GET/POST | Direct semantic search |
| `/api/compare` | POST | Compare multiple papers |
| `/api/papers/{id}/related` | GET | Find related papers |
| `/health` | GET | Health check + memory stats |

## Configuration

Key environment variables (see `.env.example` for full list):

```bash
# LLM (choose one provider)
LLM_PROVIDER=groq                    # groq, cerebras, or openai
LLM_MODEL=llama-3.3-70b-versatile    # Model name
GROQ_API_KEY=xxx                     # Required for Groq

# Infrastructure
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Agent
AGENT_MAX_ITERATIONS=5               # Max ReAct loops
AGENT_TEMPERATURE=0.7
```

## Roadmap

### Evaluation Pipeline (Priority)
- [ ] Benchmark dataset: 50 queries with ground truth relevant papers
- [ ] Retrieval metrics: Precision@K, Recall@K, MRR, NDCG
- [ ] Generation evaluation: Faithfulness & relevance scoring (LLM-as-Judge)
- [ ] A/B comparison framework for measuring improvements

### Retrieval Improvements
- [ ] Hybrid search: combine semantic (SPECTER2) + keyword (BM25) for better recall
- [ ] Re-ranking layer with cross-encoder after initial retrieval
- [ ] Query expansion / HyDE (Hypothetical Document Embeddings)
- [ ] Multi-vector retrieval: separate title & abstract embeddings

### Agent Improvements
- [ ] Streaming ReAct: show reasoning steps in real-time to UI
- [ ] Fully async agent (remove sync/async bridge)
- [ ] Concurrent tool execution for independent actions
- [ ] Better prompt engineering for reliable FINAL_ANSWER generation

### Production Readiness
- [ ] Authentication (JWT/OAuth)
- [ ] API rate limiting
- [ ] Observability: tracing LLM calls (LangSmith/Arize)
- [ ] Monitoring: latency, token usage, cost per query (Prometheus/Grafana)
- [ ] PostgreSQL migration for concurrent multi-user support

## CI/CD

5-job GitHub Actions pipeline on push to main:

| Job | Purpose |
|-----|---------|
| **Lint** | Ruff check |
| **Type Check** | mypy strict mode |
| **Test** | pytest + Qdrant service container |
| **Build** | Python wheel package |
| **Docker** | Container build (main branch only) |

## References

- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [SPECTER2](https://huggingface.co/allenai/specter2)
- [RAG Original Paper](https://arxiv.org/abs/2005.11401)

## License

MIT License

## Author

- **Tenshi** - [@Tensh1210](https://github.com/Tensh1210)
