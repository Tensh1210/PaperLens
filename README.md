<div align="center">

# PaperLens

**Agentic RAG-based ML Paper Search & Comparison Engine**

[![CI](https://github.com/Tensh1210/PaperLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Tensh1210/PaperLens/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-E10098.svg)](https://docs.astral.sh/ruff/)
[![Type check: mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC382D.svg)](https://qdrant.tech/)

An intelligent research assistant that helps ML researchers **find**, **understand**, and **compare** 117k+ academic papers using a custom ReAct agent with 4-layer memory.

[Features](#features) · [Demo](#demo) · [Quick Start](#quick-start) · [Architecture](#architecture) · [API](#api-endpoints)

</div>

## Features

- **Semantic Search**: Find papers by meaning using SPECTER2 embeddings, not just keywords
- **Agentic Reasoning**: ReAct agent that plans, searches, and synthesizes autonomously
- **Auto Comparison**: Compare papers on methodology, contributions, and results
- **4-Layer Memory System**: Semantic (Qdrant), Episodic (SQLite), Working (RAM), Belief (SQLite)
- **Personalization**: Learns user preferences via belief memory with confidence decay
- **Multi-Provider LLM**: Supports Groq, Cerebras, and OpenAI via LiteLLM

## Demo

<div align="center">

![Chat Demo](docs/screenshots/chat-demo.png)

</div>

## Example Usage

### Chat with the Agent

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find papers about vision transformers", "session_id": "demo"}'
```

<details>
<summary>Response</summary>

```json
{
  "response": "Here are some key papers about Vision Transformers:\n\n1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (2020)\n   Introduces ViT, applying pure transformer architecture directly to image patches...\n\n2. **DeiT: Training Data-Efficient Image Transformers** (2020)\n   Proposes knowledge distillation strategies for training ViT without large-scale datasets...",
  "papers": [
    {"title": "An Image is Worth 16x16 Words...", "year": 2020, "score": 0.892},
    {"title": "Training Data-Efficient Image Transformers...", "year": 2020, "score": 0.856}
  ],
  "steps_taken": 3,
  "session_id": "demo"
}
```

</details>

### Semantic Search

```bash
curl "http://localhost:8000/api/search?query=attention+mechanism&limit=5"
```

### Compare Papers

```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"paper_ids": ["1706.03762", "1810.04805"]}'
```

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

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Papers indexed** | 117,592 | CShorten/ML-ArXiv-Papers dataset |
| **Embedding dimensions** | 768 | SPECTER2 (allenai-specter) |
| **Vector index** | HNSW | Cosine similarity, Qdrant |
| **Avg search latency** | ~200ms | Qdrant query + embedding |
| **Agent response time** | 3-8s | 2-4 ReAct iterations via Groq |
| **LLM inference** | Groq | Llama 3.3 70B, ~500 tokens/s |
| **Memory footprint** | ~1.2 GB | SPECTER2 model in RAM |

> Benchmarked on RTX 3050 Laptop + Groq free tier. Latency depends on LLM provider.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Custom ReAct agent** (no LangChain) | Full control over parsing, stop sequences, and tool orchestration. No framework lock-in, easier debugging |
| **SPECTER2** over general embeddings | Trained on scientific citation graphs — understands paper semantics far better than `text-embedding-ada-002` for academic search |
| **4-layer memory** | Mirrors cognitive architecture: semantic (knowledge), episodic (experience), working (context), belief (preferences). Enables personalization without fine-tuning |
| **Groq + LiteLLM** | Groq provides fastest inference for ReAct loops (~500 tok/s). LiteLLM allows hot-swapping providers without code changes |
| **Qdrant** over Pinecone/Weaviate | Self-hosted (no vendor lock-in), rich payload filtering, native HNSW with quantization support |
| **Confidence decay** in belief memory | Prevents stale preferences from dominating. Factor 0.95 per interaction, auto-prune below 0.1 |

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
