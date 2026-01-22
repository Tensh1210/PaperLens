# PaperLens

**Agentic RAG-based ML Paper Search & Comparison Engine**

PaperLens is an intelligent research assistant that helps ML researchers find, understand, and compare academic papers using state-of-the-art Agentic RAG architecture.

## Features

- **Semantic Search**: Find papers by meaning, not just keywords
- **Agentic Reasoning**: AI agent that plans, searches, and synthesizes autonomously
- **Auto Comparison**: Automatically compare papers on methodology, contributions, results
- **Full Memory System**: Learns from your interactions (episodic, semantic, belief memory)
- **Timeline View**: Understand research evolution over time

## Architecture

PaperLens uses **Agentic RAG** - a modern approach that combines autonomous AI agents with retrieval-augmented generation:

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           AGENT LOOP (ReAct)            │
│                                         │
│   PLAN → ACT → OBSERVE → REFLECT        │
│     │                        │          │
│     └────────────────────────┘          │
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
└────────┘  └──────────┘  └────────┘
```

### Memory System

| Memory Type | Purpose | Storage |
|-------------|---------|---------|
| **Semantic** | Paper knowledge & embeddings | Qdrant |
| **Episodic** | Past searches & interactions | SQLite |
| **Working** | Current session context | In-memory |
| **Belief** | User preferences & patterns | SQLite |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding | SPECTER2 (allenai/specter2) |
| Vector DB | Qdrant |
| LLM | Groq (Llama 3.1 70B) |
| Agent | Custom ReAct implementation |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data | HuggingFace (CShorten/ML-ArXiv-Papers) |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Groq API Key (free at [console.groq.com](https://console.groq.com/))

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/paperlens.git
cd paperlens

# Copy environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start services
make up

# Index papers (first time only)
make index

# Open UI at http://localhost:8501
```

### Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
make test

# Start development servers
make dev
```

## Project Structure

```
PaperLens/
├── src/
│   ├── config.py           # Settings
│   ├── models/             # Data models
│   ├── clients/            # External API clients
│   ├── services/           # Core services (embedding, vector store, LLM)
│   ├── memory/             # Agentic memory system
│   ├── agent/              # ReAct agent implementation
│   └── api/                # FastAPI backend
├── frontend/               # Streamlit UI
├── scripts/                # Utility scripts
├── tests/                  # Test suite
└── docker-compose.yml
```

## Example Usage

```
User: "Compare recent transformer papers for NLP"

Agent: [THOUGHT] User wants transformer papers for NLP. "Recent" = 2023+.
       [ACTION] search_papers(query="transformer NLP", year_from=2023)
       [OBSERVATION] Found 10 relevant papers...
       [ACTION] compare_papers(paper_ids=[...], aspects=["methodology"])
       [RESPONSE] Here's a comparison of recent transformer papers...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Agentic chat interface |
| `/api/search` | POST | Direct semantic search |
| `/api/compare` | POST | Compare specific papers |
| `/api/papers/{id}` | GET | Get paper details |
| `/api/memory/history` | GET | Get search history |
| `/health` | GET | Health check |

## Configuration

Key environment variables:

```bash
GROQ_API_KEY=xxx              # Required
QDRANT_HOST=localhost         # Vector DB host
AGENT_MAX_ITERATIONS=10       # Max reasoning steps
MEMORY_DB_PATH=data/memory.db # SQLite path
```

See `.env.example` for full configuration options.

## References

- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [SPECTER2 Model](https://huggingface.co/allenai/specter2)

## License

MIT License

## Author

- **Tenshi** - [@Tensh1210](https://github.com/Tensh1210)
