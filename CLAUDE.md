# CLAUDE.md - PaperLens

## Project Overview

PaperLens is an **Agentic RAG (Retrieval-Augmented Generation)** system for searching, analyzing, and comparing ML papers from ArXiv. It uses a ReAct agent with 4-layer memory, SPECTER2 embeddings, Qdrant vector DB, and Groq LLM.

## Tech Stack

- **Python 3.11+**, Pydantic v2, pydantic-settings
- **Embedding**: SPECTER2 (`sentence-transformers/allenai-specter`), 768 dimensions
- **Vector DB**: Qdrant (cosine distance, `query_points` API)
- **LLM**: Groq (`llama-3.3-70b-versatile`) via LiteLLM
- **API**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Memory**: SQLite (aiosqlite) for episodic/belief, in-memory for working
- **Data**: HuggingFace `CShorten/ML-ArXiv-Papers`

## Key Architecture

- **ReAct Agent** (`src/agent/agent.py`): THOUGHT → ACTION → OBSERVATION loop, max 5 iterations
- **6 Tools** (`src/agent/tools.py`): search_papers, get_paper, get_related, compare_papers, summarize_paper, recall_memory
- **4 Memory Types**: Semantic (Qdrant), Episodic (SQLite), Working (RAM), Belief (SQLite)
- **Memory Manager** (`src/memory/manager.py`): orchestrates all memory, handles personalization (+0.05 category boost, +0.02 topic boost)
- **Query Planner** (`src/agent/planner.py`): pattern matching + LLM decomposition for complex queries

## Project Structure

```
src/
  config.py              # Settings from env vars (pydantic-settings, singleton via lru_cache)
  models/paper.py        # Paper, PaperSearchResult, PaperComparison, IndexStats
  models/memory.py       # MemoryType(StrEnum), MemoryItem, EpisodicMemory, BeliefMemory, WorkingMemoryState
  services/embedding.py  # EmbeddingService (SPECTER2, lazy load, singleton)
  services/vector_store.py # VectorStore (Qdrant, MD5 hash for point IDs, payload indexes)
  services/llm.py        # LLMService (LiteLLM, tenacity retry: 5 attempts, exponential 5-60s)
  clients/data_loader.py # HuggingFaceDataLoader
  memory/semantic.py     # SemanticMemory (wraps VectorStore + EmbeddingService)
  memory/episodic.py     # EpisodicMemoryStore (SQLite, async)
  memory/belief.py       # BeliefMemoryStore (SQLite, confidence decay 0.95)
  memory/working.py      # WorkingMemory (in-memory dict, max 20 messages)
  memory/manager.py      # MemoryManager (orchestrator, personalized search, consolidation)
  agent/agent.py         # PaperLensAgent (ReAct loop)
  agent/tools.py         # 6 tools + ToolRegistry
  agent/planner.py       # QueryPlanner (intent detection, year filters)
  agent/prompts.py       # SYSTEM_PROMPT, REACT_PROMPT, templates
  api/main.py            # FastAPI app (lifespan, CORS, routers)
  api/routes/chat.py     # Chat endpoints (/api/chat, /api/chat/stream, sessions)
  api/routes/search.py   # Search endpoints (/api/search, /api/compare, /api/papers/{id}/related)
frontend/app.py          # Streamlit chat UI
scripts/index_papers.py  # Paper indexing script
docker/Dockerfile        # Multi-stage build, non-root user
```

## Development Commands

```bash
make dev-install    # Install with dev deps + pre-commit
make dev            # Run API + Frontend
make api            # API only (uvicorn, port 8000, --reload)
make frontend       # Streamlit only (port 8501)
make test           # pytest with coverage
make test-fast      # pytest -x --no-cov
make lint           # ruff check + mypy
make format         # ruff format + fix
make up / down      # Docker compose
make index          # Index papers into Qdrant
```

## Code Style & Conventions

- **Linter**: Ruff (rules: E, W, F, I, B, C4, UP; ignore: E501, B008, B905)
- **Type checker**: mypy strict (`disallow_untyped_defs`, `warn_return_any`)
- **Target**: Python 3.11 (`target-version = "py311"`)
- **Line length**: 100
- **Enums**: Use `StrEnum` (not `str, Enum`) - enforced by ruff UP042
- **Per-file ignores**: `scripts/*` and `frontend/*` allow E402 (imports after sys.path)
- **Import sorting**: isort via ruff, first-party = `paperlens`
- **All services use singleton pattern** via module-level `get_*()` functions
- **Lazy loading** for expensive resources (models, DB connections)
- **Async memory stores** (episodic, belief) with sync wrappers where needed

## CI Pipeline (GitHub Actions)

5 jobs on push to main/develop and PRs to main:
1. **Lint**: `ruff check .`
2. **Type Check**: `mypy src --ignore-missing-imports`
3. **Test**: pytest with Qdrant service container (wait loop for health)
4. **Build**: `python -m build` (wheel)
5. **Docker Build**: only on push to main, uses `docker/Dockerfile` (not root)

## Important Patterns

- **Agent parses LLM output with regex**: looks for `THOUGHT:`, `ACTION:`, `ACTION_INPUT:`, `FINAL_ANSWER:` in response text
- **Stop sequence**: Agent calls LLM with `stop=["OBSERVATION:"]` so it stops before injecting tool results
- **Point IDs in Qdrant**: `int(hashlib.md5(arxiv_id.encode()).hexdigest()[:15], 16)` - deterministic integer from arxiv_id
- **Belief confidence**: starts 0.5, reinforced via `confidence += strength * (1 - confidence)`, decayed via `confidence *= 0.95`, pruned at < 0.1
- **Async/sync bridge** in agent: `_run_async()` uses ThreadPoolExecutor when event loop is running
- **Retry on LLM**: tenacity, 5 attempts, exponential backoff 5-60s for ConnectionError/TimeoutError/RateLimitError

## Testing

- Fixtures in `tests/conftest.py`: `setup_test_env` (autouse), mocks for LLM/embedding/vector_store, sample papers, memory fixtures with `tmp_path`
- `asyncio_mode = "auto"` in pytest config
- Test env: `GROQ_API_KEY=test-api-key`, `MEMORY_DB_PATH=data/test_memory.db`

## Common Gotchas

- Dockerfile is at `docker/Dockerfile`, not root - CI must specify `file: docker/Dockerfile`
- `StrEnum` required instead of `(str, Enum)` for Python 3.11+ with ruff UP042
- SPECTER2 model loads ~30s first time; cached after via singleton and Docker volume
- SQLite memory DB path (`data/memory.db`) - directory created automatically by stores
- CORS is `allow_origins=["*"]` - needs restriction for production
