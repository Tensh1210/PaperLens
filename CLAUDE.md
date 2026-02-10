# CLAUDE.md - Instructions for Claude Code

## Project Overview

**PaperLens** is an **Agentic RAG-based** ML Paper Search & Comparison Engine.

### Goal
Build a tool that helps researchers:
1. Search for ML papers semantically (not just keywords)
2. Auto-compare papers (methodology, contributions, timeline)
3. Understand paper relationships and evolution
4. **Learn from user interactions** via agentic memory

### Architecture Paradigm: Agentic RAG

Unlike traditional RAG (retrieve → generate), PaperLens uses an **Agentic RAG** approach:
- **Single Agent** with multiple tools for reasoning and action
- **Full Agentic Memory** (Semantic + Episodic + Working + Belief)
- **Custom Framework** built on Groq for fast inference
- **ReAct Loop**: Plan → Act → Observe → Reflect → Repeat

### Tech Stack
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Embedding | SPECTER (sentence-transformers/allenai-specter) |
| Vector DB | Qdrant |
| LLM | Groq (Llama 3.3 70B) or OpenAI (gpt-4o-mini) |
| Agent Framework | Custom (ReAct pattern) |
| Memory | Multi-store (Qdrant + SQLite + In-memory) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data | HuggingFace: CShorten/ML-ArXiv-Papers |

> **Note**: Originally planned to use SPECTER2 (`allenai/specter2`) but switched to SPECTER (`sentence-transformers/allenai-specter`) due to PEFT compatibility issues.

---

## Project Structure

```
PaperLens/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Settings (pydantic-settings)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── paper.py              # Paper data model
│   │   └── memory.py             # Memory data models
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # HuggingFace dataset loader
│   │   └── arxiv_client.py       # ArXiv API (optional)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py          # SPECTER2 embeddings
│   │   ├── vector_store.py       # Qdrant operations
│   │   └── llm.py                # LLM service (Groq via litellm)
│   │
│   ├── memory/                   # 🆕 Agentic Memory System
│   │   ├── __init__.py
│   │   ├── manager.py            # Memory orchestration
│   │   ├── semantic.py           # Semantic memory (vector store wrapper)
│   │   ├── episodic.py           # Episodic memory (interaction history)
│   │   ├── working.py            # Working memory (current session)
│   │   └── belief.py             # Belief memory (user preferences)
│   │
│   ├── agent/                    # 🆕 Agentic RAG System
│   │   ├── __init__.py
│   │   ├── agent.py              # Main ReAct agent
│   │   ├── tools.py              # Tool definitions
│   │   ├── planner.py            # Query decomposition
│   │   └── prompts.py            # Agent prompt templates
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI app
│       └── routes/
│           ├── __init__.py
│           ├── search.py         # Search endpoints
│           └── chat.py           # 🆕 Agentic chat endpoint
│
├── frontend/
│   └── app.py                    # Streamlit UI
│
├── scripts/
│   ├── index_papers.py           # Index papers to Qdrant
│   └── download_data.py          # Download dataset
│
├── data/                         # 🆕 Local data storage
│   └── memory.db                 # SQLite for episodic/belief memory
│
├── tests/
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Agentic RAG Architecture

### High-Level Flow

```
User Query: "Compare recent transformer papers for NLP"
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGENT LOOP (ReAct)                     │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  PLAN   │ → │   ACT   │ → │ OBSERVE │ → │ REFLECT │  │
│  │         │    │         │    │         │    │         │  │
│  │ Decompose│    │Use Tools│    │ Analyze │    │ Decide  │  │
│  │ query   │    │         │    │ results │    │ next    │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                           │                                 │
│                    ┌──────┴──────┐                          │
│                    │   MEMORY    │                          │
│                    │   MANAGER   │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   SEMANTIC   │   │   EPISODIC   │   │    BELIEF    │
│    MEMORY    │   │    MEMORY    │   │    MEMORY    │
│              │   │              │   │              │
│ Paper vectors│   │ Past queries │   │ User prefs   │
│ (Qdrant)     │   │ (SQLite)     │   │ (SQLite)     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Agent Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_papers` | Semantic search in paper database | query, limit, year_from, year_to, categories |
| `get_paper` | Retrieve full paper details | arxiv_id |
| `compare_papers` | Generate comparison between papers | paper_ids, aspects |
| `summarize_paper` | Summarize a single paper | arxiv_id |
| `filter_results` | Filter search results | results, criteria |
| `get_related` | Find related papers | arxiv_id, limit |
| `recall_memory` | Retrieve from episodic memory | query |
| `update_belief` | Update user preferences | preference_type, value |

### Memory System

#### 1. Semantic Memory (Qdrant)
- Paper embeddings via SPECTER2
- Fast similarity search
- Metadata filtering (year, category)

#### 2. Episodic Memory (SQLite)
- Past search queries and results
- User feedback (liked/disliked papers)
- Session history
- Enables: "Show me papers like that one I searched for last week"

#### 3. Working Memory (In-memory)
- Current conversation context
- Retrieved papers in session
- Intermediate reasoning steps
- Cleared on session end

#### 4. Belief Memory (SQLite)
- User preferences (favorite categories, authors)
- Reading level preferences
- Learned patterns from interactions
- Persistent across sessions

---

## Current Progress

### ✅ Completed
- [x] Project structure defined
- [x] README.md
- [x] pyproject.toml (dependencies)
- [x] docker-compose.yml (Qdrant + API + Frontend)
- [x] Makefile (common commands)
- [x] config.py (settings management)
- [x] models/paper.py (data model)
- [x] clients/data_loader.py (HuggingFace loader)
- [x] services/embedding.py (SPECTER2)
- [x] services/vector_store.py (Qdrant)

### ✅ Phase 1: Core Agent System
- [x] models/memory.py (memory data models)
- [x] services/llm.py (LLM service with Groq)
- [x] memory/working.py (working memory)
- [x] memory/semantic.py (semantic memory wrapper)
- [x] agent/tools.py (tool definitions)
- [x] agent/prompts.py (agent prompts)
- [x] agent/agent.py (main ReAct agent)

### ✅ Phase 2: Full Memory System
- [x] memory/episodic.py (interaction history)
- [x] memory/belief.py (user preferences)
- [x] memory/manager.py (memory orchestration)
- [x] agent/planner.py (query decomposition)

### ✅ Phase 3: API & Frontend
- [x] api/main.py (FastAPI endpoints)
- [x] api/routes/search.py (search routes)
- [x] api/routes/chat.py (agentic chat)
- [x] frontend/app.py (Streamlit UI)

### ✅ Phase 4: Data & Polish
- [x] scripts/index_papers.py (indexing)
- [x] Tests (test_models.py, test_memory.py, test_agent.py, test_api.py, test_services.py)
- [x] CI/CD (GitHub Actions workflow, pre-commit hooks)

---

## 🔧 Known Issues & TODO (as of 2026-02-10)

### Current Status: ✅ System Functional (with LLM rate limits)

The system is fully functional:
- ✅ 1000 papers indexed into Qdrant
- ✅ Search functionality working
- ✅ Agent ReAct loop working
- ✅ Memory system (episodic, belief, working) working
- ✅ Docker services running (Qdrant, API, Frontend)
- ⚠️ Groq free tier has strict rate limits (12k tokens/min) - may need to upgrade or use OpenAI

### LLM Provider Options

**Option 1: Groq (Free but rate-limited)**
- Free tier: 12,000 tokens/minute
- Upgrade to Dev Tier for higher limits: https://console.groq.com/settings/billing

**Option 2: OpenAI (Paid, no strict limits)**
Set in `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
```

### Completed Fixes (2026-02-10 session)
1. ✅ Fixed Qdrant API: `search()` → `query_points()` in `vector_store.py`
2. ✅ Fixed test cases: Updated `test_services.py` for new embedding model and data_loader behavior
3. ✅ Fixed Groq model: `llama-3.1-70b-versatile` → `llama-3.3-70b-versatile` (old model decommissioned)
4. ✅ Fixed docker-compose.yml: Removed health checks causing startup issues
5. ✅ Fixed docker-compose.yml: Added LLM environment variables to frontend service
6. ✅ Added rate limit retry logic to `llm.py` with exponential backoff
7. ✅ Reduced agent max iterations from 10 to 5 to reduce token usage
8. ✅ All 103 tests passing

### Previous Fixes (2026-01-24 session)
1. ✅ Changed embedding model from `allenai/specter2` to `sentence-transformers/allenai-specter` (PEFT compatibility)
2. ✅ Updated `.env` file with new embedding model
3. ✅ Fixed `data_loader.py` to generate IDs for papers without arxiv_id
4. ✅ Fixed `vector_store.py` to use hash-based integer IDs (Qdrant requires int or UUID)
5. ✅ Fixed `vector_store.py` `get_collection_info()` - changed `vectors_count` to `indexed_vectors_count`
6. ✅ Fixed `docker/Dockerfile` to copy README.md (required by pyproject.toml)
7. ✅ Fixed `docker-compose.yml` - removed obsolete `version` attribute

### How to Run
```bash
# 1. Start Docker services (1 terminal)
docker compose up

# 2. Index papers (if needed, with venv activated)
venv\Scripts\activate
python scripts/index_papers.py --limit 1000 --recreate

# Access:
# - Frontend: http://localhost:8501
# - API docs: http://localhost:8000/docs
# - Qdrant dashboard: http://localhost:6333/dashboard
```

---

## Key Design Decisions

### 1. Why Agentic RAG?
- **Dynamic Retrieval**: Agent decides when/what to retrieve
- **Multi-step Reasoning**: Complex queries need iterative refinement
- **Self-correction**: Agent can retry with different strategies
- **Memory Integration**: Learn from past interactions

### 2. Why Single Agent (vs Multi-Agent)?
- Simpler to implement and debug
- Sufficient for paper search domain
- Lower latency (fewer LLM calls)
- Can evolve to multi-agent later if needed

### 3. Why Custom Framework (vs LangChain)?
- Full control over agent behavior
- Lighter dependencies
- Optimized for Groq's fast inference
- Easier to understand and maintain

### 4. Why Full Agentic Memory?
- **Episodic**: "Find papers like the one I searched last week"
- **Belief**: Personalized results based on preferences
- **Working**: Maintain context in complex conversations
- **Semantic**: Core paper knowledge (already implemented)

### 5. Why SPECTER2?
- Designed specifically for scientific papers
- Trained on citation prediction
- Better than general-purpose embeddings for academic text

### 6. Why Qdrant?
- Production-ready vector DB
- Good filtering support (by year, category)
- Docker-first, easy to deploy

### 7. Why Groq?
- Free tier available
- Very fast inference (~10x faster than OpenAI)
- Llama 3.1 70B is high quality
- Critical for responsive agent loops

---

## ReAct Agent Pattern

The agent uses **ReAct** (Reasoning + Acting) pattern:

```python
# Pseudo-code for agent loop
while not done:
    # 1. THINK - Reason about current state
    thought = llm.think(query, memory, history)

    # 2. ACT - Choose and execute tool
    action = llm.choose_action(thought, available_tools)
    result = execute_tool(action)

    # 3. OBSERVE - Process tool result
    observation = process_result(result)

    # 4. REFLECT - Decide if done or continue
    if is_satisfactory(observation):
        done = True
        response = synthesize_response(history)
    else:
        history.append((thought, action, observation))
```

### Example Agent Trace

```
Query: "Compare recent transformer papers for NLP"

[THOUGHT] User wants to compare transformer papers. "Recent" suggests
         filtering by year (2023+). Need to search, then compare.

[ACTION] search_papers(query="transformer NLP", year_from=2023, limit=10)

[OBSERVATION] Found 10 papers: BERT improvements, GPT variants,
              efficient transformers...

[THOUGHT] Good results. Should compare top 5 most relevant ones
          on methodology and contributions.

[ACTION] compare_papers(paper_ids=[...], aspects=["methodology", "contributions"])

[OBSERVATION] Comparison generated covering architecture differences,
              training approaches, benchmark results.

[THOUGHT] Comparison complete. Can now respond to user.

[RESPONSE] Here's a comparison of recent transformer papers for NLP...
```

---

## API Endpoints

### Core Endpoints

```
POST /api/chat                    # 🆕 Agentic chat (main interface)
  - body: { message: str, session_id: str? }
  - returns: { response: str, papers: [...], session_id: str }

POST /api/search                  # Direct search (bypasses agent)
  - body: { query: str, limit: int, year_from: int? }
  - returns: { papers: [...], total: int }

POST /api/compare                 # Direct compare (bypasses agent)
  - body: { paper_ids: list[str], aspects: list[str]? }
  - returns: { comparison: str, papers: [...] }

GET /api/papers/{arxiv_id}
  - returns: { paper: {...} }

GET /api/stats
  - returns: { total_papers: int, categories: {...} }

GET /health
  - returns: { status: "ok", memory: {...} }
```

### Memory Endpoints

```
GET /api/memory/history           # Get search history
  - returns: { queries: [...] }

POST /api/memory/feedback         # Submit paper feedback
  - body: { arxiv_id: str, liked: bool }

GET /api/memory/preferences       # Get user preferences
  - returns: { preferences: {...} }
```

---

## Environment Variables

```bash
# LLM Provider (choose one)
# Option A: Groq (free tier, rate-limited)
GROQ_API_KEY=gsk_xxx              # Get from console.groq.com
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Option B: OpenAI (paid, no strict limits)
# OPENAI_API_KEY=sk-xxx           # Get from platform.openai.com
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini

# Qdrant (defaults work for Docker)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=papers

# Agent Configuration
AGENT_MAX_ITERATIONS=5            # Max ReAct loops (reduced to save tokens)
AGENT_TEMPERATURE=0.7             # LLM temperature for reasoning

# Memory Configuration
MEMORY_DB_PATH=data/memory.db     # SQLite path for episodic/belief
MEMORY_WORKING_SIZE=20            # Max items in working memory

# Optional
EMBEDDING_MODEL=sentence-transformers/allenai-specter
LOG_LEVEL=INFO
DEBUG=false
```

---

## Implementation Tasks for Claude Code

### Task 1: Memory Data Models
```
File: src/models/memory.py
- MemoryItem: Base memory item with timestamp
- EpisodicMemory: Past query, results, feedback
- BeliefMemory: User preference with confidence
- WorkingMemoryState: Current session state
```

### Task 2: LLM Service
```
File: src/services/llm.py
- Use litellm for Groq integration
- chat_completion() for agent reasoning
- Streaming support for responses
- Retry logic with tenacity
```

### Task 3: Working Memory
```
File: src/memory/working.py
- Store current session context
- Track retrieved papers
- Maintain conversation history
- Clear on session end
```

### Task 4: Agent Tools
```
File: src/agent/tools.py
- Tool base class with schema
- search_papers tool
- get_paper tool
- compare_papers tool
- summarize_paper tool
- All tools return structured results
```

### Task 5: Agent Prompts
```
File: src/agent/prompts.py
- SYSTEM_PROMPT: Agent persona and capabilities
- REACT_PROMPT: ReAct reasoning template
- TOOL_PROMPT: Tool selection instructions
- COMPARE_PROMPT: Paper comparison template
- SUMMARY_PROMPT: Paper summarization template
```

### Task 6: Main Agent
```
File: src/agent/agent.py
- PaperLensAgent class
- ReAct loop implementation
- Tool execution
- Memory integration
- Response synthesis
```

### Task 7: Episodic Memory
```
File: src/memory/episodic.py
- SQLite storage for history
- Store queries with timestamps
- Store paper interactions
- Retrieval by recency/relevance
```

### Task 8: Belief Memory
```
File: src/memory/belief.py
- SQLite storage for preferences
- Track favorite categories
- Track favorite authors
- Confidence scoring
```

### Task 9: Memory Manager
```
File: src/memory/manager.py
- Orchestrate all memory types
- Unified retrieval interface
- Memory consolidation
- Context building for agent
```

---

## Notes for Claude Code

1. **Always check existing code first** before creating new files
2. **Follow existing patterns** - see config.py, paper.py for style
3. **Use type hints** - project uses Python 3.11+
4. **Use structlog** for logging
5. **Use pydantic** for data validation
6. **Keep functions focused** - single responsibility
7. **Add docstrings** - Google style
8. **Run tests** after changes: `make test`
9. **Agent responses should be fast** - minimize LLM calls
10. **Memory operations should be async** - don't block agent loop

---

## PaperLens Project Status

  Architecture: Agentic RAG with ReAct pattern (Single Agent + Full Memory)

  Memory System:
  - Semantic (Qdrant) - paper vectors ✅
  - Episodic (SQLite) - interaction history ✅
  - Working (in-memory) - session context ✅
  - Belief (SQLite) - user preferences ✅
  - Manager - unified orchestration ✅

  Tech: Python 3.11, SPECTER, Qdrant, Groq/OpenAI, FastAPI, Streamlit

  All Phases Complete:
  - Phase 1: Core Agent System ✅
  - Phase 2: Full Memory System ✅
  - Phase 3: API & Frontend ✅
  - Phase 4: Data & Polish ✅ (tests, CI/CD)

  **Current Status (2026-02-10)**:
  - ✅ 1000 papers indexed into Qdrant
  - ✅ Search functionality working
  - ✅ Agent and memory system functional
  - ✅ All 103 tests passing
  - ⚠️ Groq free tier rate limits may require upgrade or switch to OpenAI
## Resources

### Core
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [SPECTER2 Model](https://huggingface.co/allenai/specter2)
- [LiteLLM Docs](https://docs.litellm.ai/)
- [Groq Console](https://console.groq.com/)

### Agentic RAG
- [Agentic RAG Survey (arXiv)](https://arxiv.org/abs/2501.09136)
- [NVIDIA: Traditional vs Agentic RAG](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/)
- [Weaviate: What is Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)

### Memory Systems
- [Memory in the Age of AI Agents (arXiv)](https://arxiv.org/abs/2512.13564)
- [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)
- [Hindsight Memory Architecture](https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision)

### Frameworks
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Dataset: ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
