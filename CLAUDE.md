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
| Embedding | SPECTER2 (allenai/specter2) |
| Vector DB | Qdrant |
| LLM | Groq (Llama 3.1 70B) - free tier |
| Agent Framework | Custom (ReAct pattern) |
| Memory | Multi-store (Qdrant + SQLite + In-memory) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data | HuggingFace: CShorten/ML-ArXiv-Papers |

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

### 🔨 Phase 1: Core Agent System
- [ ] models/memory.py (memory data models)
- [ ] services/llm.py (LLM service with Groq)
- [ ] memory/working.py (working memory)
- [ ] memory/semantic.py (semantic memory wrapper)
- [ ] agent/tools.py (tool definitions)
- [ ] agent/prompts.py (agent prompts)
- [ ] agent/agent.py (main ReAct agent)

### 📋 Phase 2: Full Memory System
- [ ] memory/episodic.py (interaction history)
- [ ] memory/belief.py (user preferences)
- [ ] memory/manager.py (memory orchestration)
- [ ] agent/planner.py (query decomposition)

### 📋 Phase 3: API & Frontend
- [ ] api/main.py (FastAPI endpoints)
- [ ] api/routes/search.py (search routes)
- [ ] api/routes/chat.py (agentic chat)
- [ ] frontend/app.py (Streamlit UI)

### 📋 Phase 4: Data & Polish
- [ ] scripts/index_papers.py (indexing)
- [ ] Tests
- [ ] CI/CD

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
# Required
GROQ_API_KEY=xxx                  # Get from console.groq.com

# Qdrant (defaults work for Docker)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=papers

# Agent Configuration
AGENT_MAX_ITERATIONS=10           # Max ReAct loops
AGENT_TEMPERATURE=0.7             # LLM temperature for reasoning

# Memory Configuration
MEMORY_DB_PATH=data/memory.db     # SQLite path for episodic/belief
MEMORY_WORKING_SIZE=20            # Max items in working memory

# Optional
EMBEDDING_MODEL=allenai/specter2
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
  - Semantic (Qdrant) - paper vectors
  - Episodic (SQLite) - interaction history
  - Working (in-memory) - session context
  - Belief (SQLite) - user preferences

  Tech: Python 3.11, SPECTER2, Qdrant, Groq (Llama 3.1 70B), FastAPI, Streamlit

  Completed:
  - config.py, models/paper.py, clients/data_loader.py
  - services/embedding.py, services/vector_store.py
  - Updated CLAUDE.md, README.md, pyproject.toml, .env.example

  Next: Phase 1 Implementation
  1. src/models/memory.py (memory data models)
  2. src/services/llm.py (Groq LLM service)
  3. src/memory/working.py (working memory)
  4. src/agent/tools.py (tool definitions)
  5. src/agent/prompts.py (agent prompts)
  6. src/agent/agent.py (main ReAct agent)

  New folders to create: src/memory/, src/agent/, data/
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
