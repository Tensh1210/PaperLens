# TONG QUAN DU AN PAPERLENS

## ML Paper Search & Comparison Engine

**Version**: 0.1.0
**License**: MIT
**Python**: >= 3.11
**Tac gia**: Tenshi

---

## MUC LUC

1. [Tong Quan Du An](#1-tong-quan-du-an)
2. [Kien Truc Chi Tiet](#2-kien-truc-chi-tiet)
3. [Chi Tiet Trien Khai](#3-chi-tiet-trien-khai)
4. [Luong Xu Ly (Workflows)](#4-luong-xu-ly-workflows)
5. [Van De & Giai Phap](#5-van-de--giai-phap)
6. [Hieu Suat & Toi Uu Hoa](#6-hieu-suat--toi-uu-hoa)
7. [Chien Luoc Testing](#7-chien-luoc-testing)
8. [Trien Khai (Deployment)](#8-trien-khai-deployment)
9. [Huong Phat Trien Tuong Lai](#9-huong-phat-trien-tuong-lai)
10. [Bai Hoc Kinh Nghiem](#10-bai-hoc-kinh-nghiem)
11. [Phu Luc](#11-phu-luc)

---

# 1. TONG QUAN DU AN

## 1.1. Gioi Thieu

PaperLens la mot **Agentic RAG (Retrieval-Augmented Generation) system** duoc thiet ke de tim kiem, phan tich va so sanh cac bai bao khoa hoc ve Machine Learning tu ArXiv. He thong ket hop giua:

- **Semantic Search** bang SPECTER2 embeddings
- **Agentic Reasoning** theo pattern ReAct (Reasoning + Acting)
- **4-Layer Memory System** (Semantic, Episodic, Working, Belief)
- **LLM-powered Analysis** thong qua Groq API voi LLaMA 3.3 70B

## 1.2. Muc Tieu Du An

| Muc tieu | Mo ta |
|----------|-------|
| **Tim kiem thong minh** | Tim bai bao bang ngon ngu tu nhien, khong can tu khoa chinh xac |
| **So sanh bai bao** | Tu dong phan tich diem giong va khac giua nhieu bai bao |
| **Tom tat noi dung** | Tao ban tom tat tu dong voi nhieu phong cach (brief/detailed/technical) |
| **Ca nhan hoa** | Hoc tu tuong tac cua nguoi dung de cai thien ket qua tim kiem |
| **Hoi thoai tu nhien** | Giao dien chat cho phep hoi dap ve bai bao mot cach tu nhien |

## 1.3. Technology Stack

### Core Technologies

| Component | Technology | Phien ban | Muc dich |
|-----------|-----------|-----------|----------|
| **Language** | Python | >= 3.11 | Ngon ngu chinh |
| **Data Models** | Pydantic v2 | >= 2.0.0 | Type-safe data validation |
| **Configuration** | pydantic-settings | >= 2.0.0 | Environment-based config |
| **Build System** | Hatchling | - | Python package build |

### AI/ML Stack

| Component | Technology | Mo ta |
|-----------|-----------|-------|
| **Embedding Model** | SPECTER2 (`allenai-specter`) | Model chuyen biet cho bai bao khoa hoc, 768 chieu |
| **ML Framework** | sentence-transformers | Wrapper cho SPECTER2 |
| **Tensor Library** | PyTorch | >= 2.0.0 |
| **LLM Provider** | Groq | Inference nhanh cho LLaMA models |
| **LLM Gateway** | LiteLLM | Unified API cho nhieu LLM providers |
| **LLM Model** | LLaMA 3.3 70B Versatile | Model chinh cho reasoning |

### Data & Storage

| Component | Technology | Mo ta |
|-----------|-----------|-------|
| **Vector Database** | Qdrant | >= 1.6.0, Cosine distance |
| **Relational Store** | SQLite (aiosqlite) | Async SQLite cho Episodic & Belief memory |
| **Dataset Source** | HuggingFace Datasets | `CShorten/ML-ArXiv-Papers` |

### API & Frontend

| Component | Technology | Mo ta |
|-----------|-----------|-------|
| **API Framework** | FastAPI | >= 0.100.0, REST API |
| **ASGI Server** | Uvicorn | Production-ready server |
| **HTTP Client** | httpx | Async HTTP requests |
| **Frontend** | Streamlit | >= 1.28.0, Chat-based UI |

### DevOps & Quality

| Component | Technology | Mo ta |
|-----------|-----------|-------|
| **Linter** | Ruff | Fast Python linter |
| **Type Checker** | mypy | Static type checking |
| **Testing** | pytest + pytest-asyncio | Unit & integration testing |
| **Coverage** | pytest-cov + Codecov | Code coverage tracking |
| **CI/CD** | GitHub Actions | 5-job pipeline |
| **Containerization** | Docker + Docker Compose | Multi-service deployment |
| **Logging** | structlog | Structured logging |
| **Retry Logic** | tenacity | Exponential backoff |
| **CLI Output** | Rich | Beautiful terminal output |

## 1.4. Cau Truc Thu Muc

```
PaperLens/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline (5 jobs)
├── docker/
│   └── Dockerfile                 # Multi-stage Docker build
├── docs/
│   └── TONG_QUAN_DU_AN.md        # Tai lieu nay
├── frontend/
│   └── app.py                     # Streamlit chat UI
├── scripts/
│   └── index_papers.py            # Script index bai bao vao Qdrant
├── src/
│   ├── __init__.py
│   ├── config.py                  # Settings tu environment variables
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py               # ReAct Agent chinh
│   │   ├── planner.py             # Query decomposition & planning
│   │   ├── prompts.py             # Prompt templates
│   │   └── tools.py               # 6 Agent tools
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py            # Chat endpoints
│   │       └── search.py          # Search endpoints
│   ├── clients/
│   │   ├── __init__.py
│   │   └── data_loader.py         # HuggingFace dataset loader
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── belief.py              # Belief Memory (SQLite)
│   │   ├── episodic.py            # Episodic Memory (SQLite)
│   │   ├── manager.py             # Memory Manager (orchestrator)
│   │   ├── semantic.py            # Semantic Memory (Qdrant wrapper)
│   │   └── working.py             # Working Memory (in-memory)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memory.py              # Memory data models
│   │   └── paper.py               # Paper data models
│   └── services/
│       ├── __init__.py
│       ├── embedding.py           # SPECTER2 embedding service
│       ├── llm.py                 # LLM service (Groq via LiteLLM)
│       └── vector_store.py        # Qdrant vector store service
├── tests/
│   ├── conftest.py                # Test fixtures & mocks
│   ├── test_agent.py
│   ├── test_data_loader.py
│   ├── test_memory.py
│   ├── test_models.py
│   └── test_services.py
├── .env.example                   # Template environment variables
├── .gitignore
├── .pre-commit-config.yaml        # Pre-commit hooks
├── docker-compose.yml             # 3-service orchestration
├── Makefile                       # Development commands
├── pyproject.toml                 # Project config, deps, tool settings
└── README.md
```

---

# 2. KIEN TRUC CHI TIET

## 2.1. Tong Quan Kien Truc

PaperLens su dung kien truc **Agentic RAG** voi 4 lop chinh:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐     │
│  │  Streamlit UI     │  │  FastAPI REST API             │     │
│  │  (frontend/app.py)│  │  (src/api/)                   │     │
│  └────────┬─────────┘  └──────────────┬───────────────┘     │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        │                                     │
├────────────────────────┼────────────────────────────────────┤
│                   AGENT LAYER                                │
│  ┌─────────────────────┴─────────────────────────────┐      │
│  │             PaperLensAgent (ReAct)                  │      │
│  │  ┌──────────────┐  ┌────────────┐  ┌────────────┐ │      │
│  │  │ QueryPlanner  │  │ ToolRegistry│  │  Prompts   │ │      │
│  │  │ (planner.py)  │  │ (tools.py)  │  │(prompts.py)│ │      │
│  │  └──────────────┘  └────────────┘  └────────────┘ │      │
│  └───────────────────────────────────────────────────┘      │
│                        │                                     │
├────────────────────────┼────────────────────────────────────┤
│                  MEMORY LAYER                                │
│  ┌─────────────────────┴─────────────────────────────┐      │
│  │              MemoryManager (manager.py)             │      │
│  │  ┌──────────┐ ┌──────────┐ ┌───────┐ ┌──────────┐ │      │
│  │  │ Semantic  │ │ Episodic │ │Working│ │  Belief  │ │      │
│  │  │ (Qdrant)  │ │ (SQLite) │ │(RAM)  │ │ (SQLite) │ │      │
│  │  └──────────┘ └──────────┘ └───────┘ └──────────┘ │      │
│  └───────────────────────────────────────────────────┘      │
│                        │                                     │
├────────────────────────┼────────────────────────────────────┤
│                 SERVICE LAYER                                │
│  ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐    │
│  │ EmbeddingService│ │  VectorStore  │ │   LLMService   │    │
│  │ (SPECTER2)      │ │  (Qdrant)     │ │ (Groq/LiteLLM) │    │
│  └─────────────────┘ └──────────────┘ └────────────────┘    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                   DATA LAYER                                 │
│  ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐    │
│  │ HuggingFace     │ │  Qdrant DB    │ │  SQLite DB     │    │
│  │ ML-ArXiv-Papers │ │  (vectors)    │ │  (memory)      │    │
│  └─────────────────┘ └──────────────┘ └────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## 2.2. Agentic RAG Architecture

### 2.2.1. ReAct Pattern

PaperLens Agent hoat dong theo pattern **ReAct (Reasoning + Acting)**, mot phuong phap cho phep LLM ket hop suy luan (reasoning) va hanh dong (acting) trong mot vong lap thong nhat.

**Vong lap ReAct:**

```
User Query
    │
    ▼
┌─────────────┐
│   THOUGHT    │  ← Agent suy luan ve cau hoi
│   (Reasoning)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   ACTION     │  ← Chon tool phu hop
│   (Acting)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ OBSERVATION  │  ← Nhan ket qua tu tool
│   (Result)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌───────────────┐
│  Du thong tin?├─NO─►  Quay lai     │
│              │     │  THOUGHT      │
└──────┬──────┘     └───────────────┘
       │YES
       ▼
┌─────────────┐
│ FINAL_ANSWER │  ← Tra loi nguoi dung
└─────────────┘
```

**Dinh dang output cua Agent:**

```
THOUGHT: [Suy luan ve buoc tiep theo]
ACTION: [ten_tool]
ACTION_INPUT: {"param1": "value1", "param2": "value2"}

OBSERVATION: [Ket qua tu tool - duoc he thong inject vao]

THOUGHT: [Suy luan tiep]
FINAL_ANSWER: [Cau tra loi cuoi cung cho nguoi dung]
```

**Gioi han vong lap:** Toi da `agent_max_iterations` = 5 vong (configurable). Neu vuot qua, raise `MaxIterationsError`.

### 2.2.2. So sanh voi Standard RAG

| Dac diem | Standard RAG | Agentic RAG (PaperLens) |
|----------|-------------|-------------------------|
| **Query Processing** | Query → Retrieve → Generate | Query → Plan → (Reason → Act → Observe)* → Generate |
| **Tool Usage** | Chi co retrieval | 6 tools chuyen biet |
| **Memory** | Khong co | 4 loai memory |
| **Multi-step** | Single retrieval | Nhieu buoc reasoning |
| **Personalization** | Khong co | Belief-based preference learning |
| **Query Planning** | Khong co | Pattern matching + LLM decomposition |

## 2.3. Memory Architecture

He thong memory 4 lop, lay cam hung tu mo hinh Cognitive Science:

### 2.3.1. Semantic Memory (Kien thuc dai han)

**File:** `src/memory/semantic.py`
**Storage:** Qdrant Vector Database
**Muc dich:** Luu tru embedding cua tat ca bai bao da index

```
┌──────────────────────────────────────────┐
│           SEMANTIC MEMORY                 │
│                                           │
│  ┌────────────────────────────────────┐  │
│  │         Qdrant Collection           │  │
│  │         "papers"                    │  │
│  │                                     │  │
│  │  Point {                            │  │
│  │    id: MD5(arxiv_id)[:15] as int    │  │
│  │    vector: [768 floats] (SPECTER2)  │  │
│  │    payload: {                       │  │
│  │      arxiv_id, title, abstract,     │  │
│  │      authors, categories, year,     │  │
│  │      citation_count, pdf_url,       │  │
│  │      arxiv_url                      │  │
│  │    }                                │  │
│  │  }                                  │  │
│  └────────────────────────────────────┘  │
│                                           │
│  Indexes: year (INTEGER), categories      │
│           (KEYWORD)                       │
│  Distance: COSINE                         │
│  Search API: query_points (v1.7+)         │
└──────────────────────────────────────────┘
```

**Cac thao tac chinh:**
- `search(query)` → Embed query bang SPECTER2 → Tim bai bao tuong tu trong Qdrant
- `get_paper(arxiv_id)` → Lay thong tin chi tiet bai bao theo ID
- `find_related(arxiv_id)` → Tim bai bao lien quan bang embedding similarity
- `get_stats()` → Thong ke ve index (tong so bai bao, trang thai)

### 2.3.2. Episodic Memory (Lich su tuong tac)

**File:** `src/memory/episodic.py`
**Storage:** SQLite (async via aiosqlite)
**Muc dich:** Ghi nho cac tuong tac truoc day de ho tro contextual recall

**Schema:**

```sql
CREATE TABLE episodic_memories (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    session_id TEXT,
    query TEXT NOT NULL,
    query_embedding TEXT,        -- JSON array cua floats
    action_type TEXT,            -- 'search', 'compare', 'summarize'
    result_paper_ids TEXT,       -- JSON array cua arxiv_ids
    result_count INTEGER,
    feedback TEXT,
    liked_paper_ids TEXT,        -- JSON array
    disliked_paper_ids TEXT,     -- JSON array
    metadata TEXT                -- JSON object
);

CREATE TABLE paper_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP,
    session_id TEXT,
    arxiv_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL,  -- 'view', 'like', 'dislike', 'compare', 'summarize'
    memory_id TEXT,
    metadata TEXT,
    FOREIGN KEY (memory_id) REFERENCES episodic_memories(id)
);
```

**Indexes:** session_id, created_at, query, arxiv_id, interaction_type

**Vi du su dung:** "Nhung bai bao toi da tim kiem tuan truoc" → Query episodic memory

### 2.3.3. Working Memory (Trang thai phien lam viec)

**File:** `src/memory/working.py`
**Storage:** In-memory (Python dict)
**Muc dich:** Duy tri context cua phien hien tai

```python
class WorkingMemoryState:
    session_id: str           # ID phien hien tai
    messages: list            # Lich su hoi thoai
    retrieved_paper_ids: list # Bai bao da truy xuat
    current_query: str        # Cau hoi dang xu ly
    agent_steps: list         # Cac buoc suy luan cua agent
    current_plan: list        # Ke hoach thuc thi hien tai
    scratch_pad: dict         # Bo nho tam thoi
```

**Dac diem:**
- Moi session co `WorkingMemoryState` rieng biet
- `max_size` = 20 messages (configurable) - trim messages cu nhat khi vuot qua
- Duoc xoa khi session ket thuc
- In-memory nen khong persist qua cac lan restart

### 2.3.4. Belief Memory (So thich nguoi dung)

**File:** `src/memory/belief.py`
**Storage:** SQLite (async via aiosqlite)
**Muc dich:** Hoc tu tuong tac de xay dung profile nguoi dung

**Schema:**

```sql
CREATE TABLE beliefs (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    belief_type TEXT NOT NULL,     -- 'favorite_category', 'favorite_author', etc.
    value TEXT NOT NULL,           -- 'cs.CL', 'Ashish Vaswani', etc.
    confidence REAL DEFAULT 0.5,  -- 0.0 - 1.0
    reinforcement_count INTEGER,
    source_memory_ids TEXT,        -- JSON array, lien ket voi episodic memories
    user_confirmed INTEGER,        -- 0 hoac 1
    UNIQUE(belief_type, value)
);
```

**6 loai Belief:**

| BeliefType | Mo ta | Vi du |
|------------|-------|-------|
| `FAVORITE_CATEGORY` | Danh muc yeu thich | `cs.CL`, `cs.LG` |
| `FAVORITE_AUTHOR` | Tac gia yeu thich | `Ashish Vaswani` |
| `PREFERRED_YEAR_RANGE` | Khoang nam uu tien | `2020-2024` |
| `READING_LEVEL` | Trinh do doc | `expert`, `beginner` |
| `INTEREST_TOPIC` | Chu de quan tam | `transformers`, `attention` |
| `DISLIKED_TOPIC` | Chu de khong thich | `reinforcement learning` |

**Co che Confidence:**
- **Khoi tao:** 0.5
- **Tang cuong (Reinforce):** `confidence += strength * (1 - confidence)`, tang `reinforcement_count`
- **Suy giam (Decay):** `confidence *= decay_factor` (0.95), chi ap dung cho beliefs chua duoc user confirm
- **Cat tia (Prune):** Xoa beliefs co `confidence < 0.1` va chua duoc user confirm
- **User Confirm:** Dat `confidence >= 0.8`, khong bi decay

### 2.3.5. Memory Manager (Dieu phoi)

**File:** `src/memory/manager.py`

Memory Manager la lop dieu phoi trung tam, cung cap:

1. **Context Building:** Tap hop thong tin tu tat ca memory stores de cung cap context cho agent
2. **Personalized Search:** Boost score dua tren preferences (`+0.05` cho category match, `+0.02` cho topic match)
3. **Memory Recording:** Ghi nhan searches va feedback vao episodic memory
4. **Learning:** Tu dong hoc tu ket qua tim kiem va feedback de cap nhat beliefs
5. **Consolidation:** Decay beliefs, prune low-confidence, extract patterns tu episodic memories
6. **Session Management:** Tao/ket thuc sessions

## 2.4. Data Flow Architecture

### 2.4.1. Search Flow

```
User: "Find papers about vision transformers from 2023"
│
├─1. Streamlit UI / FastAPI
│   └── POST /api/chat {message, session_id}
│
├─2. PaperLensAgent.run()
│   ├── Working Memory: luu query, tao/lay session
│   ├── Memory Manager: build context (beliefs + episodic + working)
│   └── ReAct Loop bat dau
│
├─3. Iteration 1: THOUGHT → ACTION
│   ├── LLM: "Tim bai bao ve vision transformers tu 2023"
│   ├── ACTION: search_papers
│   └── ACTION_INPUT: {"query": "vision transformers", "year_from": 2023}
│
├─4. Tool Execution
│   ├── SemanticMemory.search()
│   │   ├── EmbeddingService.embed_query("vision transformers")
│   │   │   └── SPECTER2 → vector [768 floats]
│   │   └── VectorStore.search(vector, year_from=2023)
│   │       └── Qdrant query_points + filter
│   ├── Ket qua: 10 papers sorted by cosine similarity
│   └── Working Memory: track paper IDs
│
├─5. Iteration 2: OBSERVATION → FINAL_ANSWER
│   ├── Agent xem ket qua
│   └── FINAL_ANSWER: "Toi tim thay 10 bai bao ve vision transformers..."
│
├─6. Post-processing
│   ├── Working Memory: luu assistant response
│   ├── Episodic Memory: ghi nhan search (query, results, session)
│   └── Belief Memory: hoc tu categories (reinforce cs.CV, cs.LG)
│
└─7. Response → UI
    └── {response, session_id, papers, steps_taken}
```

### 2.4.2. Comparison Flow

```
User: "Compare BERT and GPT"
│
├── QueryPlanner: detect intent = COMPARE
│   └── Pattern: regex match "compare"
│
├── Agent ReAct Loop
│   ├── Iteration 1: search_papers("BERT GPT")
│   │   └── Tim papers lien quan
│   │
│   ├── Iteration 2: compare_papers(paper_ids)
│   │   ├── Lay abstract cua cac papers
│   │   ├── Build comparison prompt
│   │   └── LLM generate comparison
│   │
│   └── Iteration 3: FINAL_ANSWER
│       └── Ket qua so sanh chi tiet
│
└── Response voi structured comparison
```

### 2.4.3. Memory Recall Flow

```
User: "What papers did I search for last week?"
│
├── QueryPlanner: detect intent = RECALL
│   └── Pattern: "last week", "previously"
│
├── Agent: ACTION = recall_memory
│   └── EpisodicMemoryStore.search_by_query()
│       └── SQLite: SELECT * FROM episodic_memories
│           WHERE query LIKE '%..%' ORDER BY created_at DESC
│
└── FINAL_ANSWER: "Day la cac tim kiem gan day cua ban..."
```

---

# 3. CHI TIET TRIEN KHAI

## 3.1. Configuration System

**File:** `src/config.py`

Su dung `pydantic-settings` voi `BaseSettings` de load configuration tu environment variables va file `.env`.

### 3.1.1. Settings Class

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # Bo qua env vars khong dinh nghia trong Settings
    )
```

### 3.1.2. Configuration Groups

**LLM Configuration:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `groq_api_key` | `""` | Groq API key |
| `openai_api_key` | `""` | OpenAI API key (backup) |
| `llm_provider` | `"groq"` | LLM provider: `groq` hoac `openai` |
| `llm_model` | `"llama-3.3-70b-versatile"` | Model name |

**Vector Database:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `qdrant_host` | `"localhost"` | Qdrant host |
| `qdrant_port` | `6333` | Qdrant HTTP port |
| `qdrant_collection` | `"papers"` | Ten collection |

**Embedding:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `embedding_model` | `"sentence-transformers/allenai-specter"` | SPECTER2 model |
| `embedding_dimension` | `768` | Kich thuoc vector |

**Search:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `search_top_k` | `10` | So bai bao toi da tra ve |
| `search_min_score` | `0.5` | Diem tuong tu toi thieu |
| `comparison_top_k` | `5` | So bai bao toi da so sanh |

**Agent:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `agent_max_iterations` | `5` | So vong ReAct toi da |
| `agent_temperature` | `0.7` | Temperature cho LLM reasoning |
| `agent_timeout` | `60` | Timeout (giay) |

**Memory:**

| Setting | Default | Mo ta |
|---------|---------|-------|
| `memory_db_path` | `"data/memory.db"` | Duong dan SQLite |
| `memory_working_size` | `20` | Toi da messages trong working memory |
| `memory_episodic_limit` | `100` | Toi da episodic memories tra ve |
| `memory_belief_decay` | `0.95` | He so suy giam confidence |

### 3.1.3. Computed Properties

```python
@property
def qdrant_url(self) -> str:
    return f"http://{self.qdrant_host}:{self.qdrant_port}"

@property
def llm_full_model(self) -> str:
    if self.llm_provider == "groq":
        return f"groq/{self.llm_model}"  # Vi du: "groq/llama-3.3-70b-versatile"
    return self.llm_model
```

### 3.1.4. Singleton Pattern

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()  # Import truc tiep
```

Tat ca services deu su dung `settings` singleton de doc configuration.

## 3.2. Data Models

### 3.2.1. Paper Models (`src/models/paper.py`)

**Paper** - Mo hinh bai bao khoa hoc:

```python
class Paper(BaseModel):
    arxiv_id: str           # "2301.12345"
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]   # ["cs.CL", "cs.LG"]
    published: datetime | None
    updated: datetime | None
    citation_count: int = 0
    influential_citation_count: int = 0
    tldr: str | None
    venue: str | None

    # Computed fields (tu dong tinh toan)
    @computed_field
    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"

    @computed_field
    @property
    def arxiv_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @computed_field
    @property
    def year(self) -> int | None:
        # Lay tu published date hoac tu arxiv_id (YYMM.NNNNN)
        if self.published:
            return self.published.year
        if self.arxiv_id and "." in self.arxiv_id:
            yymm = self.arxiv_id.split(".")[0]
            year = int(yymm[:2])
            return 2000 + year if year < 50 else 1900 + year
        return None
```

**PaperSearchResult** - Ket qua tim kiem:

```python
class PaperSearchResult(BaseModel):
    paper: Paper
    score: float  # 0.0 - 1.0, cosine similarity

    def __lt__(self, other) -> bool:
        return self.score > other.score  # Sort descending
```

**PaperComparison** - Ket qua so sanh:

```python
class PaperComparison(BaseModel):
    query: str
    papers: list[PaperSearchResult]
    comparison_text: str         # LLM-generated comparison
    timeline: list[dict] | None
    key_contributions: dict[str, str] | None
    model_used: str
    confidence: float
```

### 3.2.2. Memory Models (`src/models/memory.py`)

**MemoryType** - 4 loai memory:

```python
class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    WORKING = "working"
    BELIEF = "belief"
```

**MemoryItem** - Base class:

```python
class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    memory_type: MemoryType
```

**EpisodicMemory** - Lich su tuong tac:

```python
class EpisodicMemory(MemoryItem):
    query: str
    query_embedding: list[float] | None
    result_paper_ids: list[str]
    result_count: int
    feedback: str | None            # positive/negative/neutral
    liked_paper_ids: list[str]
    disliked_paper_ids: list[str]
    session_id: str | None
    action_type: str                # search, compare, summarize
    metadata: dict[str, Any]
```

**BeliefMemory** - So thich nguoi dung:

```python
class BeliefMemory(MemoryItem):
    belief_type: BeliefType         # FAVORITE_CATEGORY, INTEREST_TOPIC, etc.
    value: str
    confidence: float               # 0.0 - 1.0
    reinforcement_count: int
    source_memory_ids: list[str]    # Lien ket voi episodic memories
    user_confirmed: bool
```

**WorkingMemoryState** - Trang thai phien:

```python
class WorkingMemoryState(MemoryItem):
    session_id: str
    messages: list[ConversationMessage]
    retrieved_paper_ids: list[str]
    current_query: str | None
    agent_steps: list[AgentStep]
    current_plan: list[str] | None
    scratch_pad: dict[str, Any]
```

**AgentStep** - Buoc suy luan:

```python
class AgentStep(BaseModel):
    step_number: int
    thought: str
    action: str | None
    action_input: dict[str, Any] | None
    observation: str | None
    timestamp: datetime
```

## 3.3. Service Layer

### 3.3.1. Embedding Service (`src/services/embedding.py`)

**Muc dich:** Tao embedding vector cho bai bao va truy van su dung SPECTER2.

**SPECTER2** la model embedding duoc thiet ke dac biet cho van ban khoa hoc. No hoat dong tot nhat khi dau vao la "title + abstract".

**Dac diem:**
- **Model:** `sentence-transformers/allenai-specter`
- **Dimension:** 768
- **Lazy Loading:** Model chi duoc tai khi lan dau su dung (thong qua `@property`)
- **Batch Processing:** Ho tro embed nhieu van ban cung luc
- **Singleton Pattern:** `get_embedding_service()` tra ve instance duy nhat

**Cac method chinh:**

```python
class EmbeddingService:
    def embed_text(self, text: str) -> list[float]
        # Embed 1 van ban → vector 768 chieu

    def embed_texts(self, texts: list[str], batch_size=32) -> list[list[float]]
        # Embed nhieu van ban theo batch

    def embed_paper(self, paper: Paper) -> list[float]
        # Embed bai bao: title + abstract

    def embed_papers(self, papers: list[Paper], batch_size=32) -> list[list[float]]
        # Embed nhieu bai bao

    def embed_query(self, query: str) -> list[float]
        # Embed cau truy van (cung method voi embed_text)

    def similarity(self, embedding1, embedding2) -> float
        # Cosine similarity giua 2 vectors
```

**Tai sao chon SPECTER2:**
1. Duoc huan luyen tren bai bao khoa hoc (khong phai general-purpose)
2. Hieu ro ngu canh hoc thuat (methodology, results, contributions)
3. 768 chieu la su can bang giua chat luong va hieu suat
4. Duoc phat trien boi Allen Institute for AI

### 3.3.2. Vector Store (`src/services/vector_store.py`)

**Muc dich:** Quan ly storage va retrieval cua paper embeddings trong Qdrant.

**Qdrant Configuration:**

```python
# Tao collection voi Cosine distance
client.create_collection(
    collection_name="papers",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
)

# Tao payload indexes de filter nhanh
client.create_payload_index("papers", "year", PayloadSchemaType.INTEGER)
client.create_payload_index("papers", "categories", PayloadSchemaType.KEYWORD)
```

**Point ID Generation:**

```python
# Su dung MD5 hash cua arxiv_id de tao integer ID
point_id = int(hashlib.md5(paper.arxiv_id.encode()).hexdigest()[:15], 16)
```

Ly do: Qdrant yeu cau ID la integer hoac UUID. Su dung MD5 hash dam bao:
- Deterministic (cung arxiv_id luon cho cung ID)
- Upsert hoat dong dung (khong duplicate)
- Phan phoi deu

**Search voi Filters:**

```python
def search(self, query_vector, limit=10, year_from=None, year_to=None, categories=None):
    # Build filter conditions
    must_conditions = []

    if year_from:
        must_conditions.append(FieldCondition(key="year", range=Range(gte=year_from)))
    if year_to:
        must_conditions.append(FieldCondition(key="year", range=Range(lte=year_to)))
    if categories:
        must_conditions.append(FieldCondition(key="categories", match=MatchAny(any=categories)))

    # Su dung query_points API (Qdrant v1.7+)
    response = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        limit=limit,
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        score_threshold=min_score,
    )
```

**Tai sao chon Qdrant:**
1. Open-source, self-hosted
2. Ho tro filter dua tren payload (year, categories)
3. API don gian, Python client tot
4. Hieu suat cao voi cosine similarity
5. Ho tro upsert (insert hoac update)

### 3.3.3. LLM Service (`src/services/llm.py`)

**Muc dich:** Cung cap interface thong nhat cho cac LLM providers thong qua LiteLLM.

**Configuration:**

```python
class LLMService:
    model = "groq/llama-3.3-70b-versatile"  # Provider/model format cho LiteLLM
    temperature = 0.7
```

**Retry Logic voi tenacity:**

```python
@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, RateLimitError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)
def chat_completion(self, messages, temperature=None, max_tokens=2048, stop=None):
    response = completion(model=self.model, messages=messages, ...)
    return response.choices[0].message.content
```

**Chi tiet retry:**
- **So lan thu lai:** toi da 5 lan
- **Backoff:** Exponential, bat dau tu 5s, toi da 60s (5s → 10s → 20s → 40s → 60s)
- **Retry khi:** `ConnectionError`, `TimeoutError`, `RateLimitError`
- **Reraise:** Sau 5 lan that bai, throw exception goc

**Cac method:**

| Method | Mo ta | Sync/Async |
|--------|-------|------------|
| `chat_completion()` | Chat completion co ban | Sync |
| `achat_completion()` | Async chat completion | Async |
| `chat_completion_stream()` | Streaming response | Sync Generator |
| `achat_completion_stream()` | Async streaming | Async Generator |
| `generate_with_tools()` | Tool/function calling | Sync |
| `agenerate_with_tools()` | Async tool calling | Async |

**Tai sao chon Groq + LiteLLM:**
1. **Groq:** Inference cuc nhanh (LPU), free tier generous
2. **LiteLLM:** De dang chuyen doi provider (Groq ↔ OpenAI) chi can thay doi config
3. **LLaMA 3.3 70B:** Model manh, mien phi qua Groq, tot cho reasoning

## 3.4. Agent System

### 3.4.1. PaperLens Agent (`src/agent/agent.py`)

**Thanh phan cua Agent:**

```python
class PaperLensAgent:
    llm: LLMService           # De goi LLM
    tools: ToolRegistry        # Registry cua 6 tools
    memory: WorkingMemory      # Working memory cho session
    memory_manager: MemoryManager  # Dieu phoi 4 loai memory
    planner: QueryPlanner      # Phan tich va phan ra cau hoi
    max_iterations: int        # Toi da 5 vong ReAct
```

**Luong xu ly chinh:**

```python
def run(self, query, session_id=None):
    # 1. Khoi tao session
    session_id = session_id or str(uuid4())
    self.memory.get_session(session_id)
    self.memory.clear_steps(session_id)        # Clear buoc cu
    self.memory.set_query(session_id, query)
    self.memory.add_message(session_id, "user", query)

    # 2. Chay ReAct loop
    response = self._react_loop(query, session_id)

    # 3. Luu ket qua
    self.memory.add_message(session_id, "assistant", response)

    # 4. Tra ve response
    return AgentResponse(response, session_id, steps, papers)
```

**ReAct Loop chi tiet:**

```python
def _react_loop(self, query, session_id):
    context = self._build_context(session_id)  # Working + Beliefs + Episodic

    for iteration in range(self.max_iterations):
        # Build prompt voi context va tool schemas
        prompt = format_react_prompt(query, tool_schemas, context)

        # Them reasoning history tu session hien tai
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if steps:
            messages.append({"role": "assistant", "content": formatted_steps})
            messages.append({"role": "user", "content": "Continue your reasoning:"})

        # Goi LLM voi stop sequence "OBSERVATION:"
        response = self.llm.chat_completion(messages, stop=["OBSERVATION:"])

        # Parse response
        parsed = self._parse_response(response)

        if parsed["type"] == "final_answer":
            return parsed["content"]

        elif parsed["type"] == "action":
            # Execute tool
            result = self._execute_tool(action, action_input, session_id)
            # Update context
            context = self._build_context(session_id)

    raise MaxIterationsError(...)
```

**Response Parsing:**

Agent su dung regex de parse response tu LLM:

```python
def _parse_response(self, response):
    # Check FINAL_ANSWER
    if "FINAL_ANSWER:" in response:
        return {"type": "final_answer", "content": ...}

    # Check ACTION
    if "ACTION:" in response:
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
        action_match = re.search(r"ACTION:\s*(\w+)", response)
        input_match = re.search(r"ACTION_INPUT:\s*(\{.+?\})", response, re.DOTALL)
        return {"type": "action", "thought": ..., "action": ..., "action_input": ...}

    return {"type": "unknown", "content": response}
```

**Context Building:**

```python
def _build_context(self, session_id):
    # 1. Working memory: conversation history, papers, current query
    basic_context = format_conversation_context(history, papers, query)

    # 2. Rich context: beliefs + episodic
    rich_context = memory_manager.build_context(
        session_id=session_id,
        query=current_query,
        include_beliefs=True,
        include_episodic=True,
        max_episodic=3,
    )

    # 3. Ket hop
    return basic_context + "\n\n## User Profile\n" + formatted_rich
```

### 3.4.2. Tool System (`src/agent/tools.py`)

**6 Tools co san:**

#### 1. SearchPapersTool

```
Name: search_papers
Mo ta: Tim bai bao bang semantic search
Parameters:
  - query (string, required): Cau truy van
  - limit (integer): So luong toi da (default: 10)
  - year_from (integer): Loc tu nam
  - year_to (integer): Loc den nam
  - categories (array[string]): Loc theo danh muc ArXiv
```

**Luong xu ly:**
1. Nhan query tu Agent
2. Goi `SemanticMemory.search()` → embed query → search Qdrant
3. Format ket qua: `{arxiv_id, title, year, score, abstract_preview}`
4. Tra ve `ToolResult(success=True, data=[...])`

#### 2. GetPaperTool

```
Name: get_paper
Mo ta: Lay chi tiet bai bao theo ArXiv ID
Parameters:
  - arxiv_id (string, required): ArXiv ID
```

#### 3. GetRelatedPapersTool

```
Name: get_related
Mo ta: Tim bai bao lien quan
Parameters:
  - arxiv_id (string, required): ArXiv ID goc
  - limit (integer): So luong (default: 5)
```

**Luong xu ly:**
1. Lay paper goc tu Qdrant
2. Embed text cua paper goc (title + abstract)
3. Tim papers tuong tu bang embedding search
4. Loai bo paper goc khoi ket qua

#### 4. ComparePapersTool

```
Name: compare_papers
Mo ta: So sanh 2-5 bai bao
Parameters:
  - paper_ids (array[string], required): Danh sach ArXiv IDs
  - aspects (array[string]): Khia canh so sanh (default: methodology, contributions, key_findings)
```

**Luong xu ly:**
1. Lay thong tin cac papers tu Qdrant
2. Build comparison prompt voi abstracts
3. Goi LLM de generate comparison (max_tokens=1500)
4. Tra ve `{comparison, papers, aspects}`

#### 5. SummarizePaperTool

```
Name: summarize_paper
Mo ta: Tom tat bai bao
Parameters:
  - arxiv_id (string, required): ArXiv ID
  - style (string): Phong cach: "brief" (500 tokens), "detailed" (1000), "technical" (1200)
```

**Cau truc tom tat:**
1. Problem/Motivation
2. Approach/Methodology
3. Key Contributions
4. Results/Findings
5. Significance

#### 6. RecallMemoryTool

```
Name: recall_memory
Mo ta: Tim kiem tuong tac truoc day tu memory
Parameters:
  - query (string, required): Tim kiem gi
  - limit (integer): Toi da (default: 5)
```

**Luong xu ly:**
1. Goi `EpisodicMemoryStore.search_by_query()` (SQLite LIKE search)
2. Format ket qua: `{query, papers, result_count, liked_papers, when}`

### 3.4.3. Tool Registry Pattern

```python
class ToolRegistry:
    _tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None
    def get(self, name: str) -> Tool | None
    def list_tools(self) -> list[str]
    def get_schemas(self) -> list[dict]    # OpenAI function format
    def execute(self, name: str, **kwargs) -> ToolResult

# Default registry voi tat ca 6 tools
def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchPapersTool())
    registry.register(GetPaperTool())
    registry.register(GetRelatedPapersTool())
    registry.register(ComparePapersTool())
    registry.register(SummarizePaperTool())
    registry.register(RecallMemoryTool())
    return registry
```

### 3.4.4. Query Planner (`src/agent/planner.py`)

**Muc dich:** Phan tich cau hoi nguoi dung, xac dinh intent va tao ke hoach thuc thi.

**Intent Detection (Pattern Matching):**

| Intent | Patterns |
|--------|----------|
| `COMPARE` | `compare`, `difference between`, `vs`, `versus`, `contrast` |
| `SUMMARIZE` | `summary`, `summarize`, `explain...paper`, `tldr`, `brief overview` |
| `FIND_RELATED` | `related to`, `similar to`, `like this`, `more like` |
| `RECALL` | `last time`, `previously`, `before`, `history`, `remember` |
| `SEARCH` | Default khi khong match pattern nao |

**Year Filter Detection:**

```python
# "from 2023", "since 2023", "after 2023" → year_from = 2023
# "before 2024", "until 2023"             → year_to = 2024
# "in 2023", "2023 papers"                → year_from = year_to = 2023
# "recent", "latest", "new"               → year_from = current_year - 1
```

**Complex Query Detection:**

Query duoc coi la phuc tap khi:
- Co hon 2 menh de ("and")
- Co hon 2 dau phay
- Dai hon 15 tu
- Co nhieu dau hoi

→ Su dung LLM de phan ra thanh subtasks (output JSON).

**Query Plan Structure:**

```python
class QueryPlan:
    original_query: str
    intent: str                    # search, compare, summarize, etc.
    steps: list[PlanStep]          # Danh sach buoc thuc thi
    requires_comparison: bool
    requires_summary: bool

class PlanStep:
    task: str                      # Mo ta buoc
    tool: str | None               # Tool can su dung
    parameters: dict               # Tham so cho tool
    depends_on: list[int]          # Chi so cac buoc phu thuoc
```

### 3.4.5. Prompt Templates (`src/agent/prompts.py`)

**SYSTEM_PROMPT** - Dinh nghia persona cua Agent:

```
You are PaperLens, an intelligent research assistant specialized
in finding, analyzing, and comparing machine learning papers.

Your capabilities:
1. Search: Find papers using semantic search
2. Retrieve: Get full details of specific papers
3. Compare: Analyze similarities and differences
4. Summarize: Generate clear summaries
5. Relate: Find papers related to a given paper
```

**REACT_PROMPT** - Template chinh cho ReAct loop:

Chua 3 placeholder:
- `{tools_description}` - Mo ta va parameters cua cac tools
- `{context}` - Context tu Memory (conversation, beliefs, episodic)
- `{query}` - Cau hoi cua nguoi dung

**Cac prompt khac:**
- `COMPARE_PROMPT` - Template cho so sanh bai bao
- `SUMMARY_PROMPT` - Template cho tom tat (co 3 styles)
- `QUERY_DECOMPOSITION_PROMPT` - Template cho LLM-based query planning
- `CONVERSATION_CONTEXT_PROMPT` - Template cho context building
- `MEMORY_RECALL_PROMPT` - Template cho recall context
- `RESPONSE_TEMPLATES` - Templates cho cac loai response (search_results, paper_details, no_results, error)

## 3.5. API Layer

### 3.5.1. FastAPI Application (`src/api/main.py`)

**Application Setup:**

```python
app = FastAPI(
    title="PaperLens API",
    description="Agentic RAG-based ML Paper Search & Comparison Engine",
    version="0.1.0",
    lifespan=lifespan,   # Async lifespan handler
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
```

**Lifespan Handler:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: khoi tao Memory Manager
    get_memory_manager()
    yield
    # Shutdown: cleanup
```

### 3.5.2. API Endpoints

#### Health & Status

| Method | Path | Mo ta |
|--------|------|-------|
| `GET` | `/health` | Health check (status, version, memory stats) |
| `GET` | `/api/stats` | Thong ke chi tiet (papers, memory) |

#### Chat Endpoints (`src/api/routes/chat.py`)

| Method | Path | Mo ta |
|--------|------|-------|
| `POST` | `/api/chat` | Gui tin nhan → Agent xu ly → Tra loi |
| `POST` | `/api/chat/stream` | Chat voi streaming response (SSE) |
| `POST` | `/api/chat/session` | Tao session moi |
| `GET` | `/api/chat/session/{id}` | Lay thong tin session |
| `GET` | `/api/chat/session/{id}/history` | Lay lich su hoi thoai |
| `DELETE` | `/api/chat/session/{id}` | Xoa session |
| `GET` | `/api/chat/tools` | Lay danh sach tools co san |

**Chat Request/Response:**

```python
# Request
class ChatRequest(BaseModel):
    message: str              # Tin nhan nguoi dung
    session_id: str | None    # Session ID (tuy chon)

# Response
class ChatResponse(BaseModel):
    response: str             # Phan hoi cua Agent
    session_id: str           # Session ID
    papers: list[str]         # ArXiv IDs duoc tham chieu
    steps_taken: int          # So buoc reasoning
```

#### Search Endpoints (`src/api/routes/search.py`)

| Method | Path | Mo ta |
|--------|------|-------|
| `POST` | `/api/search` | Tim kiem (voi body JSON) |
| `GET` | `/api/search` | Tim kiem (voi query params) |
| `POST` | `/api/compare` | So sanh 2-5 bai bao |
| `GET` | `/api/papers/{id}/related` | Tim bai bao lien quan |

**Search Request/Response:**

```python
# Request
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    year_from: int | None
    year_to: int | None
    categories: list[str] | None
    use_personalization: bool = True
    session_id: str | None

# Response
class SearchResponse(BaseModel):
    papers: list[PaperResult]  # Ket qua tim kiem
    total: int
    query: str
    personalized: bool         # Co su dung personalization khong
```

#### Memory Endpoints

| Method | Path | Mo ta |
|--------|------|-------|
| `POST` | `/api/memory/feedback` | Ghi nhan feedback (like/dislike paper) |
| `GET` | `/api/memory/history` | Lay lich su tim kiem |
| `GET` | `/api/memory/preferences` | Lay so thich da hoc |
| `GET` | `/api/papers/{arxiv_id}` | Lay chi tiet bai bao |

## 3.6. Frontend (`frontend/app.py`)

### 3.6.1. Streamlit UI

**Page Config:**

```python
st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

**Session State:**

```python
st.session_state.session_id     # UUID cho moi phien
st.session_state.messages       # Lich su tin nhan
st.session_state.papers_viewed  # Bai bao da xem
```

**Sidebar:**
- **Session info:** Hien thi session ID, nut "New Session"
- **Quick Search:** Text input voi year filters (From/To)
- **Example Queries:** 4 vi du co the click
- **Stats:** So bai bao da index, so session dang hoat dong

**Main Chat Interface:**
- Su dung `st.chat_message` de hien thi hoi thoai
- `st.chat_input` de nhap tin nhan
- `st.spinner("Thinking...")` khi agent dang xu ly
- `st.expander` de hien thi papers duoc tham chieu

**Cached Resources:**

```python
@st.cache_resource
def get_cached_agent():        # Agent duoc cache, chi khoi tao 1 lan
    return get_agent()

@st.cache_resource
def get_cached_memory_manager():  # Memory manager duoc cache
    return get_memory_manager()
```

**Footer:** Hien thi "Built with Streamlit + FastAPI", "Powered by Groq + SPECTER2"

## 3.7. Data Ingestion (`scripts/index_papers.py`)

**Muc dich:** Download bai bao tu HuggingFace va index vao Qdrant.

**Su dung:**

```bash
python scripts/index_papers.py --limit 1000           # Index 1000 bai bao dau tien
python scripts/index_papers.py --batch-size 100        # Batch lon hon
python scripts/index_papers.py --recreate --limit 5000 # Tao lai collection
python scripts/index_papers.py --categories cs.LG cs.AI # Loc theo danh muc
python scripts/index_papers.py --verify                 # Chi kiem tra index
```

**Luong xu ly:**

```
1. Khoi tao services (DataLoader, EmbeddingService, VectorStore)
2. Tao/recreate collection trong Qdrant
3. Load dataset tu HuggingFace (CShorten/ML-ArXiv-Papers)
4. Xu ly theo batch (default: 50 papers/batch):
   a. Loc papers hop le (co arxiv_id, title, abstract)
   b. Generate embeddings bang SPECTER2
   c. Upsert vao Qdrant
5. Hien thi progress bar (Rich library)
6. In thong ke: processed, indexed, skipped, errors, duration
7. Verify: chay test query "transformer attention mechanism"
```

**Data Loader** (`src/clients/data_loader.py`):
- Load dataset `CShorten/ML-ArXiv-Papers` tu HuggingFace
- Parse paper: extract arxiv_id tu nhieu fields (`id`, `arxiv_id`, `paper_id`, `Unnamed: 0`)
- Parse authors: xu ly ca string ("Author1, Author2") va list formats
- Category filtering: loc theo ArXiv categories

---

# 4. LUONG XU LY (WORKFLOWS)

## 4.1. Startup Workflow

```
1. User chay: `make dev` hoac `docker compose up`

2. Qdrant:
   - Container khoi dong
   - Port 6333 (HTTP) + 6334 (gRPC) duoc expose
   - Volume qdrant_data duoc mount

3. API Server (uvicorn):
   - FastAPI app duoc load
   - Lifespan handler: khoi tao MemoryManager
   - MemoryManager lazy-load cac stores (semantic, episodic, working, belief)
   - Port 8000

4. Frontend (streamlit):
   - Streamlit app duoc load
   - Agent va MemoryManager duoc cache (@st.cache_resource)
   - Port 8501
```

## 4.2. Paper Indexing Workflow

```
scripts/index_papers.py --limit 5000 --batch-size 100
│
├── 1. Initialize
│   ├── HuggingFaceDataLoader("CShorten/ML-ArXiv-Papers")
│   ├── EmbeddingService("sentence-transformers/allenai-specter")
│   └── VectorStore(host=localhost, port=6333, collection="papers")
│
├── 2. Setup Collection
│   └── VectorStore.create_collection(vector_size=768, distance=COSINE)
│       ├── Payload index: "year" (INTEGER)
│       └── Payload index: "categories" (KEYWORD)
│
├── 3. Load Dataset
│   └── datasets.load_dataset("CShorten/ML-ArXiv-Papers", split="train")
│
├── 4. Process Batches (100 papers/batch)
│   ├── Filter: skip papers without arxiv_id, title, or abstract
│   ├── Embed: EmbeddingService.embed_papers(batch) → [[768 floats], ...]
│   └── Upsert: VectorStore.upsert_papers(papers, embeddings)
│       └── Point(id=MD5(arxiv_id), vector=embedding, payload={...})
│
├── 5. Verify
│   └── SemanticMemory.search("transformer attention mechanism", limit=3)
│
└── 6. Output Stats
    ├── Papers processed: 5000
    ├── Papers indexed: 4800
    ├── Errors: 50
    └── Duration: 120.5s
```

## 4.3. User Chat Workflow (Chi tiet)

```
User nhap: "Find recent papers about diffusion models"
│
├── 1. Streamlit UI
│   ├── st.session_state.messages.append({"role": "user", "content": ...})
│   └── run_agent_query(query, session_id)
│
├── 2. PaperLensAgent.run(query, session_id)
│   ├── memory.get_session(session_id)      # Tao/lay session
│   ├── memory.clear_steps(session_id)      # Clear buoc cu
│   ├── memory.set_query(session_id, query) # Set query hien tai
│   └── memory.add_message(session_id, "user", query)
│
├── 3. _react_loop(query, session_id)
│   │
│   ├── 3a. Build Context
│   │   ├── Working: conversation history, papers, query
│   │   ├── Beliefs: favorite categories, topics
│   │   └── Episodic: 3 relevant past searches
│   │
│   ├── 3b. Iteration 1
│   │   ├── Build messages:
│   │   │   ├── system: SYSTEM_PROMPT (persona)
│   │   │   └── user: REACT_PROMPT (tools + context + query)
│   │   │
│   │   ├── LLM Call:
│   │   │   ├── Model: groq/llama-3.3-70b-versatile
│   │   │   ├── Temperature: 0.7
│   │   │   └── Stop: ["OBSERVATION:"]
│   │   │
│   │   ├── Parse Response:
│   │   │   ├── THOUGHT: "User muon tim bai bao gan day ve diffusion models"
│   │   │   ├── ACTION: search_papers
│   │   │   └── ACTION_INPUT: {"query": "diffusion models", "year_from": 2024}
│   │   │
│   │   ├── Execute Tool:
│   │   │   ├── SearchPapersTool.execute(query="diffusion models", year_from=2024)
│   │   │   │   ├── embed_query("diffusion models") → [768 floats]
│   │   │   │   └── vector_store.search(vector, year_from=2024) → 10 results
│   │   │   └── Track papers in working memory
│   │   │
│   │   ├── Record to Episodic Memory:
│   │   │   └── EpisodicMemory(query="diffusion models", results=[...])
│   │   │
│   │   └── Format Observation:
│   │       └── "Found 10 results:\n1. Paper Title (arxiv_id)..."
│   │
│   └── 3c. Iteration 2
│       ├── Build messages (bao gom history tu Iteration 1)
│       │
│       ├── LLM Call → Parse Response:
│       │   └── THOUGHT: "Da co ket qua, tong hop cho nguoi dung"
│       │       FINAL_ANSWER: "Day la 10 bai bao gan day ve diffusion models..."
│       │
│       └── Return final answer
│
├── 4. Post-processing
│   ├── memory.add_message(session_id, "assistant", response)
│   └── Belief Learning:
│       └── Reinforce categories tu top results (cs.CV, cs.LG)
│
└── 5. Response → UI
    ├── st.markdown(response)
    └── st.expander("Referenced papers")
```

## 4.4. Memory Consolidation Workflow

```
MemoryManager.consolidate()
│
├── 1. Decay Beliefs
│   └── UPDATE beliefs SET confidence = confidence * 0.95
│       WHERE user_confirmed = 0 AND confidence > 0.1
│
├── 2. Prune Low-Confidence Beliefs
│   └── DELETE FROM beliefs WHERE confidence < 0.1 AND user_confirmed = 0
│
├── 3. Extract Patterns from Recent Episodic Memories
│   ├── Get memories from last 7 days (168 hours)
│   ├── Count keyword frequency (words > 4 chars)
│   └── Reinforce keywords appearing >= 3 times as INTEREST_TOPIC
│
└── Returns: {beliefs_decayed, beliefs_pruned, topics_reinforced}
```

## 4.5. Personalized Search Workflow

```
MemoryManager.search_papers(query, session_id, use_preferences=True)
│
├── 1. Basic Search
│   └── semantic.search(query, limit=20)  # 2x limit de co du cho filtering
│
├── 2. Get User Preferences
│   └── belief.get_preferences_summary()
│       ├── favorite_categories: [{category: "cs.CL", confidence: 0.8}, ...]
│       ├── favorite_authors: [...]
│       └── interest_topics: [{topic: "transformers", confidence: 0.7}, ...]
│
├── 3. Boost Scores
│   ├── Category match: +0.05 per matching category
│   ├── Topic match in title/abstract: +0.02 per matching topic
│   └── Cap at 1.0
│
├── 4. Re-sort by Boosted Score (descending)
│
├── 5. Trim to Original Limit (10)
│
└── 6. Track in Working Memory
```

---

# 5. VAN DE & GIAI PHAP

## 5.1. CI/CD Issues

### 5.1.1. Docker Build - Dockerfile Not Found

**Van de:** GitHub Actions Docker Build job that bai voi loi:
```
ERROR: failed to build: failed to solve: failed to read dockerfile:
open Dockerfile: no such file or directory
```

**Nguyen nhan:** Dockerfile nam tai `docker/Dockerfile`, khong phai o root. `docker/build-push-action` mac dinh tim `Dockerfile` o root.

**Giai phap:** Them `file: docker/Dockerfile` vao CI configuration:

```yaml
# .github/workflows/ci.yml
- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: docker/Dockerfile    # ← Them dong nay
    push: false
    tags: paperlens:latest
```

### 5.1.2. Ruff Lint - StrEnum vs (str, Enum)

**Van de:** Ruff rule `UP042` - "Class inherits from both `str` and `enum.Enum`"

```python
# Loi
from enum import Enum
class MemoryType(str, Enum):    # UP042
    ...
class BeliefType(str, Enum):    # UP042
    ...
```

**Giai phap:** Su dung `StrEnum` (Python 3.11+):

```python
# Dung
from enum import StrEnum
class MemoryType(StrEnum):
    ...
class BeliefType(StrEnum):
    ...
```

### 5.1.3. Qdrant Container trong CI

**Van de:** Test job can Qdrant de chay, nhung container can thoi gian khoi dong.

**Giai phap:** Su dung GitHub Actions service container voi health check wait loop:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - 6333:6333

steps:
  - name: Wait for Qdrant
    run: |
      for i in $(seq 1 30); do
        if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
          echo "Qdrant is ready"
          exit 0
        fi
        echo "Waiting for Qdrant... ($i/30)"
        sleep 2
      done
      echo "Qdrant failed to start"
      exit 1
```

## 5.2. Async/Sync Bridging

**Van de:** Agent (sync) can goi Memory Manager (async methods).

**Giai phap:** `_run_async()` helper trong Agent:

```python
def _run_async(self, coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Dang trong async context → dung ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
```

**Nhan xet:** Day la pattern huu ich nhung co the gay issues voi nested event loops. Trong tuong lai, nen chuyen Agent sang async hoan toan.

## 5.3. LLM Rate Limiting

**Van de:** Groq API co rate limits, dac biet khi chay nhieu requests lien tuc.

**Giai phap:** Exponential backoff voi tenacity:

```python
@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, RateLimitError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)
def chat_completion(self, messages, ...):
    ...
```

## 5.4. Large Payload trong Qdrant

**Van de:** Luu toan bo abstract trong payload co the gay cham khi search.

**Giai phap hien tai:** Chap nhan trade-off vi can abstract de hien thi va so sanh. Qdrant xu ly tot voi payload vua phai.

**Cai thien trong tuong lai:** Co the luu abstract rieng va chi luu metadata trong payload.

## 5.5. Model Loading Time

**Van de:** SPECTER2 model tai cham (~30s lan dau), anh huong UX.

**Giai phap:**
1. **Lazy Loading:** Model chi tai khi lan dau su dung (khong tai khi app start)
2. **Singleton Pattern:** Tai 1 lan, dung lai cho moi request
3. **Streamlit Cache:** `@st.cache_resource` giu model trong memory
4. **Docker Volume:** `model_cache:/root/.cache` persist model qua cac lan restart container

---

# 6. HIEU SUAT & TOI UU HOA

## 6.1. Embedding Performance

| Metric | Gia tri |
|--------|---------|
| Model size | ~500MB |
| First load time | ~30s |
| Single text embed | ~50ms |
| Batch embed (32 texts) | ~800ms |
| Dimension | 768 |

**Toi uu:**
- Lazy loading: chi tai khi can
- Batch processing: embed nhieu text cung luc
- Singleton: khong tai lai model

## 6.2. Vector Search Performance

| Metric | Gia tri |
|--------|---------|
| Collection size | Tuy thuoc (dataset co ~117k papers) |
| Search latency | ~10-50ms (tuy thuoc collection size) |
| Distance metric | Cosine |
| Index type | HNSW (Qdrant default) |

**Toi uu:**
- Payload indexes cho year va categories → filter nhanh
- `query_points` API (v1.7+) thay vi `search` cu
- `score_threshold` loai bo ket qua khong lien quan

## 6.3. LLM Performance

| Metric | Gia tri |
|--------|---------|
| Provider | Groq |
| Model | LLaMA 3.3 70B |
| Latency (first token) | ~200ms |
| Throughput | ~500 tokens/s |
| Max tokens | 2048 (default) |

**Toi uu:**
- Stop sequence `"OBSERVATION:"` de dung som
- Temperature 0.7 cho reasoning, 0.3 cho structured output
- Max tokens thay doi theo task (500 brief, 1000 detailed, 1500 compare)
- Retry voi exponential backoff

## 6.4. Memory Performance

| Component | Storage | Latency |
|-----------|---------|---------|
| Working Memory | In-memory (dict) | ~0ms |
| Episodic Memory | SQLite | ~1-5ms per query |
| Belief Memory | SQLite | ~1-5ms per query |
| Semantic Memory | Qdrant | ~10-50ms per search |

**Toi uu:**
- Working Memory: gioi han 20 messages, trim cu
- SQLite indexes: session_id, created_at, query, belief_type
- Lazy loading cho tat ca memory stores
- Async SQLite operations (aiosqlite)

## 6.5. Overall Request Latency

| Giai doan | Thoi gian |
|-----------|-----------|
| Context building | ~10-20ms |
| LLM reasoning (per iteration) | ~1-3s |
| Tool execution (search) | ~100-200ms |
| Total (1 iteration) | ~1.5-3.5s |
| Total (2 iterations) | ~3-7s |
| Total (max 5 iterations) | ~7-18s |

## 6.6. Singleton Pattern

Tat ca services su dung singleton pattern de tranh khoi tao lai:

```python
_embedding_service: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

**Services su dung singleton:**
- `get_settings()` (voi `@lru_cache`)
- `get_embedding_service()`
- `get_vector_store()`
- `get_llm_service()`
- `get_semantic_memory()`
- `get_working_memory()`
- `get_episodic_store()`
- `get_belief_store()`
- `get_memory_manager()`
- `get_planner()`
- `get_tool_registry()`
- `get_agent()`

---

# 7. CHIEN LUOC TESTING

## 7.1. Test Framework

| Tool | Muc dich |
|------|----------|
| **pytest** | Test runner chinh |
| **pytest-asyncio** | Ho tro async tests (`asyncio_mode = "auto"`) |
| **pytest-cov** | Code coverage |
| **unittest.mock** | Mocking services |
| **FastAPI TestClient** | API integration tests |

## 7.2. Test Configuration (`tests/conftest.py`)

### Environment Fixtures

```python
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    monkeypatch.setenv("MEMORY_DB_PATH", "data/test_memory.db")
    monkeypatch.setenv("DEBUG", "true")
```

### Mock Fixtures

**mock_llm_service:** Mock LLM de khong goi API that

```python
@pytest.fixture
def mock_llm_service():
    with patch("src.services.llm.LLMService") as mock:
        instance = MagicMock()
        instance.chat_completion.return_value = "This is a test response."
        instance.achat_completion.return_value = "This is an async test response."
        mock.return_value = instance
        yield instance
```

**mock_embedding_service:** Mock embedding de khong tai model

```python
@pytest.fixture
def mock_embedding_service():
    with patch("src.services.embedding.EmbeddingService") as mock:
        instance = MagicMock()
        instance.dimension = 768
        instance.embed_text.return_value = [0.1] * 768
        instance.embed_query.return_value = [0.1] * 768
        mock.return_value = instance
        yield instance
```

**mock_vector_store:** Mock Qdrant

### Sample Data Fixtures

```python
@pytest.fixture
def sample_paper():
    return Paper(
        arxiv_id="2301.12345",
        title="Test Paper: A Study of Testing",
        abstract="This paper presents...",
        authors=["Alice Tester", "Bob Debugger"],
        categories=["cs.SE", "cs.LG"],
        citation_count=42,
    )

@pytest.fixture
def sample_papers():
    # 3 papers: Attention Is All You Need, BERT, GPT-3
    ...
```

### Memory Fixtures

```python
@pytest.fixture
def working_memory():
    return WorkingMemory(max_size=10)

@pytest.fixture
def episodic_memory_store(tmp_path):
    return EpisodicMemoryStore(db_path=str(tmp_path / "test_memory.db"))

@pytest.fixture
def belief_memory_store(tmp_path):
    return BeliefMemoryStore(db_path=str(tmp_path / "test_memory.db"))
```

## 7.3. Test Categories

### Unit Tests

| File | Test | Mo ta |
|------|------|-------|
| `test_models.py` | Paper models | Computed fields, validation |
| `test_models.py` | Memory models | MemoryType, EpisodicMemory, BeliefMemory |
| `test_services.py` | EmbeddingService | Mock SPECTER2 |
| `test_services.py` | VectorStore | Mock Qdrant |
| `test_services.py` | LLMService | Mock Groq API |

### Integration Tests

| File | Test | Mo ta |
|------|------|-------|
| `test_memory.py` | WorkingMemory | Session management, messages, steps |
| `test_memory.py` | EpisodicMemoryStore | SQLite operations (tmp_path) |
| `test_memory.py` | BeliefMemoryStore | Confidence, reinforcement, decay |
| `test_agent.py` | PaperLensAgent | ReAct loop voi mock services |
| `test_data_loader.py` | HuggingFaceDataLoader | Paper parsing |

### API Tests

```python
@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)
```

## 7.4. CI Test Pipeline

```yaml
test:
  runs-on: ubuntu-latest
  services:
    qdrant:
      image: qdrant/qdrant:latest
      ports:
        - 6333:6333

  steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.11
    - name: Cache pip packages
    - name: Install dependencies (pip install -e ".[dev]")
    - name: Wait for Qdrant (30 attempts, 2s interval)
    - name: Run tests
      env:
        GROQ_API_KEY: ${{ secrets.GROQ_API_KEY || 'test-key' }}
        QDRANT_HOST: localhost
        QDRANT_PORT: 6333
        MEMORY_DB_PATH: data/test_memory.db
      run: pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
    - name: Upload coverage (Codecov)
```

## 7.5. Coverage

```
pytest tests/ -v --cov=src --cov-report=term-missing
```

Coverage duoc upload len Codecov trong CI pipeline.

---

# 8. TRIEN KHAI (DEPLOYMENT)

## 8.1. Docker Architecture

### 8.1.1. Multi-stage Dockerfile (`docker/Dockerfile`)

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl
COPY pyproject.toml README.md ./
RUN pip wheel --wheel-dir /wheels -e .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get install -y curl
RUN useradd --create-home --shell /bin/bash appuser   # Non-root user
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
COPY src/ frontend/ scripts/ ./
USER appuser                                           # Chay voi non-root
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**Dac diem:**
- Multi-stage build giam kich thuoc image (khong co build tools trong runtime)
- Non-root user (`appuser`) tang bao mat
- Health check tu dong
- PYTHONUNBUFFERED=1 cho logging tot hon trong container

### 8.1.2. Docker Compose (`docker-compose.yml`)

3 services:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_data:/qdrant/storage]
    restart: unless-stopped

  api:
    build: {context: ., dockerfile: docker/Dockerfile}
    ports: ["8000:8000"]
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - QDRANT_HOST=qdrant           # Service name trong Docker network
      - EMBEDDING_MODEL=sentence-transformers/allenai-specter
    volumes:
      - ./src:/app/src               # Live reload
      - model_cache:/root/.cache     # Cache SPECTER2 model
    depends_on: [qdrant]

  frontend:
    build: {context: ., dockerfile: docker/Dockerfile}
    command: streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    environment:
      - API_URL=http://api:8000
      - QDRANT_HOST=qdrant
    depends_on: [api]

volumes:
  qdrant_data: {driver: local}
  model_cache: {driver: local}

networks:
  default:
    name: paperlens-network
```

**Named Volumes:**
- `qdrant_data` - Persist vector data qua cac lan restart
- `model_cache` - Cache SPECTER2 model (tranh tai lai)

**Network:**
- Tu dong tao `paperlens-network`
- Services giao tiep qua service names (qdrant, api)

## 8.2. CI/CD Pipeline

### 8.2.1. GitHub Actions (`ci.yml`)

5 jobs:

```
┌─────────┐   ┌────────────┐   ┌────────┐
│  Lint    │   │ Type Check │   │  Test  │
│  (ruff)  │   │   (mypy)   │   │(pytest)│
└────┬────┘   └────────────┘   └───┬────┘
     │                              │
     └──────────┬───────────────────┘
                │
     ┌──────────┴──────────┐
     │                      │
┌────┴─────┐   ┌───────────┴──────────┐
│  Build   │   │    Docker Build       │
│ (wheel)  │   │ (chi push to main)    │
└──────────┘   └──────────────────────┘
```

#### Job 1: Lint

```yaml
lint:
  runs-on: ubuntu-latest
  steps:
    - pip install ruff
    - ruff check .
```

**Ruff configuration** (trong `pyproject.toml`):
- Target: Python 3.11
- Line length: 100
- Rules: E (pycodestyle), W (warnings), F (pyflakes), I (isort), B (bugbear), C4 (comprehensions), UP (pyupgrade)
- Ignore: E501 (line too long), B008 (function calls in defaults), B905 (zip without strict)
- Per-file ignores: `scripts/*` va `frontend/*` - cho phep E402 (imports after sys.path)

#### Job 2: Type Check

```yaml
type-check:
  runs-on: ubuntu-latest
  steps:
    - pip install mypy types-requests
    - pip install -e ".[dev]"
    - mypy src --ignore-missing-imports
```

**mypy configuration:**
- Python 3.11
- `warn_return_any = true`
- `warn_unused_ignores = true`
- `disallow_untyped_defs = true`

#### Job 3: Test

```yaml
test:
  runs-on: ubuntu-latest
  services:
    qdrant: {image: qdrant/qdrant:latest, ports: [6333:6333]}
  steps:
    - pip install -e ".[dev]"
    - Wait for Qdrant (health check loop)
    - pytest tests/ -v --cov=src --cov-report=xml
    - Upload coverage to Codecov
```

#### Job 4: Build

```yaml
build:
  needs: [lint, test]
  steps:
    - pip install build
    - python -m build        # Tao wheel package
    - Upload artifact: dist/
```

#### Job 5: Docker Build

```yaml
docker:
  needs: [lint, test]
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - docker/setup-buildx-action@v3
    - docker/build-push-action@v5
      with:
        context: .
        file: docker/Dockerfile
        push: false
        tags: paperlens:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

**Triggers:**
- Push to `main` hoac `develop`: Chay tat ca 5 jobs
- Pull Request to `main`: Chay lint, type-check, test, build (khong docker)

## 8.3. Development Commands (Makefile)

| Command | Mo ta |
|---------|-------|
| `make install` | Install dependencies |
| `make dev-install` | Install voi dev dependencies + pre-commit |
| `make dev` | Chay ca API va Frontend song song |
| `make api` | Chi chay API (uvicorn, port 8000, reload) |
| `make frontend` | Chi chay Streamlit (port 8501) |
| `make test` | Chay tests voi coverage |
| `make test-fast` | Chay tests nhanh (no coverage, stop on first failure) |
| `make lint` | Ruff check + mypy |
| `make format` | Ruff format + auto-fix |
| `make up` | Docker compose up |
| `make down` | Docker compose down |
| `make logs` | Docker compose logs |
| `make build` | Docker compose build |
| `make qdrant` | Chi chay Qdrant container |
| `make index` | Index papers vao Qdrant |
| `make clean` | Xoa cache va build files |

## 8.4. Environment Variables (`.env.example`)

```bash
# LLM
GROQ_API_KEY=your-groq-api-key-here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding
EMBEDDING_MODEL=sentence-transformers/allenai-specter

# API
API_HOST=0.0.0.0
API_PORT=8000

# Memory
MEMORY_DB_PATH=data/memory.db

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

---

# 9. HUONG PHAT TRIEN TUONG LAI

## 9.1. Tinh Nang Moi

### 9.1.1. Full-Text PDF Processing
- Tai va phan tich toan van bai bao (khong chi abstract)
- Su dung PDF parser de trich xuat sections (Introduction, Methods, Results)
- Embedding tung section de search chinh xac hon

### 9.1.2. Citation Graph Analysis
- Xay dung do thi trieu dan (citation graph)
- Tim kiem bai bao anh huong nhat trong mot linh vuc
- Truc quan hoa moi quan he giua cac bai bao

### 9.1.3. Multi-Modal Search
- Tim kiem bang hinh anh (figures, diagrams)
- Tim kiem bang cong thuc toan hoc
- Tim kiem bang code snippets

### 9.1.4. Collaborative Features
- Chia se danh sach reading lists
- Annotation va note chia se
- Group search sessions

### 9.1.5. Real-time Updates
- Theo doi bai bao moi tren ArXiv
- Thong bao khi co bai bao moi phu hop voi interests
- Auto-update index

## 9.2. Cai Thien Ky Thuat

### 9.2.1. Async Agent
- Chuyen PaperLensAgent sang hoan toan async
- Loai bo `_run_async()` bridge
- Ho tro concurrent tool execution

### 9.2.2. Streaming ReAct
- Stream reasoning steps real-time den UI
- Hien thi "thinking" process cho nguoi dung
- True SSE streaming thay vi chunked response

### 9.2.3. Better Embedding Models
- Nang cap len SPECTER2+ hoac cac model moi hon
- Fine-tune embedding cho domain cụ the
- Multi-vector retrieval (title + abstract rieng)

### 9.2.4. Advanced Retrieval
- Hybrid search: ket hop semantic + keyword (BM25)
- Re-ranking voi cross-encoder
- Query expansion
- Multi-hop retrieval

### 9.2.5. Memory Improvements
- Vector-based episodic recall (thay vi SQLite LIKE search)
- Graph-based belief network
- Long-term memory compression
- Cross-session learning

### 9.2.6. Scalability
- Qdrant cluster mode cho large-scale deployment
- Redis caching layer
- API rate limiting va authentication
- Horizontal scaling voi multiple API instances

## 9.3. Production Readiness

| Yeu cau | Trang thai | Ghi chu |
|---------|-----------|---------|
| Authentication | Chua co | Can them JWT/OAuth |
| Rate Limiting | Chua co | Can them cho API |
| CORS Config | `allow_origins=["*"]` | Can config cụ the cho production |
| Logging | structlog | Tot, can them log rotation |
| Monitoring | Chua co | Can them Prometheus/Grafana |
| Error Tracking | Chua co | Can them Sentry |
| Database Backup | Chua co | Can backup strategy cho SQLite va Qdrant |
| Load Testing | Chua co | Can benchmark voi k6/locust |

---

# 10. BAI HOC KINH NGHIEM

## 10.1. Architecture Decisions

### 10.1.1. Tai Sao Chon Agentic RAG Thay Vi Standard RAG

**Standard RAG:** Query → Retrieve → Generate. Don gian nhung han che:
- Khong the xu ly queries phuc tap (so sanh, tim lien quan)
- Khong co kha nang multi-step reasoning
- Khong co memory hoac personalization

**Agentic RAG:** Cho phep agent tu suy luan va chon tools phu hop:
- Linh hoat: co the xu ly nhieu loai queries
- Extensible: de dang them tools moi
- Personalized: hoc tu tuong tac

**Trade-off:** Cham hon (nhieu LLM calls) nhung manh hon.

### 10.1.2. Tai Sao 4 Loai Memory

Lay cam hung tu Cognitive Science:
- **Semantic:** Kien thuc dai han (nhu kien thuc chung cua nguoi)
- **Episodic:** Ky uc su kien (nhu nho nhung gi da lam)
- **Working:** Bo nho lam viec (nhu suy nghi hien tai)
- **Belief:** Niem tin/so thich (nhu taste cua nguoi)

Moi loai co muc dich rieng va khong the thay the bang loai khac.

### 10.1.3. Tai Sao SPECTER2

General-purpose embeddings (nhu OpenAI, Cohere) khong hieu tot ngu canh hoc thuat:
- "Attention Is All You Need" - model chung se hieu ve "attention" nhu su chu y noi chung
- SPECTER2 hieu day la ve "self-attention mechanism in neural networks"

**Trade-off:** Model chuyen biet nhung khong phu hop cho general text.

### 10.1.4. Tai Sao SQLite Cho Memory

- Don gian, khong can setup server rieng
- Tot cho single-instance deployment
- Async ho tro qua aiosqlite
- Du nhanh cho scale hien tai

**Khi nao can chuyen:** Khi co nhieu users dong thoi hoac can distributed deployment → chuyen sang PostgreSQL hoac Redis.

## 10.2. Design Patterns

### 10.2.1. Singleton Pattern
- Dung cho tat ca services (embedding, vector store, LLM, memory stores)
- Tranh tai lai model va ket noi lai database moi request
- `lru_cache` cho Settings

### 10.2.2. Lazy Loading
- EmbeddingService: model chi tai khi lan dau goi `embed_text()`
- VectorStore: client chi connect khi lan dau goi method
- MemoryManager: cac stores chi init khi truy cap property

### 10.2.3. Factory Pattern
- `create_default_registry()`: tao ToolRegistry voi 6 tools
- `get_agent()`: tao Agent singleton

### 10.2.4. Strategy Pattern
- `LLMService`: co the chuyen provider (Groq/OpenAI) qua config
- `EmbeddingService`: co the doi model qua config
- Summary styles: brief/detailed/technical

### 10.2.5. Observer Pattern (Implicit)
- Tool execution → ghi vao episodic memory
- Search results → cap nhat beliefs
- Feedback → reinforce preferences

## 10.3. Code Quality

### 10.3.1. Type Safety
- Pydantic v2 cho data validation
- Type hints o moi function
- mypy strict mode trong CI
- `computed_field` de tinh toan tu dong

### 10.3.2. Error Handling
- Custom exceptions: `AgentError`, `MaxIterationsError`, `ParseError`, `LLMError`, `LLMRateLimitError`
- Graceful degradation: neu memory recording fail, operation chinh van tiep tuc
- Try/except o tool execution: tra ve `ToolResult(success=False, error=...)` thay vi crash

### 10.3.3. Logging
- structlog cho structured logging
- Log levels: DEBUG cho chi tiet, INFO cho operations, WARNING/ERROR cho van de
- Context trong logs: session_id, query, tool_name

### 10.3.4. Configuration
- Tat ca magic numbers nam trong Settings
- Environment-based config (12-factor app)
- Defaults hop ly, de override

## 10.4. Nhung Dieu Nen Lam Tot Hon

1. **Async Agent:** Nen async tu dau, khong can bridge
2. **Dependency Injection:** Su dung DI framework thay vi manual singletons
3. **Testing:** Them integration tests voi Qdrant that (khong chi mock)
4. **API Versioning:** Them `/api/v1/` prefix
5. **Documentation:** OpenAPI docs tu dong (FastAPI cung cap, nhung can them examples)
6. **Security:** CORS config cụ the, API authentication, input sanitization
7. **Monitoring:** Prometheus metrics, request tracing

---

# 11. PHU LUC

## 11.1. Tat Ca Dependencies

### Production Dependencies

```toml
[project.dependencies]
# Core
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Data
datasets>=2.14.0
pandas>=2.0.0

# ML/Embeddings
sentence-transformers>=2.2.0
torch>=2.0.0

# Vector DB
qdrant-client>=1.6.0

# LLM
litellm>=1.0.0
groq>=0.4.0

# API
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
httpx>=0.24.0

# Frontend
streamlit>=1.28.0

# Utilities
tenacity>=8.2.0
structlog>=23.1.0
rich>=13.0.0

# Agent & Memory
aiosqlite>=0.19.0
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    pytest>=7.4.0
    pytest-asyncio>=0.21.0
    pytest-cov>=4.1.0
    ruff>=0.1.0
    mypy>=1.5.0
    pre-commit>=3.4.0
    ipython>=8.0.0
    jupyter>=1.0.0
]
```

## 11.2. API Reference

### Health

```
GET /health
Response: {status, version, memory}
```

### Search

```
POST /api/search
Body: {query, limit?, year_from?, year_to?, categories?, use_personalization?, session_id?}
Response: {papers: [{arxiv_id, title, abstract, authors, categories, year, score, pdf_url, arxiv_url}], total, query, personalized}

GET /api/search?query=...&limit=10&year_from=2023
Response: same as POST
```

### Chat

```
POST /api/chat
Body: {message, session_id?}
Response: {response, session_id, papers, steps_taken}

POST /api/chat/stream
Body: {message, session_id?}
Response: SSE stream

POST /api/chat/session
Response: {session_id}

GET /api/chat/session/{id}
Response: {session_id, message_count, papers_viewed, created_at}

GET /api/chat/session/{id}/history
Response: {session_id, messages: [{role, content}], total}

DELETE /api/chat/session/{id}
Response: {message}

GET /api/chat/tools
Response: {tools: [{name, description, parameters}]}
```

### Compare & Related

```
POST /api/compare
Body: {paper_ids: [string], aspects?: [string]}
Response: {comparison, papers, aspects}

GET /api/papers/{arxiv_id}/related?limit=5
Response: {source_paper, related_papers}
```

### Memory

```
POST /api/memory/feedback
Body: {arxiv_id, liked, session_id?}
Response: {success, message}

GET /api/memory/history?session_id=...&limit=20
Response: {queries: [{query, result_count, papers, timestamp}], total}

GET /api/memory/preferences
Response: {preferences: {favorite_categories, favorite_authors, interest_topics, reading_level}}
```

### Paper

```
GET /api/papers/{arxiv_id}
Response: {paper: {arxiv_id, title, abstract, authors, categories, year, citation_count, pdf_url, arxiv_url}}
```

## 11.3. Danh Sach Files va Dong Code

| Directory | Files | Mo ta |
|-----------|-------|-------|
| `src/config.py` | 1 file | Configuration management |
| `src/models/` | 3 files | Data models (Paper, Memory) |
| `src/services/` | 4 files | Core services (Embedding, VectorStore, LLM) |
| `src/clients/` | 2 files | External clients (HuggingFace) |
| `src/memory/` | 6 files | Memory system (4 stores + manager) |
| `src/agent/` | 5 files | Agent system (ReAct, tools, planner, prompts) |
| `src/api/` | 4 files | FastAPI application |
| `frontend/` | 1 file | Streamlit UI |
| `scripts/` | 1 file | Indexing script |
| `tests/` | 6 files | Test suite |
| **Tong** | **~35 source files** | |

## 11.4. Glossary

| Thuat ngu | Giai thich |
|-----------|-----------|
| **RAG** | Retrieval-Augmented Generation - Ket hop truy xuat thong tin va sinh van ban |
| **Agentic RAG** | RAG voi autonomous agent co kha nang reasoning va tool usage |
| **ReAct** | Reasoning + Acting - Pattern cho LLM agents |
| **SPECTER2** | Model embedding chuyen biet cho bai bao khoa hoc (Allen Institute for AI) |
| **Embedding** | Bieu dien vector so hoc cua van ban |
| **Cosine Similarity** | Phuong phap do do tuong tu giua 2 vectors |
| **Vector Database** | Co so du lieu toi uu cho tim kiem vector tuong tu |
| **Qdrant** | Vector database open-source |
| **LiteLLM** | Thu vien cung cap unified API cho nhieu LLM providers |
| **Groq** | LLM inference provider su dung LPU (Language Processing Unit) |
| **LLaMA** | Large Language Model cua Meta |
| **Token** | Don vi co ban cua text trong LLM processing |
| **Temperature** | Tham so dieu khien do ngau nhien cua LLM output |
| **Semantic Search** | Tim kiem dua tren y nghia, khong phai keyword |
| **Payload Index** | Index tren metadata trong Qdrant de filter nhanh |
| **HNSW** | Hierarchical Navigable Small World - Thuat toan ANN (Approximate Nearest Neighbor) |
| **Episodic Memory** | Bo nho luu tru su kien da xay ra |
| **Belief Memory** | Bo nho luu tru niem tin/so thich da hoc |
| **Working Memory** | Bo nho tam thoi cho phien hien tai |
| **Confidence Decay** | Su suy giam do tin cay theo thoi gian |
| **Reinforcement** | Tang cuong confidence khi co bang chung moi |
| **SSE** | Server-Sent Events - Protocol cho streaming data |
| **CORS** | Cross-Origin Resource Sharing |
| **FastAPI** | Web framework Python hieu suat cao |
| **Streamlit** | Framework Python de tao web app nhanh |
| **Pre-commit Hook** | Script chay tu dong truoc moi git commit |
| **Ruff** | Linter va formatter Python nhanh |
| **mypy** | Static type checker cho Python |

## 11.5. Cac Lenh Huu Ich

### Development

```bash
# Setup
make dev-install                    # Install tat ca dependencies
cp .env.example .env               # Tao file env
# Sua .env: them GROQ_API_KEY

# Chay
make qdrant                        # Khoi dong Qdrant
make index                         # Index papers
make dev                           # Chay API + Frontend

# Quality
make lint                          # Kiem tra code
make format                        # Format code
make test                          # Chay tests
make test-fast                     # Tests nhanh (no coverage)
```

### Docker

```bash
# Toan bo he thong
docker compose up -d               # Khoi dong
docker compose logs -f             # Xem logs
docker compose down                # Dung

# Chi Qdrant
docker compose up -d qdrant
```

### Indexing

```bash
python scripts/index_papers.py --limit 1000
python scripts/index_papers.py --recreate --limit 5000
python scripts/index_papers.py --verify
python scripts/index_papers.py --categories cs.CL cs.LG
```

### Debug

```bash
# Test embedding
python -c "from src.services.embedding import EmbeddingService; s = EmbeddingService(); print(s.embed_text('test')[:5])"

# Test Qdrant connection
python -c "from src.services.vector_store import VectorStore; s = VectorStore(); print(s.get_collection_info())"

# Test LLM
python -c "from src.services.llm import LLMService; s = LLMService(); print(s.chat_completion([{'role': 'user', 'content': 'Hi'}]))"

# Test agent
python src/agent/agent.py "Find papers about transformers"
```

---

*Tai lieu nay duoc tao tu dong bang cach phan tich toan bo source code cua du an PaperLens. Moi section duoc xay dung tu viec doc va hieu chi tiet tung file trong codebase.*

*Cap nhat lan cuoi: 2026-02-12*
