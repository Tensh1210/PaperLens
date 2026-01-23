"""
FastAPI application for PaperLens.

Provides REST API endpoints for:
- Paper search and retrieval
- Agentic chat interface
- Memory management
- Health checks
"""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.routes import chat, search
from src.config import settings
from src.memory.manager import get_memory_manager

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting PaperLens API", host=settings.api_host, port=settings.api_port)

    # Initialize memory manager
    memory_manager = get_memory_manager()
    logger.info("Memory manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down PaperLens API")


# Create FastAPI app
app = FastAPI(
    title="PaperLens API",
    description="Agentic RAG-based ML Paper Search & Comparison Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


# =========================================================================
# Health & Status Endpoints
# =========================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    memory: dict[str, Any] = Field(default_factory=dict, description="Memory stats")


class StatsResponse(BaseModel):
    """Statistics response."""

    total_papers: int = Field(..., description="Total papers in index")
    memory_stats: dict[str, Any] = Field(..., description="Memory statistics")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check API health status.

    Returns service status and basic memory statistics.
    """
    try:
        memory_manager = get_memory_manager()
        semantic_stats = memory_manager.semantic.get_stats()

        return HealthResponse(
            status="ok",
            version="0.1.0",
            memory={
                "semantic": semantic_stats,
                "working_sessions": len(memory_manager.working.list_sessions()),
            },
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            memory={"error": str(e)},
        )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get detailed statistics about the system.

    Returns paper count, memory stats, and more.
    """
    try:
        memory_manager = get_memory_manager()
        stats = await memory_manager.get_stats()

        return StatsResponse(
            total_papers=stats.get("semantic", {}).get("total_papers", 0),
            memory_stats=stats,
        )
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Memory Endpoints
# =========================================================================


class FeedbackRequest(BaseModel):
    """Request to submit paper feedback."""

    arxiv_id: str = Field(..., description="Paper ArXiv ID")
    liked: bool = Field(..., description="Whether the paper was liked")
    session_id: str | None = Field(default=None, description="Session ID")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    success: bool
    message: str


@app.post("/api/memory/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit feedback on a paper.

    Records whether user liked or disliked a paper.
    """
    try:
        memory_manager = get_memory_manager()
        await memory_manager.record_feedback(
            arxiv_id=request.arxiv_id,
            liked=request.liked,
            session_id=request.session_id or "anonymous",
        )

        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded for {request.arxiv_id}",
        )
    except Exception as e:
        logger.error("Failed to record feedback", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class HistoryResponse(BaseModel):
    """Response with search history."""

    queries: list[dict[str, Any]]
    total: int


@app.get("/api/memory/history", response_model=HistoryResponse)
async def get_history(
    session_id: str | None = None,
    limit: int = 20,
) -> HistoryResponse:
    """
    Get search history.

    Returns recent search queries and results.
    """
    try:
        memory_manager = get_memory_manager()
        memories = await memory_manager.episodic.get_recent(
            limit=limit,
            session_id=session_id,
        )

        queries = [
            {
                "query": m.query,
                "result_count": m.result_count,
                "papers": m.result_paper_ids[:5],
                "timestamp": m.created_at.isoformat(),
            }
            for m in memories
        ]

        return HistoryResponse(queries=queries, total=len(queries))
    except Exception as e:
        logger.error("Failed to get history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class PreferencesResponse(BaseModel):
    """Response with user preferences."""

    preferences: dict[str, Any]


@app.get("/api/memory/preferences", response_model=PreferencesResponse)
async def get_preferences() -> PreferencesResponse:
    """
    Get learned user preferences.

    Returns favorite categories, topics, and authors.
    """
    try:
        memory_manager = get_memory_manager()
        prefs = await memory_manager.belief.get_preferences_summary()

        return PreferencesResponse(preferences=prefs)
    except Exception as e:
        logger.error("Failed to get preferences", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Paper Endpoint
# =========================================================================


class PaperResponse(BaseModel):
    """Response with paper details."""

    paper: dict[str, Any]


@app.get("/api/papers/{arxiv_id}", response_model=PaperResponse)
async def get_paper(arxiv_id: str) -> PaperResponse:
    """
    Get details of a specific paper.

    Args:
        arxiv_id: ArXiv ID of the paper.
    """
    try:
        memory_manager = get_memory_manager()
        paper = memory_manager.semantic.get_paper(arxiv_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        # Record view
        await memory_manager.record_view(arxiv_id, "api")

        return PaperResponse(
            paper={
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "categories": paper.categories,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.arxiv_url,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get paper", arxiv_id=arxiv_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Run with uvicorn
# =========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
