"""
Search routes for PaperLens API.

Provides direct search and comparison endpoints that bypass the agent.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import settings
from src.memory.manager import get_memory_manager
from src.services.llm import get_llm_service

logger = structlog.get_logger()

router = APIRouter()


# =========================================================================
# Request/Response Models
# =========================================================================


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    year_from: int | None = Field(default=None, description="Filter from year")
    year_to: int | None = Field(default=None, description="Filter to year")
    categories: list[str] | None = Field(default=None, description="Filter by categories")
    use_personalization: bool = Field(default=True, description="Use user preferences")
    session_id: str | None = Field(default=None, description="Session for personalization")


class PaperResult(BaseModel):
    """A paper in search results."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    year: int | None
    score: float
    pdf_url: str
    arxiv_url: str


class SearchResponse(BaseModel):
    """Search response."""

    papers: list[PaperResult]
    total: int
    query: str
    personalized: bool


class CompareRequest(BaseModel):
    """Comparison request body."""

    paper_ids: list[str] = Field(..., min_length=2, max_length=5, description="Papers to compare")
    aspects: list[str] | None = Field(
        default=None,
        description="Aspects to compare (methodology, contributions, results)",
    )


class CompareResponse(BaseModel):
    """Comparison response."""

    comparison: str
    papers: list[dict[str, Any]]
    aspects: list[str]


# =========================================================================
# Search Endpoint
# =========================================================================


@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest) -> SearchResponse:
    """
    Search for papers by natural language query.

    Performs semantic search using SPECTER2 embeddings.
    Optionally personalizes results based on user preferences.
    """
    try:
        memory_manager = get_memory_manager()
        session_id = request.session_id or "anonymous"

        # Perform search with optional personalization
        if request.use_personalization and request.session_id:
            results = await memory_manager.search_papers(
                query=request.query,
                session_id=session_id,
                limit=request.limit,
                use_preferences=True,
                year_from=request.year_from,
                year_to=request.year_to,
                categories=request.categories,
            )
            personalized = True
        else:
            # Direct search without personalization
            results = memory_manager.semantic.search(
                query=request.query,
                limit=request.limit,
                year_from=request.year_from,
                year_to=request.year_to,
                categories=request.categories,
            )
            personalized = False

        # Record search to episodic memory
        await memory_manager.record_search(
            query=request.query,
            results=results,
            session_id=session_id,
        )

        # Format response
        papers = [
            PaperResult(
                arxiv_id=r.paper.arxiv_id,
                title=r.paper.title,
                abstract=r.paper.abstract,
                authors=r.paper.authors,
                categories=r.paper.categories,
                year=r.paper.year,
                score=round(r.score, 4),
                pdf_url=r.paper.pdf_url,
                arxiv_url=r.paper.arxiv_url,
            )
            for r in results
        ]

        logger.info(
            "Search completed",
            query=request.query[:30],
            results=len(papers),
            personalized=personalized,
        )

        return SearchResponse(
            papers=papers,
            total=len(papers),
            query=request.query,
            personalized=personalized,
        )

    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse)
async def search_papers_get(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50),
    year_from: int | None = Query(default=None),
    year_to: int | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> SearchResponse:
    """
    Search for papers (GET method).

    Simple search endpoint for quick queries.
    """
    request = SearchRequest(
        query=query,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        session_id=session_id,
        use_personalization=session_id is not None,
    )
    return await search_papers(request)


# =========================================================================
# Compare Endpoint
# =========================================================================


@router.post("/compare", response_model=CompareResponse)
async def compare_papers(request: CompareRequest) -> CompareResponse:
    """
    Compare multiple papers.

    Generates a detailed comparison of 2-5 papers on specified aspects.
    """
    try:
        memory_manager = get_memory_manager()
        llm = get_llm_service()

        # Retrieve papers
        papers = memory_manager.semantic.get_papers(request.paper_ids)

        if len(papers) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Could only find {len(papers)} of {len(request.paper_ids)} papers",
            )

        # Default aspects
        aspects = request.aspects or ["methodology", "contributions", "key_findings"]

        # Build comparison prompt
        papers_text = "\n\n".join([
            f"Paper {i+1}: {p.title}\n"
            f"ArXiv ID: {p.arxiv_id}\n"
            f"Year: {p.year}\n"
            f"Abstract: {p.abstract}"
            for i, p in enumerate(papers)
        ])

        aspects_text = ", ".join(aspects)

        prompt = f"""Compare the following academic papers on these aspects: {aspects_text}

{papers_text}

Provide a structured comparison that:
1. Identifies key similarities
2. Highlights important differences
3. Notes the progression of ideas if papers are related
4. Summarizes the unique contribution of each paper

Be concise but thorough. Use markdown formatting."""

        # Generate comparison
        messages = [
            {"role": "system", "content": "You are an expert at analyzing and comparing academic papers."},
            {"role": "user", "content": prompt},
        ]

        comparison = llm.chat_completion(messages, max_tokens=2000)

        logger.info("Comparison generated", paper_count=len(papers))

        return CompareResponse(
            comparison=comparison,
            papers=[
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "year": p.year,
                }
                for p in papers
            ],
            aspects=aspects,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Related Papers Endpoint
# =========================================================================


class RelatedResponse(BaseModel):
    """Response with related papers."""

    source_paper: dict[str, Any]
    related_papers: list[PaperResult]


@router.get("/papers/{arxiv_id}/related", response_model=RelatedResponse)
async def get_related_papers(
    arxiv_id: str,
    limit: int = Query(default=5, ge=1, le=20),
) -> RelatedResponse:
    """
    Find papers related to a given paper.

    Returns papers with similar content based on embedding similarity.
    """
    try:
        memory_manager = get_memory_manager()

        # Get source paper
        source = memory_manager.semantic.get_paper(arxiv_id)
        if not source:
            raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")

        # Find related
        results = memory_manager.semantic.find_related(
            arxiv_id=arxiv_id,
            limit=limit,
            exclude_self=True,
        )

        related = [
            PaperResult(
                arxiv_id=r.paper.arxiv_id,
                title=r.paper.title,
                abstract=r.paper.abstract,
                authors=r.paper.authors,
                categories=r.paper.categories,
                year=r.paper.year,
                score=round(r.score, 4),
                pdf_url=r.paper.pdf_url,
                arxiv_url=r.paper.arxiv_url,
            )
            for r in results
        ]

        logger.info("Related papers found", source=arxiv_id, count=len(related))

        return RelatedResponse(
            source_paper={
                "arxiv_id": source.arxiv_id,
                "title": source.title,
                "year": source.year,
            },
            related_papers=related,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to find related papers", arxiv_id=arxiv_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
