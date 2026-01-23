"""
Semantic Memory for PaperLens.

Wraps the vector store to provide a unified interface for paper knowledge.
Handles embedding, search, and retrieval of papers.
"""

from typing import Any

import structlog

from src.config import settings
from src.models.paper import Paper, PaperSearchResult
from src.services.embedding import EmbeddingService, get_embedding_service
from src.services.vector_store import VectorStore, get_vector_store

logger = structlog.get_logger()


class SemanticMemory:
    """
    Semantic memory for paper knowledge.

    Provides a high-level interface for:
    - Searching papers by natural language query
    - Retrieving papers by ID
    - Finding related papers
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """
        Initialize semantic memory.

        Args:
            vector_store: Vector store instance. Uses singleton if None.
            embedding_service: Embedding service instance. Uses singleton if None.
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service

        logger.info("Semantic memory initialized")

    @property
    def vector_store(self) -> VectorStore:
        """Get vector store (lazy load)."""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get embedding service (lazy load)."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def search(
        self,
        query: str,
        limit: int | None = None,
        min_score: float | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
    ) -> list[PaperSearchResult]:
        """
        Search for papers by natural language query.

        Args:
            query: Search query in natural language.
            limit: Maximum results to return.
            min_score: Minimum similarity score (0-1).
            year_from: Filter papers from this year.
            year_to: Filter papers up to this year.
            categories: Filter by ArXiv categories (e.g., ['cs.CL', 'cs.LG']).

        Returns:
            List of PaperSearchResult ordered by relevance.
        """
        limit = limit or settings.search_top_k

        logger.info(
            "Searching papers",
            query=query[:50] + "..." if len(query) > 50 else query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            categories=categories,
        )

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            min_score=min_score,
            year_from=year_from,
            year_to=year_to,
            categories=categories,
        )

        logger.info("Search complete", results_count=len(results))
        return results

    def get_paper(self, arxiv_id: str) -> Paper | None:
        """
        Retrieve a specific paper by ArXiv ID.

        Args:
            arxiv_id: Paper ArXiv ID.

        Returns:
            Paper if found, None otherwise.
        """
        logger.debug("Retrieving paper", arxiv_id=arxiv_id)

        # Search for exact match using the paper's own embedding
        # We use the arxiv_id as the search query since it's stored in payload
        try:
            # Convert arxiv_id to Qdrant point ID format
            point_id = arxiv_id.replace(".", "_").replace("/", "_")

            # Retrieve the point directly
            points = self.vector_store.client.retrieve(
                collection_name=self.vector_store.collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if not points:
                logger.warning("Paper not found", arxiv_id=arxiv_id)
                return None

            payload = points[0].payload
            paper = Paper(
                arxiv_id=payload["arxiv_id"],
                title=payload["title"],
                abstract=payload["abstract"],
                authors=payload.get("authors", []),
                categories=payload.get("categories", []),
                citation_count=payload.get("citation_count", 0),
            )

            logger.debug("Paper retrieved", arxiv_id=arxiv_id, title=paper.title[:50])
            return paper

        except Exception as e:
            logger.error("Failed to retrieve paper", arxiv_id=arxiv_id, error=str(e))
            return None

    def get_papers(self, arxiv_ids: list[str]) -> list[Paper]:
        """
        Retrieve multiple papers by ArXiv ID.

        Args:
            arxiv_ids: List of ArXiv IDs.

        Returns:
            List of found papers (may be shorter if some not found).
        """
        papers = []
        for arxiv_id in arxiv_ids:
            paper = self.get_paper(arxiv_id)
            if paper:
                papers.append(paper)
        return papers

    def find_related(
        self,
        arxiv_id: str,
        limit: int | None = None,
        exclude_self: bool = True,
    ) -> list[PaperSearchResult]:
        """
        Find papers related to a given paper.

        Args:
            arxiv_id: Source paper ArXiv ID.
            limit: Maximum related papers to return.
            exclude_self: Whether to exclude the source paper from results.

        Returns:
            List of related papers with similarity scores.
        """
        limit = limit or settings.search_top_k

        logger.info("Finding related papers", arxiv_id=arxiv_id, limit=limit)

        # Get the source paper
        paper = self.get_paper(arxiv_id)
        if not paper:
            logger.warning("Source paper not found", arxiv_id=arxiv_id)
            return []

        # Search for similar papers using the paper's text
        search_text = paper.to_embedding_text()
        query_embedding = self.embedding_service.embed_query(search_text)

        # Increase limit if excluding self
        search_limit = limit + 1 if exclude_self else limit

        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=search_limit,
        )

        # Filter out the source paper if requested
        if exclude_self:
            results = [r for r in results if r.paper.arxiv_id != arxiv_id][:limit]

        logger.info("Found related papers", count=len(results))
        return results

    def find_by_embedding(
        self,
        embedding: list[float],
        limit: int | None = None,
        min_score: float | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
    ) -> list[PaperSearchResult]:
        """
        Search for papers by pre-computed embedding.

        Useful for episodic memory recall where we already have
        stored query embeddings.

        Args:
            embedding: Pre-computed query embedding.
            limit: Maximum results to return.
            min_score: Minimum similarity score.
            year_from: Filter by year (from).
            year_to: Filter by year (to).
            categories: Filter by categories.

        Returns:
            List of matching papers.
        """
        limit = limit or settings.search_top_k

        return self.vector_store.search(
            query_vector=embedding,
            limit=limit,
            min_score=min_score,
            year_from=year_from,
            year_to=year_to,
            categories=categories,
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the paper index.

        Returns:
            Dict with index statistics.
        """
        info = self.vector_store.get_collection_info()
        return {
            "total_papers": info.get("points_count", 0),
            "status": info.get("status", "unknown"),
            "collection_name": info.get("name"),
        }

    def is_available(self) -> bool:
        """
        Check if semantic memory is available.

        Returns:
            True if vector store is accessible.
        """
        try:
            info = self.vector_store.get_collection_info()
            return info.get("status") != "not_found"
        except Exception as e:
            logger.error("Semantic memory not available", error=str(e))
            return False


# Singleton instance
_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    """Get or create the semantic memory singleton."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory


if __name__ == "__main__":
    # Quick test
    memory = SemanticMemory()

    print(f"Available: {memory.is_available()}")
    print(f"Stats: {memory.get_stats()}")

    # Test search (requires indexed papers)
    results = memory.search("transformer attention mechanism", limit=3)
    for r in results:
        print(f"\n{r.paper.title} (score: {r.score:.3f})")
