"""
Vector store service using Qdrant.

Handles storage and retrieval of paper embeddings.
"""

import hashlib
from collections.abc import Sequence

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.models.paper import Paper, PaperSearchResult

logger = structlog.get_logger()


def arxiv_id_to_point_id(arxiv_id: str) -> int:
    """Convert an ArXiv ID to a deterministic Qdrant point ID.

    Uses MD5 hash truncated to 15 hex chars, then converted to int.
    This must match the ID used during upsert.

    Args:
        arxiv_id: ArXiv paper ID (e.g., '2301.12345').

    Returns:
        Integer point ID for Qdrant.
    """
    return int(hashlib.md5(arxiv_id.encode()).hexdigest()[:15], 16)


class VectorStore:
    """Vector store for paper embeddings using Qdrant."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            host: Qdrant host. Defaults to config value.
            port: Qdrant port. Defaults to config value.
            collection_name: Collection name. Defaults to config value.
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection

        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        """Lazy load the Qdrant client."""
        if self._client is None:
            logger.info("Connecting to Qdrant", host=self.host, port=self.port)
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def create_collection(
        self,
        vector_size: int | None = None,
        recreate: bool = False,
    ) -> None:
        """
        Create the papers collection.

        Args:
            vector_size: Embedding dimension. Defaults to config value.
            recreate: If True, delete existing collection first.
        """
        vector_size = vector_size or settings.embedding_dimension

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            logger.warning("Deleting existing collection", collection=self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creating collection",
                collection=self.collection_name,
                vector_size=vector_size,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="categories",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info("Collection created", collection=self.collection_name)
        else:
            logger.info("Collection already exists", collection=self.collection_name)

    def upsert_papers(
        self,
        papers: Sequence[Paper],
        embeddings: Sequence[list[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update papers in the collection.

        Args:
            papers: Papers to insert.
            embeddings: Corresponding embeddings.
            batch_size: Batch size for upsert.

        Returns:
            Number of papers upserted.
        """
        if len(papers) != len(embeddings):
            raise ValueError("Papers and embeddings must have the same length")

        logger.info("Upserting papers", count=len(papers), batch_size=batch_size)

        points = []
        for paper, embedding in zip(papers, embeddings):
            # Generate a valid integer ID from arxiv_id using hash
            point_id = arxiv_id_to_point_id(paper.arxiv_id)
            point = models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "pdf_url": paper.pdf_url,
                    "arxiv_url": paper.arxiv_url,
                },
            )
            points.append(point)

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug("Batch upserted", batch_num=i // batch_size + 1)

        logger.info("Upsert complete", count=len(papers))
        return len(papers)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
    ) -> list[PaperSearchResult]:
        """
        Search for similar papers.

        Args:
            query_vector: Query embedding.
            limit: Maximum results to return.
            min_score: Minimum similarity score.
            year_from: Filter papers from this year onwards.
            year_to: Filter papers up to this year.
            categories: Filter by ArXiv categories.

        Returns:
            List of PaperSearchResult objects.
        """
        min_score = min_score if min_score is not None else settings.search_min_score

        # Build filter conditions
        must_conditions: list[models.FieldCondition | models.IsEmptyCondition | models.IsNullCondition | models.HasIdCondition | models.HasVectorCondition | models.NestedCondition | models.Filter] = []

        if year_from is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=year_from),
                )
            )

        if year_to is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(lte=year_to),
                )
            )

        if categories:
            must_conditions.append(
                models.FieldCondition(
                    key="categories",
                    match=models.MatchAny(any=categories),
                )
            )

        # Build filter
        query_filter = None
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)

        # Execute search using query_points (new API in qdrant-client 1.7+)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=min_score,
        )

        # Convert to PaperSearchResult
        search_results = []
        for result in response.points:
            payload = result.payload or {}
            paper = Paper(
                arxiv_id=payload["arxiv_id"],
                title=payload["title"],
                abstract=payload["abstract"],
                authors=payload.get("authors", []),
                categories=payload.get("categories", []),
                citation_count=payload.get("citation_count", 0),
            )
            search_results.append(
                PaperSearchResult(paper=paper, score=result.score)
            )

        logger.info("Search complete", results=len(search_results), limit=limit)
        return search_results

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except UnexpectedResponse:
            return {"name": self.collection_name, "status": "not_found"}

    def delete_collection(self) -> None:
        """Delete the collection."""
        logger.warning("Deleting collection", collection=self.collection_name)
        self.client.delete_collection(self.collection_name)

    def count(self) -> int:
        """Get number of papers in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except UnexpectedResponse:
            return 0


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


if __name__ == "__main__":
    # Quick test
    store = VectorStore()

    # Check connection
    info = store.get_collection_info()
    print(f"Collection info: {info}")
