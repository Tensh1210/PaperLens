"""
Embedding service using SPECTER2.

SPECTER2 is specifically designed for scientific papers and provides
high-quality embeddings for academic text.
"""

from collections.abc import Sequence

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models.paper import Paper

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating paper embeddings using SPECTER2."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Model to use. Defaults to config value.
        """
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Model loaded",
                model=self.model_name,
                dimension=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        dim = self.model.get_sentence_embedding_dimension()
        assert dim is not None
        return dim

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return list(embedding.tolist())

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embed multiple texts in batches.

        Args:
            texts: Texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            List of embedding vectors.
        """
        logger.info("Embedding texts", count=len(texts), batch_size=batch_size)

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return [list(e) for e in embeddings.tolist()]

    def embed_paper(self, paper: Paper) -> list[float]:
        """
        Embed a single paper.

        Uses title + abstract for best results with SPECTER2.

        Args:
            paper: Paper to embed.

        Returns:
            Embedding vector.
        """
        text = paper.to_embedding_text()
        return self.embed_text(text)

    def embed_papers(
        self,
        papers: Sequence[Paper],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Embed multiple papers.

        Args:
            papers: Papers to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            List of embedding vectors.
        """
        texts = [paper.to_embedding_text() for paper in papers]
        return self.embed_texts(texts, batch_size=batch_size, show_progress=show_progress)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.

        For SPECTER2, query embedding uses the same method as documents.

        Args:
            query: Search query.

        Returns:
            Query embedding vector.
        """
        return self.embed_text(query)

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity score (0 to 1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


if __name__ == "__main__":
    # Quick test
    service = EmbeddingService()

    # Test with sample text
    text = "Attention Is All You Need. We propose a new simple network architecture, the Transformer."
    embedding = service.embed_text(text)

    print(f"Model: {service.model_name}")
    print(f"Dimension: {service.dimension}")
    print(f"Embedding shape: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test similarity
    text2 = "The Transformer architecture uses self-attention mechanisms."
    embedding2 = service.embed_text(text2)

    similarity = service.similarity(embedding, embedding2)
    print(f"\nSimilarity between texts: {similarity:.4f}")
