"""
Tests for PaperLens services.
"""

from unittest.mock import patch

import pytest


class TestEmbeddingService:
    """Tests for the embedding service."""

    def test_embed_text(self, mock_embedding_service):
        """Test embedding a single text."""
        embedding = mock_embedding_service.embed_text("test text")

        assert len(embedding) == 768
        assert isinstance(embedding[0], float)

    def test_embed_texts(self, mock_embedding_service):
        """Test embedding multiple texts."""
        texts = ["text 1", "text 2", "text 3"]
        embeddings = mock_embedding_service.embed_texts(texts)

        assert len(embeddings) == 1  # Mock returns single list
        assert len(embeddings[0]) == 768

    def test_embed_paper(self, mock_embedding_service, sample_paper):
        """Test embedding a paper."""
        embedding = mock_embedding_service.embed_paper(sample_paper)

        assert len(embedding) == 768

    def test_embed_query(self, mock_embedding_service):
        """Test embedding a search query."""
        embedding = mock_embedding_service.embed_query("transformer attention")

        assert len(embedding) == 768

    def test_similarity_calculation(self):
        """Test cosine similarity calculation."""
        from src.services.embedding import EmbeddingService

        service = EmbeddingService.__new__(EmbeddingService)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = service.similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, rel=0.01)

        vec3 = [0.0, 1.0, 0.0]
        similarity = service.similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0, rel=0.01)


class TestVectorStore:
    """Tests for the vector store."""

    def test_vector_store_initialization(self, mock_vector_store):
        """Test vector store initialization."""
        assert mock_vector_store.collection_name == "test_papers"

    def test_collection_info(self, mock_vector_store):
        """Test getting collection info."""
        info = mock_vector_store.get_collection_info()

        assert "name" in info
        assert info["name"] == "test_papers"

    def test_count(self, mock_vector_store):
        """Test getting paper count."""
        count = mock_vector_store.count()
        assert count == 100

    def test_search_empty_results(self, mock_vector_store):
        """Test search with no results."""
        results = mock_vector_store.search([0.1] * 768)
        assert results == []


class TestLLMService:
    """Tests for the LLM service."""

    def test_chat_completion(self, mock_llm_service):
        """Test chat completion."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = mock_llm_service.chat_completion(messages)
        assert response == "This is a test response."

    def test_async_chat_completion(self, mock_llm_service):
        """Test async chat completion."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = mock_llm_service.achat_completion(messages)
        assert response == "This is an async test response."


class TestDataLoader:
    """Tests for the data loader."""

    def test_parse_authors_string(self):
        """Test parsing authors from string."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        # Comma-separated
        authors = loader._parse_authors("Alice, Bob, Charlie")
        assert authors == ["Alice", "Bob", "Charlie"]

        # And-separated
        authors = loader._parse_authors("Alice and Bob")
        assert authors == ["Alice", "Bob"]

        # Single author
        authors = loader._parse_authors("Alice")
        assert authors == ["Alice"]

    def test_parse_authors_list(self):
        """Test parsing authors from list."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        authors = loader._parse_authors(["Alice", "Bob"])
        assert authors == ["Alice", "Bob"]

    def test_parse_paper_valid(self):
        """Test parsing a valid paper item."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        item = {
            "id": "2301.12345",
            "title": "Test Paper",
            "abstract": "This is a test abstract.",
            "authors": "Alice, Bob",
            "categories": "cs.LG cs.AI",
        }

        paper = loader._parse_paper(item)

        assert paper is not None
        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert len(paper.categories) == 2

    def test_parse_paper_missing_id(self):
        """Test parsing paper without ID generates an ID."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        item = {
            "title": "Test Paper",
            "abstract": "This is a test abstract.",
        }

        paper = loader._parse_paper(item)
        assert paper is not None
        assert paper.arxiv_id.startswith("paper_")
        assert paper.title == "Test Paper"

    def test_parse_categories_string(self):
        """Test parsing categories from string."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        item = {
            "id": "2301.12345",
            "title": "Test",
            "abstract": "Test",
            "categories": "cs.LG cs.AI cs.CL",
        }

        paper = loader._parse_paper(item)
        assert paper.categories == ["cs.LG", "cs.AI", "cs.CL"]

    def test_parse_categories_list(self):
        """Test parsing categories from list."""
        from src.clients.data_loader import HuggingFaceDataLoader

        loader = HuggingFaceDataLoader()

        item = {
            "id": "2301.12345",
            "title": "Test",
            "abstract": "Test",
            "categories": ["cs.LG", "cs.AI"],
        }

        paper = loader._parse_paper(item)
        assert paper.categories == ["cs.LG", "cs.AI"]


class TestConfig:
    """Tests for configuration."""

    def test_settings_defaults(self):
        """Test default settings."""
        from src.config import Settings

        # Create settings with minimal config (no .env file to avoid overrides)
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}, clear=False):
            settings = Settings(_env_file=None)

            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333
            assert settings.embedding_model == "allenai/specter2_base"
            assert settings.embedding_dimension == 768

    def test_qdrant_url_property(self):
        """Test computed qdrant_url property."""
        from src.config import Settings

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            settings = Settings()

            assert settings.qdrant_url == "http://localhost:6333"

    def test_llm_full_model_groq(self):
        """Test LLM model name for Groq."""
        from src.config import Settings

        with patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key",
            "LLM_PROVIDER": "groq",
        }):
            settings = Settings()

            assert "groq/" in settings.llm_full_model

    def test_llm_full_model_openai(self):
        """Test LLM model name for OpenAI."""
        from src.config import Settings

        with patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4",
        }):
            settings = Settings()

            assert settings.llm_full_model == "gpt-4"
