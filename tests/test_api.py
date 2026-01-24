"""
Tests for PaperLens API endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, test_client):
        """Test health check returns OK."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.semantic.get_stats.return_value = {"total_papers": 100}
            mock.working.list_sessions.return_value = []
            mock_manager.return_value = mock

            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["ok", "degraded"]
            assert "version" in data


class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_papers_post(self, test_client):
        """Test POST search endpoint."""
        with patch("src.api.routes.search.get_memory_manager") as mock_manager:
            # Mock search results
            mock_paper = MagicMock()
            mock_paper.arxiv_id = "1706.03762"
            mock_paper.title = "Attention Is All You Need"
            mock_paper.abstract = "Test abstract"
            mock_paper.authors = ["Author 1"]
            mock_paper.categories = ["cs.CL"]
            mock_paper.year = 2017
            mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
            mock_paper.arxiv_url = "https://arxiv.org/abs/1706.03762"

            mock_result = MagicMock()
            mock_result.paper = mock_paper
            mock_result.score = 0.95

            mock = MagicMock()
            mock.semantic.search.return_value = [mock_result]
            mock.record_search = AsyncMock()
            mock_manager.return_value = mock

            response = test_client.post(
                "/api/search",
                json={"query": "transformers", "limit": 10}
            )

            assert response.status_code == 200
            data = response.json()
            assert "papers" in data
            assert data["query"] == "transformers"

    def test_search_papers_get(self, test_client):
        """Test GET search endpoint."""
        with patch("src.api.routes.search.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.semantic.search.return_value = []
            mock.record_search = AsyncMock()
            mock_manager.return_value = mock

            response = test_client.get("/api/search?query=transformers&limit=5")

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "transformers"

    def test_search_validation_error(self, test_client):
        """Test search with invalid input."""
        response = test_client.post("/api/search", json={})

        assert response.status_code == 422  # Validation error

    def test_compare_papers(self, test_client):
        """Test paper comparison endpoint."""
        with patch("src.api.routes.search.get_memory_manager") as mock_manager, \
             patch("src.api.routes.search.get_llm_service") as mock_llm:

            # Mock paper retrieval
            mock_paper = MagicMock()
            mock_paper.arxiv_id = "1706.03762"
            mock_paper.title = "Test Paper"
            mock_paper.abstract = "Test abstract"
            mock_paper.year = 2017

            mock = MagicMock()
            mock.semantic.get_papers.return_value = [mock_paper, mock_paper]
            mock_manager.return_value = mock

            # Mock LLM
            llm_mock = MagicMock()
            llm_mock.chat_completion.return_value = "Comparison: Both papers..."
            mock_llm.return_value = llm_mock

            response = test_client.post(
                "/api/compare",
                json={"paper_ids": ["1706.03762", "1810.04805"]}
            )

            assert response.status_code == 200
            data = response.json()
            assert "comparison" in data

    def test_compare_not_enough_papers(self, test_client):
        """Test comparison with insufficient papers."""
        response = test_client.post(
            "/api/compare",
            json={"paper_ids": ["1706.03762"]}
        )

        assert response.status_code == 422  # Validation error


class TestChatEndpoints:
    """Tests for chat endpoints."""

    def test_chat_endpoint(self, test_client):
        """Test chat endpoint."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            # Mock agent result
            mock_response = MagicMock()
            mock_response.response = "Here are the papers..."
            mock_response.session_id = "test-session"
            mock_response.papers = ["1706.03762"]
            mock_response.steps = []

            mock = MagicMock()
            mock.run.return_value = mock_response
            mock_agent.return_value = mock

            response = test_client.post(
                "/api/chat",
                json={"message": "Find papers about transformers"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "session_id" in data

    def test_chat_with_session(self, test_client):
        """Test chat with existing session."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            mock_response = MagicMock()
            mock_response.response = "Following up..."
            mock_response.session_id = "existing-session"
            mock_response.papers = []
            mock_response.steps = []

            mock = MagicMock()
            mock.run.return_value = mock_response
            mock_agent.return_value = mock

            response = test_client.post(
                "/api/chat",
                json={
                    "message": "What about GPT?",
                    "session_id": "existing-session"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "existing-session"

    def test_create_session(self, test_client):
        """Test session creation endpoint."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            mock = MagicMock()
            mock.memory.get_session.return_value = MagicMock()
            mock_agent.return_value = mock

            response = test_client.post("/api/chat/session")

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_get_session_info(self, test_client):
        """Test getting session info."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            mock_state = MagicMock()
            mock_state.messages = []
            mock_state.retrieved_paper_ids = ["1706.03762"]
            # Remove created_at so hasattr returns False
            del mock_state.created_at

            mock = MagicMock()
            mock.memory.session_exists.return_value = True
            mock.memory.get_session.return_value = mock_state
            mock_agent.return_value = mock

            response = test_client.get("/api/chat/session/test-session")

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-session"

    def test_get_session_not_found(self, test_client):
        """Test getting nonexistent session."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            mock = MagicMock()
            mock.memory.session_exists.return_value = False
            mock_agent.return_value = mock

            response = test_client.get("/api/chat/session/nonexistent")

            assert response.status_code == 404

    def test_get_available_tools(self, test_client):
        """Test getting available tools."""
        with patch("src.api.routes.chat.get_agent") as mock_agent:
            mock = MagicMock()
            mock.tools.get_schemas.return_value = [
                {
                    "function": {
                        "name": "search_papers",
                        "description": "Search for papers",
                        "parameters": {"properties": {"query": {}}}
                    }
                }
            ]
            mock_agent.return_value = mock

            response = test_client.get("/api/chat/tools")

            assert response.status_code == 200
            data = response.json()
            assert "tools" in data
            assert len(data["tools"]) >= 1


class TestMemoryEndpoints:
    """Tests for memory-related endpoints."""

    def test_submit_feedback(self, test_client):
        """Test submitting paper feedback."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.record_feedback = AsyncMock()
            mock_manager.return_value = mock

            response = test_client.post(
                "/api/memory/feedback",
                json={"arxiv_id": "1706.03762", "liked": True}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_get_history(self, test_client):
        """Test getting search history."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock_memory = MagicMock()
            mock_memory.query = "transformers"
            mock_memory.result_count = 5
            mock_memory.result_paper_ids = ["1706.03762"]
            mock_memory.created_at = MagicMock()
            mock_memory.created_at.isoformat.return_value = "2024-01-01T00:00:00"

            mock = MagicMock()
            mock.episodic.get_recent = AsyncMock(return_value=[mock_memory])
            mock_manager.return_value = mock

            response = test_client.get("/api/memory/history")

            assert response.status_code == 200
            data = response.json()
            assert "queries" in data

    def test_get_preferences(self, test_client):
        """Test getting user preferences."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.belief.get_preferences_summary = AsyncMock(return_value={
                "favorite_categories": [],
                "interest_topics": [],
            })
            mock_manager.return_value = mock

            response = test_client.get("/api/memory/preferences")

            assert response.status_code == 200
            data = response.json()
            assert "preferences" in data


class TestPaperEndpoint:
    """Tests for paper retrieval endpoint."""

    def test_get_paper(self, test_client):
        """Test getting a specific paper."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock_paper = MagicMock()
            mock_paper.arxiv_id = "1706.03762"
            mock_paper.title = "Attention Is All You Need"
            mock_paper.abstract = "Test abstract"
            mock_paper.authors = ["Author 1"]
            mock_paper.categories = ["cs.CL"]
            mock_paper.year = 2017
            mock_paper.citation_count = 50000
            mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
            mock_paper.arxiv_url = "https://arxiv.org/abs/1706.03762"

            mock = MagicMock()
            mock.semantic.get_paper.return_value = mock_paper
            mock.record_view = AsyncMock()
            mock_manager.return_value = mock

            response = test_client.get("/api/papers/1706.03762")

            assert response.status_code == 200
            data = response.json()
            assert data["paper"]["arxiv_id"] == "1706.03762"

    def test_get_paper_not_found(self, test_client):
        """Test getting nonexistent paper."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.semantic.get_paper.return_value = None
            mock_manager.return_value = mock

            response = test_client.get("/api/papers/nonexistent")

            assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    def test_get_stats(self, test_client):
        """Test getting system stats."""
        with patch("src.api.main.get_memory_manager") as mock_manager:
            mock = MagicMock()
            mock.get_stats = AsyncMock(return_value={
                "semantic": {"total_papers": 100},
                "episodic": {"total_memories": 50},
                "belief": {"total_beliefs": 10},
                "working": {"active_sessions": 2},
            })
            mock_manager.return_value = mock

            response = test_client.get("/api/stats")

            assert response.status_code == 200
            data = response.json()
            assert "total_papers" in data
            assert "memory_stats" in data
