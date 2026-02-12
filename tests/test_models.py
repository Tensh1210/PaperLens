"""
Tests for PaperLens data models.
"""


import pytest


class TestPaper:
    """Tests for the Paper model."""

    def test_paper_creation(self, sample_paper):
        """Test basic paper creation."""
        assert sample_paper.arxiv_id == "2301.12345"
        assert sample_paper.title == "Test Paper: A Study of Testing"
        assert len(sample_paper.authors) == 2
        assert sample_paper.citation_count == 42

    def test_paper_computed_urls(self, sample_paper):
        """Test computed URL properties."""
        assert sample_paper.pdf_url == "https://arxiv.org/pdf/2301.12345.pdf"
        assert sample_paper.arxiv_url == "https://arxiv.org/abs/2301.12345"

    def test_paper_year_extraction(self, sample_paper):
        """Test year extraction from arxiv_id."""
        assert sample_paper.year == 2023

    def test_paper_to_embedding_text(self, sample_paper):
        """Test embedding text generation."""
        text = sample_paper.to_embedding_text()
        assert sample_paper.title in text
        assert sample_paper.abstract in text

    def test_paper_to_display_dict(self, sample_paper):
        """Test display dict conversion."""
        display = sample_paper.to_display_dict()
        assert display["arxiv_id"] == sample_paper.arxiv_id
        assert display["title"] == sample_paper.title
        assert "pdf_url" in display


class TestPaperSearchResult:
    """Tests for the PaperSearchResult model."""

    def test_search_result_creation(self, sample_paper):
        """Test search result creation."""
        from src.models.paper import PaperSearchResult

        result = PaperSearchResult(paper=sample_paper, score=0.95)
        assert result.paper.arxiv_id == sample_paper.arxiv_id
        assert result.score == 0.95

    def test_search_result_ordering(self, sample_papers):
        """Test search results can be sorted by score."""
        from src.models.paper import PaperSearchResult

        results = [
            PaperSearchResult(paper=sample_papers[0], score=0.7),
            PaperSearchResult(paper=sample_papers[1], score=0.9),
            PaperSearchResult(paper=sample_papers[2], score=0.8),
        ]

        sorted_results = sorted(results)
        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.8
        assert sorted_results[2].score == 0.7


class TestMemoryModels:
    """Tests for memory data models."""

    def test_episodic_memory_creation(self):
        """Test episodic memory creation."""
        from src.models.memory import EpisodicMemory

        memory = EpisodicMemory(
            query="transformer attention",
            session_id="test-session",
            result_paper_ids=["1706.03762", "1810.04805"],
            result_count=2,
        )

        assert memory.query == "transformer attention"
        assert memory.session_id == "test-session"
        assert len(memory.result_paper_ids) == 2
        assert memory.id is not None

    def test_episodic_memory_feedback(self):
        """Test adding feedback to episodic memory."""
        from src.models.memory import EpisodicMemory

        memory = EpisodicMemory(query="test query")
        memory.add_feedback("1706.03762", liked=True)

        assert "1706.03762" in memory.liked_paper_ids
        assert "1706.03762" not in memory.disliked_paper_ids

        memory.add_feedback("1706.03762", liked=False)
        assert "1706.03762" not in memory.liked_paper_ids
        assert "1706.03762" in memory.disliked_paper_ids

    def test_belief_memory_creation(self):
        """Test belief memory creation."""
        from src.models.memory import BeliefMemory, BeliefType

        belief = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.7,
        )

        assert belief.belief_type == BeliefType.FAVORITE_CATEGORY
        assert belief.value == "cs.CL"
        assert belief.confidence == 0.7

    def test_belief_memory_reinforce(self):
        """Test belief reinforcement."""
        from src.models.memory import BeliefMemory, BeliefType

        belief = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.5,
        )

        original_confidence = belief.confidence
        belief.reinforce(strength=0.1)

        assert belief.confidence > original_confidence
        assert belief.reinforcement_count == 2

    def test_belief_memory_decay(self):
        """Test belief decay."""
        from src.models.memory import BeliefMemory, BeliefType

        belief = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.8,
        )

        original_confidence = belief.confidence
        belief.decay(factor=0.9)

        assert belief.confidence < original_confidence
        assert belief.confidence == pytest.approx(0.72, rel=0.01)

    def test_belief_user_confirmed_no_decay(self):
        """Test that user-confirmed beliefs don't decay."""
        from src.models.memory import BeliefMemory, BeliefType

        belief = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.8,
            user_confirmed=True,
        )

        original_confidence = belief.confidence
        belief.decay(factor=0.9)

        assert belief.confidence == original_confidence

    def test_working_memory_state(self):
        """Test working memory state operations."""
        from src.models.memory import WorkingMemoryState

        state = WorkingMemoryState()

        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there!")

        assert len(state.messages) == 2
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"

    def test_working_memory_agent_steps(self):
        """Test adding agent steps to working memory."""
        from src.models.memory import WorkingMemoryState

        state = WorkingMemoryState()

        step = state.add_step(
            thought="I need to search for papers",
            action="search_papers",
            action_input={"query": "transformers"},
        )

        assert step.step_number == 1
        assert step.thought == "I need to search for papers"
        assert step.action == "search_papers"

    def test_working_memory_clear(self):
        """Test clearing working memory."""
        from src.models.memory import WorkingMemoryState

        state = WorkingMemoryState()
        state.add_message("user", "Hello")
        state.add_paper("1706.03762")
        state.add_step(thought="Thinking...")

        state.clear()

        assert len(state.messages) == 0
        assert len(state.retrieved_paper_ids) == 0
        assert len(state.agent_steps) == 0
