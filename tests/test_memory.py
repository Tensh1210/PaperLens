"""
Tests for PaperLens memory system.
"""

import pytest
from datetime import datetime


class TestWorkingMemory:
    """Tests for working memory."""

    def test_session_creation(self, working_memory):
        """Test session creation."""
        session = working_memory.get_session()
        assert session.session_id is not None
        assert working_memory.session_exists(session.session_id)

    def test_session_with_id(self, working_memory):
        """Test session creation with specific ID."""
        session = working_memory.get_session("test-session-123")
        assert session.session_id == "test-session-123"

    def test_add_message(self, working_memory):
        """Test adding messages to session."""
        session = working_memory.get_session("test-session")
        working_memory.add_message("test-session", "user", "Hello!")
        working_memory.add_message("test-session", "assistant", "Hi there!")

        messages = working_memory.get_conversation("test-session")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello!"

    def test_add_step(self, working_memory):
        """Test adding agent steps."""
        working_memory.get_session("test-session")
        step = working_memory.add_step(
            "test-session",
            thought="I should search for papers",
            action="search_papers",
            action_input={"query": "transformers"},
        )

        assert step.step_number == 1
        assert step.thought == "I should search for papers"
        assert step.action == "search_papers"

    def test_update_step_observation(self, working_memory):
        """Test updating step observation."""
        working_memory.get_session("test-session")
        step = working_memory.add_step(
            "test-session",
            thought="Searching...",
            action="search_papers",
        )

        working_memory.update_step_observation(
            "test-session",
            step.step_number,
            "Found 5 papers",
        )

        steps = working_memory.get_steps("test-session")
        assert steps[0].observation == "Found 5 papers"

    def test_add_paper(self, working_memory):
        """Test tracking papers in session."""
        working_memory.get_session("test-session")
        working_memory.add_paper("test-session", "1706.03762")
        working_memory.add_paper("test-session", "1810.04805")

        papers = working_memory.get_retrieved_papers("test-session")
        assert "1706.03762" in papers
        assert "1810.04805" in papers

    def test_get_messages_for_llm(self, working_memory):
        """Test formatting messages for LLM."""
        working_memory.get_session("test-session")
        working_memory.add_message("test-session", "user", "Find papers")
        working_memory.add_message("test-session", "assistant", "Searching...")

        messages = working_memory.get_messages_for_llm(
            "test-session",
            system_prompt="You are a helpful assistant.",
        )

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_context_summary(self, working_memory):
        """Test getting context summary."""
        working_memory.get_session("test-session")
        working_memory.add_message("test-session", "user", "Test")
        working_memory.add_paper("test-session", "1706.03762")
        working_memory.set_query("test-session", "transformer papers")

        context = working_memory.get_context_summary("test-session")

        assert context["message_count"] == 1
        assert context["retrieved_paper_count"] == 1
        assert context["current_query"] == "transformer papers"

    def test_clear_session(self, working_memory):
        """Test clearing session."""
        working_memory.get_session("test-session")
        working_memory.add_message("test-session", "user", "Test")
        working_memory.add_paper("test-session", "1706.03762")

        working_memory.clear_session("test-session")
        session = working_memory.get_session("test-session")

        assert len(session.messages) == 0
        assert len(session.retrieved_paper_ids) == 0

    def test_max_size_enforcement(self, working_memory):
        """Test max size is enforced."""
        working_memory.get_session("test-session")

        # Add more messages than max size
        for i in range(15):
            working_memory.add_message("test-session", "user", f"Message {i}")

        messages = working_memory.get_conversation("test-session")
        assert len(messages) == 10  # max_size was set to 10


class TestEpisodicMemoryStore:
    """Tests for episodic memory store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, episodic_memory_store):
        """Test storing and retrieving memory."""
        from src.models.memory import EpisodicMemory

        memory = EpisodicMemory(
            query="transformer attention",
            session_id="test-session",
            result_paper_ids=["1706.03762"],
            result_count=1,
        )

        memory_id = await episodic_memory_store.store(memory)
        retrieved = await episodic_memory_store.get(memory_id)

        assert retrieved is not None
        assert retrieved.query == "transformer attention"
        assert retrieved.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_get_recent(self, episodic_memory_store):
        """Test getting recent memories."""
        from src.models.memory import EpisodicMemory

        for i in range(5):
            memory = EpisodicMemory(
                query=f"test query {i}",
                session_id="test-session",
            )
            await episodic_memory_store.store(memory)

        recent = await episodic_memory_store.get_recent(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_search_by_query(self, episodic_memory_store):
        """Test searching memories by query text."""
        from src.models.memory import EpisodicMemory

        await episodic_memory_store.store(
            EpisodicMemory(query="transformer attention mechanism")
        )
        await episodic_memory_store.store(
            EpisodicMemory(query="BERT language model")
        )

        results = await episodic_memory_store.search_by_query("transformer")
        assert len(results) >= 1
        assert any("transformer" in r.query.lower() for r in results)

    @pytest.mark.asyncio
    async def test_record_interaction(self, episodic_memory_store):
        """Test recording paper interactions."""
        await episodic_memory_store.record_interaction(
            arxiv_id="1706.03762",
            interaction_type="view",
            session_id="test-session",
        )

        interactions = await episodic_memory_store.get_paper_interactions(
            arxiv_id="1706.03762"
        )
        assert len(interactions) >= 1
        assert interactions[0]["arxiv_id"] == "1706.03762"

    @pytest.mark.asyncio
    async def test_add_feedback(self, episodic_memory_store):
        """Test adding feedback to memory."""
        from src.models.memory import EpisodicMemory

        memory = EpisodicMemory(
            query="test query",
            result_paper_ids=["1706.03762"],
        )
        memory_id = await episodic_memory_store.store(memory)

        await episodic_memory_store.add_feedback(
            memory_id=memory_id,
            paper_id="1706.03762",
            liked=True,
        )

        retrieved = await episodic_memory_store.get(memory_id)
        assert "1706.03762" in retrieved.liked_paper_ids

    @pytest.mark.asyncio
    async def test_get_stats(self, episodic_memory_store):
        """Test getting store statistics."""
        from src.models.memory import EpisodicMemory

        await episodic_memory_store.store(EpisodicMemory(query="test 1"))
        await episodic_memory_store.store(EpisodicMemory(query="test 2"))

        stats = await episodic_memory_store.get_stats()
        assert stats["total_memories"] >= 2


class TestBeliefMemoryStore:
    """Tests for belief memory store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, belief_memory_store):
        """Test storing and retrieving beliefs."""
        from src.models.memory import BeliefMemory, BeliefType

        belief = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.7,
        )

        belief_id = await belief_memory_store.store(belief)
        retrieved = await belief_memory_store.get(belief_id)

        assert retrieved is not None
        assert retrieved.belief_type == BeliefType.FAVORITE_CATEGORY
        assert retrieved.value == "cs.CL"

    @pytest.mark.asyncio
    async def test_reinforcement(self, belief_memory_store):
        """Test that storing same belief reinforces it."""
        from src.models.memory import BeliefMemory, BeliefType

        # Store first time
        belief1 = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.5,
        )
        belief_id1 = await belief_memory_store.store(belief1)

        # Store again (should reinforce)
        belief2 = BeliefMemory(
            belief_type=BeliefType.FAVORITE_CATEGORY,
            value="cs.CL",
            confidence=0.5,
        )
        belief_id2 = await belief_memory_store.store(belief2)

        # Should be same belief, reinforced
        assert belief_id1 == belief_id2

        retrieved = await belief_memory_store.get(belief_id1)
        assert retrieved.confidence > 0.5
        assert retrieved.reinforcement_count > 1

    @pytest.mark.asyncio
    async def test_get_by_type(self, belief_memory_store):
        """Test getting beliefs by type."""
        from src.models.memory import BeliefType

        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.CL")
        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.LG")
        await belief_memory_store.reinforce(BeliefType.FAVORITE_AUTHOR, "John Smith")

        categories = await belief_memory_store.get_by_type(
            BeliefType.FAVORITE_CATEGORY
        )
        assert len(categories) == 2

        authors = await belief_memory_store.get_by_type(
            BeliefType.FAVORITE_AUTHOR
        )
        assert len(authors) == 1

    @pytest.mark.asyncio
    async def test_get_favorite_categories(self, belief_memory_store):
        """Test getting favorite categories."""
        from src.models.memory import BeliefType

        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.CL")
        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.LG")

        categories = await belief_memory_store.get_favorite_categories()
        assert len(categories) == 2
        assert any(c[0] == "cs.CL" for c in categories)

    @pytest.mark.asyncio
    async def test_decay_all(self, belief_memory_store):
        """Test decaying all beliefs."""
        from src.models.memory import BeliefType

        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.CL")

        # Get initial confidence
        beliefs = await belief_memory_store.get_by_type(BeliefType.FAVORITE_CATEGORY)
        initial_confidence = beliefs[0].confidence

        # Decay
        await belief_memory_store.decay_all()

        # Check decayed
        beliefs = await belief_memory_store.get_by_type(BeliefType.FAVORITE_CATEGORY)
        assert beliefs[0].confidence < initial_confidence

    @pytest.mark.asyncio
    async def test_set_preference(self, belief_memory_store):
        """Test setting user preference."""
        from src.models.memory import BeliefType

        belief_id = await belief_memory_store.set_preference(
            BeliefType.READING_LEVEL,
            "expert",
            user_confirmed=True,
        )

        belief = await belief_memory_store.get(belief_id)
        assert belief.value == "expert"
        assert belief.user_confirmed is True
        assert belief.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_get_preferences_summary(self, belief_memory_store):
        """Test getting preferences summary."""
        from src.models.memory import BeliefType

        await belief_memory_store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.CL")
        await belief_memory_store.reinforce(BeliefType.INTEREST_TOPIC, "transformers")

        summary = await belief_memory_store.get_preferences_summary()

        assert "favorite_categories" in summary
        assert "interest_topics" in summary
