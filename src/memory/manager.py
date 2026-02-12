"""
Memory Manager for PaperLens.

Orchestrates all memory types and provides a unified interface:
- Semantic memory (paper vectors)
- Episodic memory (interaction history)
- Working memory (session context)
- Belief memory (user preferences)

Handles memory consolidation and context building for the agent.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from src.config import settings
from src.memory.belief import BeliefMemoryStore, get_belief_store
from src.memory.episodic import EpisodicMemoryStore, get_episodic_store
from src.memory.semantic import SemanticMemory, get_semantic_memory
from src.memory.working import WorkingMemory, get_working_memory
from src.models.memory import BeliefType, EpisodicMemory
from src.models.paper import PaperSearchResult

logger = structlog.get_logger()


class MemoryManager:
    """
    Unified memory manager for the agent.

    Coordinates across all memory stores and provides:
    - Unified context retrieval
    - Memory consolidation (episodic → belief)
    - Personalized search enhancement
    """

    def __init__(
        self,
        semantic: SemanticMemory | None = None,
        episodic: EpisodicMemoryStore | None = None,
        working: WorkingMemory | None = None,
        belief: BeliefMemoryStore | None = None,
    ):
        """
        Initialize memory manager.

        Args:
            semantic: Semantic memory instance.
            episodic: Episodic memory store.
            working: Working memory instance.
            belief: Belief memory store.
        """
        self._semantic = semantic
        self._episodic = episodic
        self._working = working
        self._belief = belief

        logger.info("Memory manager initialized")

    @property
    def semantic(self) -> SemanticMemory:
        """Get semantic memory (lazy load)."""
        if self._semantic is None:
            self._semantic = get_semantic_memory()
        return self._semantic

    @property
    def episodic(self) -> EpisodicMemoryStore:
        """Get episodic memory (lazy load)."""
        if self._episodic is None:
            self._episodic = get_episodic_store()
        return self._episodic

    @property
    def working(self) -> WorkingMemory:
        """Get working memory (lazy load)."""
        if self._working is None:
            self._working = get_working_memory()
        return self._working

    @property
    def belief(self) -> BeliefMemoryStore:
        """Get belief memory (lazy load)."""
        if self._belief is None:
            self._belief = get_belief_store()
        return self._belief

    # =========================================================================
    # Context Building
    # =========================================================================

    async def build_context(
        self,
        session_id: str,
        query: str | None = None,
        include_beliefs: bool = True,
        include_episodic: bool = True,
        max_episodic: int = 5,
    ) -> dict[str, Any]:
        """
        Build rich context for the agent from all memory stores.

        Args:
            session_id: Current session ID.
            query: Current query (for finding relevant episodic memories).
            include_beliefs: Whether to include user preferences.
            include_episodic: Whether to include past interactions.
            max_episodic: Maximum episodic memories to include.

        Returns:
            Dict with context from all memory stores.
        """
        context: dict[str, Any] = {
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Working memory context
        working_state = self.working.get_session(session_id)
        context["working"] = {
            "message_count": len(working_state.messages),
            "retrieved_papers": working_state.retrieved_paper_ids[-10:],
            "current_query": working_state.current_query,
            "step_count": len(working_state.agent_steps),
        }

        # Recent conversation
        context["conversation"] = [
            {"role": m.role, "content": m.content[:200]}
            for m in working_state.messages[-5:]
        ]

        # Belief memory context
        if include_beliefs:
            context["beliefs"] = await self.belief.get_preferences_summary()

        # Episodic memory context
        if include_episodic:
            # Get recent relevant memories
            if query:
                episodic_memories = await self.episodic.search_by_query(
                    query, limit=max_episodic
                )
            else:
                episodic_memories = await self.episodic.get_recent(
                    limit=max_episodic, session_id=None
                )

            context["episodic"] = [
                {
                    "query": m.query,
                    "result_count": m.result_count,
                    "action_type": m.action_type,
                    "created_at": m.created_at.isoformat(),
                }
                for m in episodic_memories
            ]

            # Get liked papers for personalization
            liked = await self.episodic.get_liked_papers(limit=10)
            context["liked_papers"] = liked

        logger.debug("Built context", session_id=session_id, keys=list(context.keys()))
        return context

    def format_context_for_prompt(self, context: dict[str, Any]) -> str:
        """
        Format context dict as text for inclusion in prompts.

        Args:
            context: Context dict from build_context.

        Returns:
            Formatted text string.
        """
        lines = []

        # Working memory
        working = context.get("working", {})
        if working.get("retrieved_papers"):
            lines.append(f"Papers in session: {', '.join(working['retrieved_papers'][:5])}")
        if working.get("current_query"):
            lines.append(f"Current focus: {working['current_query']}")

        # Beliefs
        beliefs = context.get("beliefs", {})
        if beliefs.get("favorite_categories"):
            cats = [c["category"] for c in beliefs["favorite_categories"][:3]]
            lines.append(f"User interests: {', '.join(cats)}")
        if beliefs.get("interest_topics"):
            topics = [t["topic"] for t in beliefs["interest_topics"][:5]]
            lines.append(f"Topics of interest: {', '.join(topics)}")

        # Episodic
        episodic = context.get("episodic", [])
        if episodic:
            lines.append("Recent searches:")
            for mem in episodic[:3]:
                lines.append(f"  - \"{mem['query']}\" ({mem['result_count']} results)")

        # Liked papers
        liked = context.get("liked_papers", [])
        if liked:
            lines.append(f"Liked papers: {', '.join(liked[:5])}")

        return "\n".join(lines) if lines else "No prior context."

    # =========================================================================
    # Search with Personalization
    # =========================================================================

    async def search_papers(
        self,
        query: str,
        session_id: str,
        limit: int | None = None,
        use_preferences: bool = True,
        **kwargs: Any,
    ) -> list[PaperSearchResult]:
        """
        Search papers with personalization from beliefs.

        Args:
            query: Search query.
            session_id: Session ID.
            limit: Maximum results.
            use_preferences: Whether to boost results based on preferences.
            **kwargs: Additional search parameters.

        Returns:
            List of search results.
        """
        limit = limit or settings.search_top_k

        # Basic search
        results = self.semantic.search(query, limit=limit * 2, **kwargs)

        if use_preferences and results:
            # Get user preferences
            prefs = await self.belief.get_preferences_summary()

            # Extract preferred categories and topics
            fav_categories = {c["category"] for c in prefs.get("favorite_categories", [])}
            fav_topics = {t["topic"].lower() for t in prefs.get("interest_topics", [])}

            # Boost scores for matching preferences
            for result in results:
                boost = 0.0

                # Category match
                paper_cats = set(result.paper.categories)
                if paper_cats & fav_categories:
                    boost += 0.05

                # Topic match in title/abstract
                text = f"{result.paper.title} {result.paper.abstract}".lower()
                for topic in fav_topics:
                    if topic in text:
                        boost += 0.02

                # Apply boost (cap at 1.0)
                result.score = min(1.0, result.score + boost)

            # Re-sort by boosted score
            results.sort(key=lambda r: r.score, reverse=True)

        # Trim to limit
        results = results[:limit]

        # Track in working memory
        for r in results:
            self.working.add_paper(session_id, r.paper.arxiv_id)

        logger.debug("Personalized search", query=query[:30], results=len(results))
        return results

    # =========================================================================
    # Memory Recording
    # =========================================================================

    async def record_search(
        self,
        query: str,
        results: list[PaperSearchResult],
        session_id: str,
        query_embedding: list[float] | None = None,
    ) -> str:
        """
        Record a search in episodic memory.

        Args:
            query: Search query.
            results: Search results.
            session_id: Session ID.
            query_embedding: Query embedding for later recall.

        Returns:
            Memory ID.
        """
        memory = EpisodicMemory(
            query=query,
            query_embedding=query_embedding,
            session_id=session_id,
            action_type="search",
            result_paper_ids=[r.paper.arxiv_id for r in results],
            result_count=len(results),
        )

        memory_id = await self.episodic.store(memory)

        # Learn from search (update beliefs)
        await self._learn_from_search(results, memory_id)

        logger.debug("Recorded search", memory_id=memory_id, results=len(results))
        return memory_id

    async def record_feedback(
        self,
        arxiv_id: str,
        liked: bool,
        session_id: str,
        memory_id: str | None = None,
    ) -> None:
        """
        Record user feedback on a paper.

        Args:
            arxiv_id: Paper ArXiv ID.
            liked: Whether user liked the paper.
            session_id: Session ID.
            memory_id: Associated search memory ID.
        """
        # Record interaction
        await self.episodic.record_interaction(
            arxiv_id=arxiv_id,
            interaction_type="like" if liked else "dislike",
            session_id=session_id,
            memory_id=memory_id,
        )

        # Update memory if provided
        if memory_id:
            await self.episodic.add_feedback(memory_id, arxiv_id, liked)

        # Learn from feedback
        if liked:
            await self._learn_from_liked_paper(arxiv_id, memory_id)

        logger.debug("Recorded feedback", arxiv_id=arxiv_id, liked=liked)

    async def record_view(
        self,
        arxiv_id: str,
        session_id: str,
    ) -> None:
        """
        Record that a paper was viewed.

        Args:
            arxiv_id: Paper ArXiv ID.
            session_id: Session ID.
        """
        await self.episodic.record_interaction(
            arxiv_id=arxiv_id,
            interaction_type="view",
            session_id=session_id,
        )

        # Track in working memory
        self.working.add_paper(session_id, arxiv_id)

    # =========================================================================
    # Learning & Consolidation
    # =========================================================================

    async def _learn_from_search(
        self,
        results: list[PaperSearchResult],
        memory_id: str,
    ) -> None:
        """Learn interests from search results."""
        if not results:
            return

        # Extract categories from top results
        categories: dict[str, int] = {}
        for r in results[:5]:  # Top 5 results
            for cat in r.paper.categories:
                categories[cat] = categories.get(cat, 0) + 1

        # Reinforce frequent categories
        for cat, count in categories.items():
            if count >= 2:  # Appeared in at least 2 top results
                await self.belief.reinforce(
                    BeliefType.FAVORITE_CATEGORY,
                    cat,
                    strength=0.05 * count,
                    source_memory_id=memory_id,
                )

    async def _learn_from_liked_paper(
        self,
        arxiv_id: str,
        memory_id: str | None,
    ) -> None:
        """Learn preferences from liked paper."""
        # Get paper details
        paper = self.semantic.get_paper(arxiv_id)
        if not paper:
            return

        # Reinforce categories
        for cat in paper.categories:
            await self.belief.reinforce(
                BeliefType.FAVORITE_CATEGORY,
                cat,
                strength=0.1,
                source_memory_id=memory_id,
            )

        # Reinforce first author
        if paper.authors:
            await self.belief.reinforce(
                BeliefType.FAVORITE_AUTHOR,
                paper.authors[0],
                strength=0.1,
                source_memory_id=memory_id,
            )

    async def consolidate(self) -> dict[str, int]:
        """
        Perform memory consolidation.

        - Decay old beliefs
        - Prune low-confidence beliefs
        - Extract patterns from episodic memory

        Returns:
            Dict with consolidation stats.
        """
        stats = {}

        # Decay beliefs
        decayed = await self.belief.decay_all()
        stats["beliefs_decayed"] = decayed

        # Prune low-confidence beliefs
        pruned = await self.belief.prune(min_confidence=0.1)
        stats["beliefs_pruned"] = pruned

        # Extract patterns from recent episodic memories
        recent = await self.episodic.get_recent(limit=100, hours=168)  # Last week

        # Count frequently searched topics
        topics: dict[str, int] = {}
        for mem in recent:
            # Simple keyword extraction from queries
            words = mem.query.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    topics[word] = topics.get(word, 0) + 1

        # Reinforce frequent topics
        reinforced = 0
        for topic, count in topics.items():
            if count >= 3:  # Appeared in at least 3 searches
                await self.belief.reinforce(
                    BeliefType.INTEREST_TOPIC,
                    topic,
                    strength=0.05 * min(count, 10),
                )
                reinforced += 1

        stats["topics_reinforced"] = reinforced

        logger.info("Memory consolidation complete", **stats)
        return stats

    # =========================================================================
    # Recall
    # =========================================================================

    async def recall(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Recall relevant memories for a query.

        Args:
            query: Recall query.
            session_id: Optional session filter.
            limit: Maximum memories to return.

        Returns:
            List of relevant memory summaries.
        """
        # Search episodic memories
        memories = await self.episodic.search_by_query(query, limit=limit)

        results = []
        for mem in memories:
            results.append({
                "type": "episodic",
                "query": mem.query,
                "result_count": mem.result_count,
                "papers": mem.result_paper_ids[:5],
                "liked": mem.liked_paper_ids,
                "when": mem.created_at.isoformat(),
            })

        return results

    async def recall_similar_queries(
        self,
        query: str,
        limit: int = 3,
    ) -> list[EpisodicMemory]:
        """
        Find similar past queries.

        Args:
            query: Current query.
            limit: Maximum results.

        Returns:
            List of similar episodic memories.
        """
        return await self.episodic.search_by_query(query, limit=limit)

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start a new session.

        Args:
            session_id: Optional session ID. Generated if None.

        Returns:
            Session ID.
        """
        state = self.working.get_session(session_id)
        sid: str = state.session_id
        logger.info("Started session", session_id=sid)
        return sid

    async def end_session(self, session_id: str) -> None:
        """
        End a session and persist important data.

        Args:
            session_id: Session ID.
        """
        # The working memory conversation is already saved to episodic
        # Just clear the working memory
        self.working.clear_session(session_id)
        logger.info("Ended session", session_id=session_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics from all memory stores.

        Returns:
            Dict with statistics from each store.
        """
        stats = {}

        # Semantic
        stats["semantic"] = self.semantic.get_stats()

        # Episodic
        stats["episodic"] = await self.episodic.get_stats()

        # Belief
        stats["belief"] = await self.belief.get_stats()

        # Working
        stats["working"] = {
            "active_sessions": len(self.working.list_sessions()),
        }

        return stats


# Singleton instance
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create the memory manager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


if __name__ == "__main__":
    # Quick test
    async def test() -> None:
        manager = MemoryManager()

        # Start a session
        session_id = manager.start_session()
        print(f"Session: {session_id}")

        # Build context
        context = await manager.build_context(session_id, "transformer attention")
        print(f"Context keys: {list(context.keys())}")

        # Get stats
        stats = await manager.get_stats()
        print(f"Stats: {stats}")

        # End session
        await manager.end_session(session_id)

    asyncio.run(test())
