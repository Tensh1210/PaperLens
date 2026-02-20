"""
Episodic Memory for PaperLens.

Stores interaction history using SQLite for persistence:
- Past search queries and results
- User feedback (liked/disliked papers)
- Session history

Enables queries like "Show me papers like the one I searched for last week".
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.config import settings
from src.models.memory import EpisodicMemory

logger = structlog.get_logger()

# SQL schema for episodic memory
SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    query TEXT NOT NULL,
    query_embedding TEXT,
    action_type TEXT DEFAULT 'search',
    result_paper_ids TEXT,
    result_count INTEGER DEFAULT 0,
    feedback TEXT,
    liked_paper_ids TEXT,
    disliked_paper_ids TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_created ON episodic_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_episodic_query ON episodic_memories(query);

CREATE TABLE IF NOT EXISTS paper_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    arxiv_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    memory_id TEXT,
    metadata TEXT,
    FOREIGN KEY (memory_id) REFERENCES episodic_memories(id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_paper ON paper_interactions(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON paper_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON paper_interactions(interaction_type);
"""


class EpisodicMemoryStore:
    """
    SQLite-backed episodic memory store.

    Stores and retrieves interaction history for contextual recall.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize episodic memory store.

        Args:
            db_path: Path to SQLite database. Defaults to config value.
        """
        self.db_path = db_path or settings.memory_db_path
        self._initialized = False

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Episodic memory store initialized", db_path=self.db_path)

    async def _ensure_initialized(self, conn: aiosqlite.Connection) -> None:
        """Ensure schema exists."""
        if not self._initialized:
            await conn.executescript(SCHEMA)
            await conn.commit()
            self._initialized = True
            logger.debug("Database schema initialized")

    async def store(self, memory: EpisodicMemory) -> str:
        """
        Store an episodic memory.

        Args:
            memory: EpisodicMemory to store.

        Returns:
            Memory ID.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            await conn.execute(
                """
                INSERT INTO episodic_memories (
                    id, created_at, updated_at, session_id, query, query_embedding,
                    action_type, result_paper_ids, result_count, feedback,
                    liked_paper_ids, disliked_paper_ids, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.session_id,
                    memory.query,
                    json.dumps(memory.query_embedding) if memory.query_embedding else None,
                    memory.action_type,
                    json.dumps(memory.result_paper_ids),
                    memory.result_count,
                    memory.feedback,
                    json.dumps(memory.liked_paper_ids),
                    json.dumps(memory.disliked_paper_ids),
                    json.dumps(memory.metadata),
                ),
            )
            await conn.commit()

        memory_id: str = memory.id
        logger.debug("Stored episodic memory", id=memory_id, query=memory.query[:30])
        return memory_id

    async def get(self, memory_id: str) -> EpisodicMemory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            EpisodicMemory or None if not found.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                "SELECT * FROM episodic_memories WHERE id = ?",
                (memory_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    async def get_recent(
        self,
        limit: int | None = None,
        session_id: str | None = None,
        action_type: str | None = None,
        hours: int | None = None,
    ) -> list[EpisodicMemory]:
        """
        Get recent memories.

        Args:
            limit: Maximum memories to return.
            session_id: Filter by session.
            action_type: Filter by action type.
            hours: Only memories from last N hours.

        Returns:
            List of EpisodicMemory ordered by recency.
        """
        limit = limit or settings.memory_episodic_limit

        conditions: list[str] = []
        params: list[str | int] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if action_type:
            conditions.append("action_type = ?")
            params.append(action_type)

        if hours:
            cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
            conditions.append("created_at > ?")
            params.append(cutoff)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                f"""
                SELECT * FROM episodic_memories
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            )
            rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def search_by_query(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[EpisodicMemory]:
        """
        Search memories by query text (simple text matching).

        Args:
            query_text: Text to search for.
            limit: Maximum results.

        Returns:
            List of matching memories.
        """
        # TODO: Consider using SQLite FTS5 for better full-text search.
        # Current LIKE-based search is adequate for small datasets but
        # will degrade with scale. FTS5 would also support ranking.
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT * FROM episodic_memories
                WHERE query LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query_text}%", limit),
            )
            rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def get_by_paper(
        self,
        arxiv_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemory]:
        """
        Get memories that reference a specific paper.

        Args:
            arxiv_id: Paper ArXiv ID.
            limit: Maximum results.

        Returns:
            List of memories referencing the paper.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT * FROM episodic_memories
                WHERE result_paper_ids LIKE ? OR liked_paper_ids LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f'%"{arxiv_id}"%', f'%"{arxiv_id}"%', limit),
            )
            rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def add_feedback(
        self,
        memory_id: str,
        paper_id: str,
        liked: bool,
    ) -> None:
        """
        Add feedback to a memory.

        Args:
            memory_id: Memory ID.
            paper_id: Paper ArXiv ID.
            liked: Whether the paper was liked.
        """
        memory = await self.get(memory_id)
        if not memory:
            logger.warning("Memory not found for feedback", id=memory_id)
            return

        memory.add_feedback(paper_id, liked)

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            await conn.execute(
                """
                UPDATE episodic_memories
                SET liked_paper_ids = ?, disliked_paper_ids = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(memory.liked_paper_ids),
                    json.dumps(memory.disliked_paper_ids),
                    datetime.now(UTC).isoformat(),
                    memory_id,
                ),
            )
            await conn.commit()

        logger.debug("Added feedback", memory_id=memory_id, paper_id=paper_id, liked=liked)

    async def record_interaction(
        self,
        arxiv_id: str,
        interaction_type: str,
        session_id: str | None = None,
        memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a paper interaction.

        Args:
            arxiv_id: Paper ArXiv ID.
            interaction_type: Type of interaction (view, like, dislike, compare, summarize).
            session_id: Session ID.
            memory_id: Associated memory ID.
            metadata: Additional metadata.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            await conn.execute(
                """
                INSERT INTO paper_interactions (
                    session_id, arxiv_id, interaction_type, memory_id, metadata
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    arxiv_id,
                    interaction_type,
                    memory_id,
                    json.dumps(metadata or {}),
                ),
            )
            await conn.commit()

        logger.debug(
            "Recorded interaction",
            arxiv_id=arxiv_id,
            type=interaction_type,
        )

    async def get_paper_interactions(
        self,
        arxiv_id: str | None = None,
        interaction_type: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get paper interactions.

        Args:
            arxiv_id: Filter by paper.
            interaction_type: Filter by type.
            session_id: Filter by session.
            limit: Maximum results.

        Returns:
            List of interaction records.
        """
        conditions: list[str] = []
        params: list[str | int] = []

        if arxiv_id:
            conditions.append("arxiv_id = ?")
            params.append(arxiv_id)

        if interaction_type:
            conditions.append("interaction_type = ?")
            params.append(interaction_type)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                f"""
                SELECT * FROM paper_interactions
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def get_frequently_viewed_papers(
        self,
        limit: int = 10,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Get most frequently viewed papers.

        Args:
            limit: Maximum papers to return.
            days: Only consider interactions from last N days.

        Returns:
            List of papers with view counts.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT arxiv_id, COUNT(*) as view_count
                FROM paper_interactions
                WHERE interaction_type = 'view' AND created_at > ?
                GROUP BY arxiv_id
                ORDER BY view_count DESC
                LIMIT ?
                """,
                (cutoff, limit),
            )
            rows = await cursor.fetchall()

        return [{"arxiv_id": row["arxiv_id"], "view_count": row["view_count"]} for row in rows]

    async def get_liked_papers(
        self,
        limit: int = 50,
    ) -> list[str]:
        """
        Get all liked paper IDs.

        Args:
            limit: Maximum papers to return.

        Returns:
            List of ArXiv IDs.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT DISTINCT arxiv_id
                FROM paper_interactions
                WHERE interaction_type = 'like'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

        return [row["arxiv_id"] for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """
        Get episodic memory statistics.

        Returns:
            Dict with statistics.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            # Total memories
            cursor = await conn.execute("SELECT COUNT(*) as count FROM episodic_memories")
            row = await cursor.fetchone()
            total_memories = row["count"] if row else 0

            # Total interactions
            cursor = await conn.execute("SELECT COUNT(*) as count FROM paper_interactions")
            row = await cursor.fetchone()
            total_interactions = row["count"] if row else 0

            # Unique papers interacted with
            cursor = await conn.execute(
                "SELECT COUNT(DISTINCT arxiv_id) as count FROM paper_interactions"
            )
            row = await cursor.fetchone()
            unique_papers = row["count"] if row else 0

            # Memories by action type
            cursor = await conn.execute(
                """
                SELECT action_type, COUNT(*) as count
                FROM episodic_memories
                GROUP BY action_type
                """
            )
            rows = await cursor.fetchall()
            by_action = {row["action_type"]: row["count"] for row in rows}

        return {
            "total_memories": total_memories,
            "total_interactions": total_interactions,
            "unique_papers": unique_papers,
            "memories_by_action": by_action,
        }

    async def clear(self, session_id: str | None = None) -> int:
        """
        Clear memories.

        Args:
            session_id: If provided, only clear this session's memories.

        Returns:
            Number of memories deleted.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            if session_id:
                cursor = await conn.execute(
                    "DELETE FROM episodic_memories WHERE session_id = ?",
                    (session_id,),
                )
                await conn.execute(
                    "DELETE FROM paper_interactions WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cursor = await conn.execute("DELETE FROM episodic_memories")
                await conn.execute("DELETE FROM paper_interactions")

            await conn.commit()
            deleted = cursor.rowcount

        logger.info("Cleared episodic memories", count=deleted, session_id=session_id)
        return deleted

    def _row_to_memory(self, row: aiosqlite.Row) -> EpisodicMemory:
        """Convert database row to EpisodicMemory."""
        return EpisodicMemory(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            session_id=row["session_id"],
            query=row["query"],
            query_embedding=json.loads(row["query_embedding"]) if row["query_embedding"] else None,
            action_type=row["action_type"],
            result_paper_ids=json.loads(row["result_paper_ids"]) if row["result_paper_ids"] else [],
            result_count=row["result_count"],
            feedback=row["feedback"],
            liked_paper_ids=json.loads(row["liked_paper_ids"]) if row["liked_paper_ids"] else [],
            disliked_paper_ids=json.loads(row["disliked_paper_ids"]) if row["disliked_paper_ids"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # Sync wrappers for convenience
    def store_sync(self, memory: EpisodicMemory) -> str:
        """Synchronous wrapper for store."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.store(memory))
        finally:
            loop.close()

    def get_recent_sync(self, **kwargs: Any) -> list[EpisodicMemory]:
        """Synchronous wrapper for get_recent."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.get_recent(**kwargs))
        finally:
            loop.close()


# Singleton instance
_episodic_store: EpisodicMemoryStore | None = None


def get_episodic_store() -> EpisodicMemoryStore:
    """Get or create the episodic memory store singleton."""
    global _episodic_store
    if _episodic_store is None:
        _episodic_store = EpisodicMemoryStore()
    return _episodic_store


if __name__ == "__main__":
    # Quick test
    async def test() -> None:
        store = EpisodicMemoryStore("data/test_memory.db")

        # Create a memory
        memory = EpisodicMemory(
            query="transformer attention mechanism",
            session_id="test-session",
            result_paper_ids=["1706.03762", "1810.04805"],
            result_count=2,
            action_type="search",
        )

        # Store it
        await store.store(memory)
        print(f"Stored memory: {memory.id}")

        # Retrieve it
        retrieved = await store.get(memory.id)
        if retrieved:
            print(f"Retrieved: {retrieved.query}")

        # Get recent
        recent = await store.get_recent(limit=5)
        print(f"Recent memories: {len(recent)}")

        # Record interaction
        await store.record_interaction("1706.03762", "view", "test-session")

        # Get stats
        stats = await store.get_stats()
        print(f"Stats: {stats}")

    asyncio.run(test())
