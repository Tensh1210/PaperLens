"""
Belief Memory for PaperLens.

Stores user preferences using SQLite for persistence:
- Favorite categories and topics
- Favorite authors
- Reading level preferences
- Learned patterns from interactions

Beliefs have confidence scores that can be reinforced or decayed.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.config import settings
from src.models.memory import BeliefMemory, BeliefType

logger = structlog.get_logger()

# SQL schema for belief memory
SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    belief_type TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    reinforcement_count INTEGER DEFAULT 1,
    source_memory_ids TEXT,
    user_confirmed INTEGER DEFAULT 0,
    metadata TEXT,
    UNIQUE(belief_type, value)
);

CREATE INDEX IF NOT EXISTS idx_beliefs_type ON beliefs(belief_type);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
CREATE INDEX IF NOT EXISTS idx_beliefs_updated ON beliefs(updated_at);
"""


class BeliefMemoryStore:
    """
    SQLite-backed belief memory store.

    Stores and manages user preferences with confidence scoring.
    """

    def __init__(
        self,
        db_path: str | None = None,
        decay_factor: float | None = None,
    ):
        """
        Initialize belief memory store.

        Args:
            db_path: Path to SQLite database. Defaults to config value.
            decay_factor: Confidence decay factor. Defaults to config value.
        """
        self.db_path = db_path or settings.memory_db_path
        self.decay_factor = decay_factor or settings.memory_belief_decay
        self._initialized = False

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Belief memory store initialized",
            db_path=self.db_path,
            decay_factor=self.decay_factor,
        )

    async def _ensure_initialized(self, conn: aiosqlite.Connection) -> None:
        """Ensure schema exists."""
        if not self._initialized:
            await conn.executescript(SCHEMA)
            await conn.commit()
            self._initialized = True
            logger.debug("Belief database schema initialized")

    async def store(self, belief: BeliefMemory) -> str:
        """
        Store or update a belief.

        If a belief with the same type and value exists, it will be reinforced.

        Args:
            belief: BeliefMemory to store.

        Returns:
            Belief ID.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            # Check if belief already exists
            cursor = await conn.execute(
                "SELECT id, confidence, reinforcement_count FROM beliefs WHERE belief_type = ? AND value = ?",
                (belief.belief_type.value, belief.value),
            )
            existing = await cursor.fetchone()

            if existing:
                # Reinforce existing belief
                new_confidence = min(
                    1.0,
                    existing["confidence"] + 0.1 * (1 - existing["confidence"])
                )
                new_count = existing["reinforcement_count"] + 1

                # Merge source memory IDs
                cursor = await conn.execute(
                    "SELECT source_memory_ids FROM beliefs WHERE id = ?",
                    (existing["id"],),
                )
                row = await cursor.fetchone()
                existing_sources = json.loads(row["source_memory_ids"]) if row and row["source_memory_ids"] else []
                merged_sources = list(set(existing_sources + belief.source_memory_ids))

                await conn.execute(
                    """
                    UPDATE beliefs
                    SET confidence = ?, reinforcement_count = ?, source_memory_ids = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        new_confidence,
                        new_count,
                        json.dumps(merged_sources),
                        datetime.now(UTC).isoformat(),
                        existing["id"],
                    ),
                )
                await conn.commit()

                existing_id: str = existing["id"]
                logger.debug(
                    "Reinforced belief",
                    id=existing_id,
                    type=belief.belief_type.value,
                    confidence=new_confidence,
                )
                return existing_id
            else:
                # Insert new belief
                await conn.execute(
                    """
                    INSERT INTO beliefs (
                        id, created_at, updated_at, belief_type, value, confidence,
                        reinforcement_count, source_memory_ids, user_confirmed, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        belief.id,
                        belief.created_at.isoformat(),
                        belief.updated_at.isoformat(),
                        belief.belief_type.value,
                        belief.value,
                        belief.confidence,
                        belief.reinforcement_count,
                        json.dumps(belief.source_memory_ids),
                        1 if belief.user_confirmed else 0,
                        json.dumps({}),
                    ),
                )
                await conn.commit()

                belief_id: str = belief.id
                logger.debug(
                    "Stored new belief",
                    id=belief_id,
                    type=belief.belief_type.value,
                    value=belief.value,
                )
                return belief_id

    async def get(self, belief_id: str) -> BeliefMemory | None:
        """
        Get a belief by ID.

        Args:
            belief_id: Belief ID.

        Returns:
            BeliefMemory or None.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                "SELECT * FROM beliefs WHERE id = ?",
                (belief_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_belief(row)

    async def get_by_type(
        self,
        belief_type: BeliefType,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[BeliefMemory]:
        """
        Get beliefs of a specific type.

        Args:
            belief_type: Type of beliefs to retrieve.
            min_confidence: Minimum confidence threshold.
            limit: Maximum results.

        Returns:
            List of beliefs ordered by confidence.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT * FROM beliefs
                WHERE belief_type = ? AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (belief_type.value, min_confidence, limit),
            )
            rows = await cursor.fetchall()

        return [self._row_to_belief(row) for row in rows]

    async def get_top_beliefs(
        self,
        limit: int = 20,
        min_confidence: float = 0.3,
    ) -> list[BeliefMemory]:
        """
        Get top beliefs across all types.

        Args:
            limit: Maximum results.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of beliefs ordered by confidence.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                SELECT * FROM beliefs
                WHERE confidence >= ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (min_confidence, limit),
            )
            rows = await cursor.fetchall()

        return [self._row_to_belief(row) for row in rows]

    async def get_favorite_categories(
        self,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Get favorite categories with confidence scores.

        Args:
            limit: Maximum results.
            min_confidence: Minimum confidence.

        Returns:
            List of (category, confidence) tuples.
        """
        beliefs = await self.get_by_type(
            BeliefType.FAVORITE_CATEGORY,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [(b.value, b.confidence) for b in beliefs]

    async def get_favorite_authors(
        self,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Get favorite authors with confidence scores.

        Args:
            limit: Maximum results.
            min_confidence: Minimum confidence.

        Returns:
            List of (author, confidence) tuples.
        """
        beliefs = await self.get_by_type(
            BeliefType.FAVORITE_AUTHOR,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [(b.value, b.confidence) for b in beliefs]

    async def get_interest_topics(
        self,
        limit: int = 20,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Get interest topics with confidence scores.

        Args:
            limit: Maximum results.
            min_confidence: Minimum confidence.

        Returns:
            List of (topic, confidence) tuples.
        """
        beliefs = await self.get_by_type(
            BeliefType.INTEREST_TOPIC,
            min_confidence=min_confidence,
            limit=limit,
        )
        return [(b.value, b.confidence) for b in beliefs]

    async def reinforce(
        self,
        belief_type: BeliefType,
        value: str,
        strength: float = 0.1,
        source_memory_id: str | None = None,
    ) -> str:
        """
        Reinforce a belief (create if doesn't exist).

        Args:
            belief_type: Type of belief.
            value: Belief value.
            strength: Reinforcement strength.
            source_memory_id: ID of memory that led to this reinforcement.

        Returns:
            Belief ID.
        """
        belief = BeliefMemory(
            belief_type=belief_type,
            value=value,
            confidence=0.5,
            source_memory_ids=[source_memory_id] if source_memory_id else [],
        )
        return await self.store(belief)

    async def confirm(self, belief_id: str) -> None:
        """
        Mark a belief as user-confirmed (won't decay).

        Args:
            belief_id: Belief ID.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            await conn.execute(
                """
                UPDATE beliefs
                SET user_confirmed = 1, confidence = MAX(confidence, 0.8), updated_at = ?
                WHERE id = ?
                """,
                (datetime.now(UTC).isoformat(), belief_id),
            )
            await conn.commit()

        logger.debug("Confirmed belief", id=belief_id)

    async def decay_all(self) -> int:
        """
        Apply decay to all non-confirmed beliefs.

        Returns:
            Number of beliefs decayed.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                """
                UPDATE beliefs
                SET confidence = confidence * ?, updated_at = ?
                WHERE user_confirmed = 0 AND confidence > 0.1
                """,
                (self.decay_factor, datetime.now(UTC).isoformat()),
            )
            await conn.commit()
            decayed = cursor.rowcount

        logger.debug("Decayed beliefs", count=decayed)
        return decayed

    async def prune(self, min_confidence: float = 0.1) -> int:
        """
        Remove beliefs below confidence threshold.

        Args:
            min_confidence: Minimum confidence to keep.

        Returns:
            Number of beliefs removed.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            cursor = await conn.execute(
                "DELETE FROM beliefs WHERE confidence < ? AND user_confirmed = 0",
                (min_confidence,),
            )
            await conn.commit()
            pruned = cursor.rowcount

        logger.info("Pruned low-confidence beliefs", count=pruned)
        return pruned

    async def set_preference(
        self,
        belief_type: BeliefType,
        value: str,
        user_confirmed: bool = True,
    ) -> str:
        """
        Set a user preference directly.

        Args:
            belief_type: Type of preference.
            value: Preference value.
            user_confirmed: Whether this is a direct user setting.

        Returns:
            Belief ID.
        """
        belief = BeliefMemory(
            belief_type=belief_type,
            value=value,
            confidence=0.9 if user_confirmed else 0.5,
            user_confirmed=user_confirmed,
        )

        belief_id = await self.store(belief)

        if user_confirmed:
            await self.confirm(belief_id)

        return belief_id

    async def get_preferences_summary(self) -> dict[str, Any]:
        """
        Get a summary of user preferences.

        Returns:
            Dict with preference summaries.
        """
        categories = await self.get_favorite_categories(limit=5)
        authors = await self.get_favorite_authors(limit=5)
        topics = await self.get_interest_topics(limit=10)

        # Get reading level if set
        reading_levels = await self.get_by_type(BeliefType.READING_LEVEL, limit=1)
        reading_level = reading_levels[0].value if reading_levels else None

        return {
            "favorite_categories": [{"category": c, "confidence": conf} for c, conf in categories],
            "favorite_authors": [{"author": a, "confidence": conf} for a, conf in authors],
            "interest_topics": [{"topic": t, "confidence": conf} for t, conf in topics],
            "reading_level": reading_level,
        }

    async def get_stats(self) -> dict[str, Any]:
        """
        Get belief memory statistics.

        Returns:
            Dict with statistics.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            # Total beliefs
            cursor = await conn.execute("SELECT COUNT(*) as count FROM beliefs")
            row = await cursor.fetchone()
            total = row["count"] if row else 0

            # By type
            cursor = await conn.execute(
                """
                SELECT belief_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM beliefs
                GROUP BY belief_type
                """
            )
            rows = await cursor.fetchall()
            by_type = {
                r["belief_type"]: {
                    "count": r["count"],
                    "avg_confidence": round(r["avg_confidence"], 3),
                }
                for r in rows
            }

            # User confirmed
            cursor = await conn.execute(
                "SELECT COUNT(*) as count FROM beliefs WHERE user_confirmed = 1"
            )
            row = await cursor.fetchone()
            confirmed = row["count"] if row else 0

        return {
            "total_beliefs": total,
            "user_confirmed": confirmed,
            "beliefs_by_type": by_type,
        }

    async def clear(self, belief_type: BeliefType | None = None) -> int:
        """
        Clear beliefs.

        Args:
            belief_type: If provided, only clear this type.

        Returns:
            Number of beliefs deleted.
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await self._ensure_initialized(conn)
            if belief_type:
                cursor = await conn.execute(
                    "DELETE FROM beliefs WHERE belief_type = ?",
                    (belief_type.value,),
                )
            else:
                cursor = await conn.execute("DELETE FROM beliefs")

            await conn.commit()
            deleted = cursor.rowcount

        logger.info("Cleared beliefs", count=deleted, type=belief_type)
        return deleted

    def _row_to_belief(self, row: aiosqlite.Row) -> BeliefMemory:
        """Convert database row to BeliefMemory."""
        return BeliefMemory(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            belief_type=BeliefType(row["belief_type"]),
            value=row["value"],
            confidence=row["confidence"],
            reinforcement_count=row["reinforcement_count"],
            source_memory_ids=json.loads(row["source_memory_ids"]) if row["source_memory_ids"] else [],
            user_confirmed=bool(row["user_confirmed"]),
        )

    # Sync wrappers for convenience
    def get_preferences_summary_sync(self) -> dict[str, Any]:
        """Synchronous wrapper for get_preferences_summary."""
        return asyncio.get_event_loop().run_until_complete(self.get_preferences_summary())

    def reinforce_sync(self, belief_type: BeliefType, value: str, **kwargs: Any) -> str:
        """Synchronous wrapper for reinforce."""
        return asyncio.get_event_loop().run_until_complete(
            self.reinforce(belief_type, value, **kwargs)
        )


# Singleton instance
_belief_store: BeliefMemoryStore | None = None


def get_belief_store() -> BeliefMemoryStore:
    """Get or create the belief memory store singleton."""
    global _belief_store
    if _belief_store is None:
        _belief_store = BeliefMemoryStore()
    return _belief_store


if __name__ == "__main__":
    # Quick test
    async def test() -> None:
        store = BeliefMemoryStore("data/test_memory.db")

        # Add some beliefs
        await store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.CL")
        await store.reinforce(BeliefType.FAVORITE_CATEGORY, "cs.LG")
        await store.reinforce(BeliefType.INTEREST_TOPIC, "transformers")
        await store.reinforce(BeliefType.INTEREST_TOPIC, "attention mechanisms")

        # Set a preference
        await store.set_preference(BeliefType.READING_LEVEL, "expert", user_confirmed=True)

        # Get preferences
        prefs = await store.get_preferences_summary()
        print(f"Preferences: {json.dumps(prefs, indent=2)}")

        # Get stats
        stats = await store.get_stats()
        print(f"Stats: {stats}")

        # Get favorites
        categories = await store.get_favorite_categories()
        print(f"Favorite categories: {categories}")

    asyncio.run(test())
