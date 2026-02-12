"""
Memory data models for PaperLens.

Defines the data structures for the agentic memory system:
- MemoryItem: Base memory item
- EpisodicMemory: Interaction history
- BeliefMemory: User preferences
- WorkingMemoryState: Current session state
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Types of memory in the system."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    WORKING = "working"
    BELIEF = "belief"


class MemoryItem(BaseModel):
    """Base class for all memory items."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Last update timestamp")
    memory_type: MemoryType = Field(..., description="Type of memory")

    def touch(self) -> None:
        """Update the timestamp."""
        self.updated_at = datetime.now(UTC)


class EpisodicMemory(MemoryItem):
    """
    Episodic memory for storing interaction history.

    Stores past queries, results, and user feedback to enable
    contextual retrieval like "papers I searched for last week".
    """

    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)

    # Query information
    query: str = Field(..., description="The user's search query")
    query_embedding: list[float] | None = Field(default=None, description="Query embedding for similarity search")

    # Results
    result_paper_ids: list[str] = Field(default_factory=list, description="ArXiv IDs of returned papers")
    result_count: int = Field(default=0, description="Number of results returned")

    # User feedback
    feedback: str | None = Field(default=None, description="User feedback (positive/negative/neutral)")
    liked_paper_ids: list[str] = Field(default_factory=list, description="Papers the user liked")
    disliked_paper_ids: list[str] = Field(default_factory=list, description="Papers the user disliked")

    # Context
    session_id: str | None = Field(default=None, description="Session this memory belongs to")
    action_type: str = Field(default="search", description="Type of action (search, compare, summarize)")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_feedback(self, paper_id: str, liked: bool) -> None:
        """Add feedback for a paper."""
        if liked:
            if paper_id not in self.liked_paper_ids:
                self.liked_paper_ids.append(paper_id)
            if paper_id in self.disliked_paper_ids:
                self.disliked_paper_ids.remove(paper_id)
        else:
            if paper_id not in self.disliked_paper_ids:
                self.disliked_paper_ids.append(paper_id)
            if paper_id in self.liked_paper_ids:
                self.liked_paper_ids.remove(paper_id)
        self.touch()


class BeliefType(StrEnum):
    """Types of user beliefs/preferences."""

    FAVORITE_CATEGORY = "favorite_category"
    FAVORITE_AUTHOR = "favorite_author"
    PREFERRED_YEAR_RANGE = "preferred_year_range"
    READING_LEVEL = "reading_level"
    INTEREST_TOPIC = "interest_topic"
    DISLIKED_TOPIC = "disliked_topic"


class BeliefMemory(MemoryItem):
    """
    Belief memory for storing user preferences.

    Tracks learned patterns from user interactions with confidence scoring.
    Beliefs decay over time if not reinforced.
    """

    memory_type: MemoryType = Field(default=MemoryType.BELIEF)

    # Belief content
    belief_type: BeliefType = Field(..., description="Type of belief")
    value: str = Field(..., description="The belief value (e.g., 'transformers', 'cs.CL')")

    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this belief")
    reinforcement_count: int = Field(default=1, description="Number of times this belief was reinforced")

    # Source
    source_memory_ids: list[str] = Field(default_factory=list, description="Episodic memories that led to this belief")

    # User override
    user_confirmed: bool = Field(default=False, description="Whether user explicitly confirmed this")

    def reinforce(self, strength: float = 0.1) -> None:
        """
        Reinforce this belief, increasing confidence.

        Args:
            strength: How much to increase confidence (0-1).
        """
        self.confidence = min(1.0, self.confidence + strength * (1 - self.confidence))
        self.reinforcement_count += 1
        self.touch()

    def decay(self, factor: float = 0.95) -> None:
        """
        Apply decay to this belief's confidence.

        Args:
            factor: Decay multiplier (0-1).
        """
        if not self.user_confirmed:  # Don't decay user-confirmed beliefs
            self.confidence *= factor
            self.touch()


class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """A single step in the agent's reasoning process."""

    step_number: int = Field(..., description="Step number in the sequence")
    thought: str = Field(..., description="Agent's reasoning")
    action: str | None = Field(default=None, description="Tool to use")
    action_input: dict[str, Any] | None = Field(default=None, description="Tool parameters")
    observation: str | None = Field(default=None, description="Result of the action")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkingMemoryState(MemoryItem):
    """
    Working memory for current session state.

    Maintains conversation context, retrieved papers, and reasoning steps.
    Cleared when session ends.
    """

    memory_type: MemoryType = Field(default=MemoryType.WORKING)

    # Session
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Current session ID")

    # Conversation
    messages: list[ConversationMessage] = Field(default_factory=list, description="Conversation history")

    # Retrieved context
    retrieved_paper_ids: list[str] = Field(default_factory=list, description="Papers retrieved this session")
    current_query: str | None = Field(default=None, description="Current user query being processed")

    # Agent reasoning
    agent_steps: list[AgentStep] = Field(default_factory=list, description="Agent reasoning steps")
    current_plan: list[str] | None = Field(default=None, description="Current execution plan")

    # Temporary storage
    scratch_pad: dict[str, Any] = Field(default_factory=dict, description="Temporary data storage")

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(role=role, content=content, metadata=metadata))
        self.touch()

    def add_step(
        self,
        thought: str,
        action: str | None = None,
        action_input: dict[str, Any] | None = None,
        observation: str | None = None,
    ) -> AgentStep:
        """Add a reasoning step."""
        step = AgentStep(
            step_number=len(self.agent_steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        self.agent_steps.append(step)
        self.touch()
        return step

    def add_paper(self, arxiv_id: str) -> None:
        """Track a retrieved paper."""
        if arxiv_id not in self.retrieved_paper_ids:
            self.retrieved_paper_ids.append(arxiv_id)
            self.touch()

    def get_conversation_text(self, max_messages: int | None = None) -> str:
        """
        Get conversation history as formatted text.

        Args:
            max_messages: Maximum messages to include (None = all).

        Returns:
            Formatted conversation string.
        """
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]

        lines = []
        for msg in messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all working memory (end of session)."""
        self.messages = []
        self.retrieved_paper_ids = []
        self.current_query = None
        self.agent_steps = []
        self.current_plan = None
        self.scratch_pad = {}
        self.touch()


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""

    query_text: str | None = Field(default=None, description="Text to search for")
    query_embedding: list[float] | None = Field(default=None, description="Embedding for similarity search")
    memory_types: list[MemoryType] | None = Field(default=None, description="Filter by memory type")
    session_id: str | None = Field(default=None, description="Filter by session")
    time_from: datetime | None = Field(default=None, description="Filter by time (from)")
    time_to: datetime | None = Field(default=None, description="Filter by time (to)")
    limit: int = Field(default=10, description="Maximum results to return")


class MemorySearchResult(BaseModel):
    """Result of a memory search."""

    memory: MemoryItem
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    match_reason: str | None = Field(default=None, description="Why this memory matched")
