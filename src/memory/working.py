"""
Working Memory for PaperLens.

Maintains current session state including:
- Conversation history
- Retrieved papers
- Agent reasoning steps
- Temporary scratch pad

Cleared when session ends.
"""

from typing import Any
from uuid import uuid4

import structlog

from src.config import settings
from src.models.memory import AgentStep, ConversationMessage, WorkingMemoryState

logger = structlog.get_logger()


class WorkingMemory:
    """
    In-memory working memory for current session.

    Stores conversation context, retrieved papers, and agent reasoning.
    Each session has its own isolated working memory state.
    """

    def __init__(self, max_size: int | None = None):
        """
        Initialize working memory.

        Args:
            max_size: Maximum items to keep in memory. Defaults to config value.
        """
        self.max_size = max_size or settings.memory_working_size
        self._sessions: dict[str, WorkingMemoryState] = {}

        logger.info("Working memory initialized", max_size=self.max_size)

    def get_session(self, session_id: str | None = None) -> WorkingMemoryState:
        """
        Get or create a session's working memory.

        Args:
            session_id: Session ID. Creates new session if None.

        Returns:
            Working memory state for the session.
        """
        if session_id is None:
            session_id = str(uuid4())

        if session_id not in self._sessions:
            self._sessions[session_id] = WorkingMemoryState(session_id=session_id)
            logger.debug("Created new session", session_id=session_id)

        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **metadata: Any,
    ) -> ConversationMessage:
        """
        Add a message to the conversation.

        Args:
            session_id: Session ID.
            role: Message role (user, assistant, system).
            content: Message content.
            **metadata: Additional metadata.

        Returns:
            The created message.
        """
        state = self.get_session(session_id)
        state.add_message(role, content, **metadata)

        # Enforce max size by removing oldest messages
        if len(state.messages) > self.max_size:
            removed = len(state.messages) - self.max_size
            state.messages = state.messages[-self.max_size:]
            logger.debug(
                "Trimmed conversation history",
                session_id=session_id,
                removed=removed,
            )

        return state.messages[-1]

    def add_step(
        self,
        session_id: str,
        thought: str,
        action: str | None = None,
        action_input: dict[str, Any] | None = None,
        observation: str | None = None,
    ) -> AgentStep:
        """
        Add an agent reasoning step.

        Args:
            session_id: Session ID.
            thought: Agent's reasoning.
            action: Tool to use.
            action_input: Tool parameters.
            observation: Result of the action.

        Returns:
            The created step.
        """
        state = self.get_session(session_id)
        step = state.add_step(thought, action, action_input, observation)

        logger.debug(
            "Added agent step",
            session_id=session_id,
            step_number=step.step_number,
            action=action,
        )

        return step

    def update_step_observation(
        self,
        session_id: str,
        step_number: int,
        observation: str,
    ) -> None:
        """
        Update the observation for an existing step.

        Args:
            session_id: Session ID.
            step_number: Step number to update.
            observation: Observation to set.
        """
        state = self.get_session(session_id)

        for step in state.agent_steps:
            if step.step_number == step_number:
                step.observation = observation
                state.touch()
                logger.debug(
                    "Updated step observation",
                    session_id=session_id,
                    step_number=step_number,
                )
                return

        logger.warning(
            "Step not found",
            session_id=session_id,
            step_number=step_number,
        )

    def add_paper(self, session_id: str, arxiv_id: str) -> None:
        """
        Track a retrieved paper.

        Args:
            session_id: Session ID.
            arxiv_id: Paper ArXiv ID.
        """
        state = self.get_session(session_id)
        state.add_paper(arxiv_id)

        logger.debug(
            "Added paper to working memory",
            session_id=session_id,
            arxiv_id=arxiv_id,
        )

    def set_query(self, session_id: str, query: str) -> None:
        """
        Set the current query being processed.

        Args:
            session_id: Session ID.
            query: User query.
        """
        state = self.get_session(session_id)
        state.current_query = query
        state.touch()

    def set_plan(self, session_id: str, plan: list[str]) -> None:
        """
        Set the current execution plan.

        Args:
            session_id: Session ID.
            plan: List of planned steps.
        """
        state = self.get_session(session_id)
        state.current_plan = plan
        state.touch()

    def set_scratch(self, session_id: str, key: str, value: Any) -> None:
        """
        Store temporary data in scratch pad.

        Args:
            session_id: Session ID.
            key: Data key.
            value: Data value.
        """
        state = self.get_session(session_id)
        state.scratch_pad[key] = value
        state.touch()

    def get_scratch(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        Retrieve data from scratch pad.

        Args:
            session_id: Session ID.
            key: Data key.
            default: Default value if not found.

        Returns:
            Stored value or default.
        """
        state = self.get_session(session_id)
        return state.scratch_pad.get(key, default)

    def get_conversation(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[ConversationMessage]:
        """
        Get conversation history.

        Args:
            session_id: Session ID.
            max_messages: Maximum messages to return.

        Returns:
            List of conversation messages.
        """
        state = self.get_session(session_id)
        messages = state.messages

        if max_messages:
            messages = messages[-max_messages:]

        return messages

    def get_messages_for_llm(
        self,
        session_id: str,
        system_prompt: str | None = None,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Get conversation formatted for LLM API.

        Args:
            session_id: Session ID.
            system_prompt: Optional system prompt to prepend.
            max_messages: Maximum messages to include.

        Returns:
            List of message dicts for LLM API.
        """
        messages = self.get_conversation(session_id, max_messages)

        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    def get_steps(self, session_id: str) -> list[AgentStep]:
        """
        Get agent reasoning steps.

        Args:
            session_id: Session ID.

        Returns:
            List of agent steps.
        """
        state = self.get_session(session_id)
        return state.agent_steps

    def get_retrieved_papers(self, session_id: str) -> list[str]:
        """
        Get list of retrieved paper IDs.

        Args:
            session_id: Session ID.

        Returns:
            List of ArXiv IDs.
        """
        state = self.get_session(session_id)
        return state.retrieved_paper_ids

    def get_context_summary(self, session_id: str) -> dict[str, Any]:
        """
        Get a summary of the current session context.

        Args:
            session_id: Session ID.

        Returns:
            Dict with context summary.
        """
        state = self.get_session(session_id)

        return {
            "session_id": session_id,
            "message_count": len(state.messages),
            "retrieved_paper_count": len(state.retrieved_paper_ids),
            "step_count": len(state.agent_steps),
            "current_query": state.current_query,
            "has_plan": state.current_plan is not None,
            "scratch_keys": list(state.scratch_pad.keys()),
        }

    def clear_session(self, session_id: str) -> None:
        """
        Clear a session's working memory.

        Args:
            session_id: Session ID.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info("Cleared session", session_id=session_id)

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session entirely.

        Args:
            session_id: Session ID.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Deleted session", session_id=session_id)

    def clear_steps(self, session_id: str) -> None:
        """
        Clear agent reasoning steps (start fresh reasoning).

        Args:
            session_id: Session ID.
        """
        state = self.get_session(session_id)
        state.agent_steps = []
        state.current_plan = None
        state.touch()

        logger.debug("Cleared agent steps", session_id=session_id)

    def list_sessions(self) -> list[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs.
        """
        return list(self._sessions.keys())

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session ID.

        Returns:
            True if session exists.
        """
        return session_id in self._sessions


# Singleton instance
_working_memory: WorkingMemory | None = None


def get_working_memory() -> WorkingMemory:
    """Get or create the working memory singleton."""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory()
    return _working_memory


if __name__ == "__main__":
    # Quick test
    memory = WorkingMemory()

    # Create a session
    session = memory.get_session()
    session_id = session.session_id

    print(f"Session ID: {session_id}")

    # Add messages
    memory.add_message(session_id, "user", "Find papers about transformers")
    memory.add_message(session_id, "assistant", "I'll search for transformer papers.")

    # Add a step
    memory.add_step(
        session_id,
        thought="User wants transformer papers. I should search.",
        action="search_papers",
        action_input={"query": "transformers"},
    )

    # Add a paper
    memory.add_paper(session_id, "1706.03762")

    # Get context
    context = memory.get_context_summary(session_id)
    print(f"\nContext: {context}")

    # Get messages for LLM
    messages = memory.get_messages_for_llm(
        session_id,
        system_prompt="You are a helpful assistant.",
    )
    print(f"\nMessages for LLM: {messages}")
