"""
Chat routes for PaperLens API.

Provides the main agentic chat interface for interacting with the paper search engine.
"""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.agent import AgentError, get_agent

logger = structlog.get_logger()

router = APIRouter()


# =========================================================================
# Request/Response Models
# =========================================================================


class ChatRequest(BaseModel):
    """Chat request body."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Session ID for context")


class ChatResponse(BaseModel):
    """Chat response."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID for follow-up")
    papers: list[str] = Field(default_factory=list, description="Referenced paper IDs")
    steps_taken: int = Field(default=0, description="Number of reasoning steps")


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    message_count: int
    papers_viewed: list[str]
    created_at: str | None


class ConversationMessage(BaseModel):
    """A message in the conversation."""

    role: str
    content: str
    timestamp: str | None = None


class ConversationResponse(BaseModel):
    """Conversation history response."""

    session_id: str
    messages: list[ConversationMessage]
    total: int


# =========================================================================
# Chat Endpoints
# =========================================================================


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the PaperLens agent.

    The agent will:
    1. Analyze your query
    2. Search for relevant papers
    3. Provide a helpful response

    Use the session_id to maintain conversation context across requests.
    """
    try:
        agent = get_agent()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())

        logger.info(
            "Chat request received",
            session_id=session_id,
            message_length=len(request.message),
        )

        # Run the agent
        result = agent.run(request.message, session_id=session_id)

        logger.info(
            "Chat response generated",
            session_id=session_id,
            steps=len(result.steps),
            papers=len(result.papers),
        )

        return ChatResponse(
            response=result.response,
            session_id=result.session_id,
            papers=result.papers,
            steps_taken=len(result.steps),
        )

    except AgentError as e:
        logger.error("Agent error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}") from e
    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Send a message and stream the response.

    Returns a Server-Sent Events stream with the agent's response.
    Note: Currently returns the full response as streaming is complex with ReAct.
    """
    try:
        agent = get_agent()
        session_id = request.session_id or str(uuid4())

        async def generate() -> AsyncIterator[str]:
            try:
                # For now, we run the full agent and return the result
                # True streaming would require restructuring the ReAct loop
                result = agent.run(request.message, session_id=session_id)

                # Send metadata first
                yield "data: {\n"
                yield f'data:   "session_id": "{result.session_id}",\n'
                yield f'data:   "papers": {result.papers},\n'
                yield f'data:   "steps": {len(result.steps)}\n'
                yield "data: }\n\n"

                # Send response in chunks to simulate streaming
                chunk_size = 50
                response = result.response
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    yield f"data: {chunk}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error("Stream chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# =========================================================================
# Session Endpoints
# =========================================================================


@router.get("/chat/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    """
    Get information about a chat session.

    Returns message count and papers viewed in the session.
    """
    try:
        agent = get_agent()

        if not agent.memory.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        state = agent.memory.get_session(session_id)

        return SessionInfo(
            session_id=session_id,
            message_count=len(state.messages),
            papers_viewed=state.retrieved_paper_ids,
            created_at=state.created_at.isoformat() if hasattr(state, 'created_at') else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session info", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chat/session/{session_id}/history", response_model=ConversationResponse)
async def get_conversation_history(session_id: str) -> ConversationResponse:
    """
    Get the conversation history for a session.

    Returns all messages exchanged in the session.
    """
    try:
        agent = get_agent()

        if not agent.memory.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        messages = agent.get_session_history(session_id)

        return ConversationResponse(
            session_id=session_id,
            messages=[
                ConversationMessage(
                    role=m["role"],
                    content=m["content"],
                )
                for m in messages
            ],
            total=len(messages),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get history", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, str]:
    """
    Clear a chat session.

    Removes all conversation history and context for the session.
    """
    try:
        agent = get_agent()

        if not agent.memory.session_exists(session_id):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        agent.clear_session(session_id)

        logger.info("Session cleared", session_id=session_id)

        return {"message": f"Session {session_id} cleared"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to clear session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/session")
async def create_session() -> dict[str, str]:
    """
    Create a new chat session.

    Returns a new session ID that can be used for subsequent chat requests.
    """
    try:
        agent = get_agent()
        session_id = str(uuid4())

        # Initialize the session
        agent.memory.get_session(session_id)

        logger.info("Session created", session_id=session_id)

        return {"session_id": session_id}

    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# =========================================================================
# Utility Endpoints
# =========================================================================


class ToolsResponse(BaseModel):
    """Response with available tools."""

    tools: list[dict[str, Any]]


@router.get("/chat/tools", response_model=ToolsResponse)
async def get_available_tools() -> ToolsResponse:
    """
    Get list of tools available to the agent.

    Returns tool names and descriptions.
    """
    try:
        agent = get_agent()
        schemas = agent.tools.get_schemas()

        tools = [
            {
                "name": s["function"]["name"],
                "description": s["function"]["description"],
                "parameters": list(s["function"]["parameters"]["properties"].keys()),
            }
            for s in schemas
        ]

        return ToolsResponse(tools=tools)

    except Exception as e:
        logger.error("Failed to get tools", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
