"""
Main ReAct Agent for PaperLens.

Implements the ReAct (Reasoning + Acting) pattern for paper search and analysis.
The agent reasons about queries, uses tools, and synthesizes responses.
"""

import asyncio
import concurrent.futures
import json
import re
from typing import Any
from uuid import uuid4

import structlog

from src.agent.planner import QueryPlanner, get_planner
from src.agent.prompts import (
    SYSTEM_PROMPT,
    format_conversation_context,
    format_react_prompt,
)
from src.agent.tools import ToolRegistry, ToolResult, get_tool_registry
from src.config import settings
from src.memory.manager import MemoryManager, get_memory_manager
from src.memory.working import WorkingMemory, get_working_memory
from src.models.memory import AgentStep
from src.services.llm import LLMService, get_llm_service

logger = structlog.get_logger()


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


class MaxIterationsError(AgentError):
    """Agent exceeded maximum iterations."""

    pass


class ParseError(AgentError):
    """Failed to parse agent response."""

    pass


class AgentResponse:
    """Response from the agent."""

    def __init__(
        self,
        response: str,
        session_id: str,
        steps: list[AgentStep],
        papers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.response = response
        self.session_id = session_id
        self.steps = steps
        self.papers = papers or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "session_id": self.session_id,
            "steps_count": len(self.steps),
            "papers_referenced": self.papers,
            "metadata": self.metadata,
        }


class PaperLensAgent:
    """
    ReAct agent for paper search and analysis.

    Implements the ReAct pattern:
    1. THINK - Reason about what to do
    2. ACT - Execute a tool
    3. OBSERVE - Process the result
    4. REPEAT or RESPOND
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        tool_registry: ToolRegistry | None = None,
        working_memory: WorkingMemory | None = None,
        memory_manager: MemoryManager | None = None,
        planner: QueryPlanner | None = None,
        max_iterations: int | None = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_service: LLM service for reasoning.
            tool_registry: Registry of available tools.
            working_memory: Working memory for session state.
            memory_manager: Full memory manager for personalization.
            planner: Query planner for decomposition.
            max_iterations: Maximum ReAct iterations.
        """
        self.llm = llm_service or get_llm_service()
        self.tools = tool_registry or get_tool_registry()
        self.memory = working_memory or get_working_memory()
        self.memory_manager = memory_manager or get_memory_manager()
        self.planner = planner or get_planner()
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        logger.info(
            "Agent initialized",
            tools=self.tools.list_tools(),
            max_iterations=self.max_iterations,
        )

    def run(
        self,
        query: str,
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Run the agent on a query.

        Args:
            query: User query to process.
            session_id: Session ID for context. Creates new if None.

        Returns:
            AgentResponse with the result.

        Raises:
            MaxIterationsError: If max iterations exceeded.
            AgentError: For other agent failures.
        """
        # Initialize session
        session_id = session_id or str(uuid4())
        self.memory.get_session(session_id)

        # Clear previous reasoning steps for new query
        self.memory.clear_steps(session_id)
        self.memory.set_query(session_id, query)

        # Add user message to history
        self.memory.add_message(session_id, "user", query)

        logger.info("Agent run started", session_id=session_id, query=query[:50])

        try:
            # Run ReAct loop
            response = self._react_loop(query, session_id)

            # Add assistant response to history
            self.memory.add_message(session_id, "assistant", response)

            # Get steps and papers
            steps = self.memory.get_steps(session_id)
            papers = self.memory.get_retrieved_papers(session_id)

            logger.info(
                "Agent run completed",
                session_id=session_id,
                steps=len(steps),
                papers=len(papers),
            )

            return AgentResponse(
                response=response,
                session_id=session_id,
                steps=steps,
                papers=papers,
                metadata={"iterations": len(steps)},
            )

        except Exception as e:
            logger.error("Agent run failed", session_id=session_id, error=str(e))
            raise AgentError(f"Agent failed: {e}") from e

    def _react_loop(self, query: str, session_id: str) -> str:
        """
        Execute the ReAct loop.

        Args:
            query: User query.
            session_id: Session ID.

        Returns:
            Final response string.
        """
        # Build context
        context = self._build_context(session_id)

        # Get planner guidance for non-trivial queries
        plan_guidance = self._get_plan_guidance(query)
        if plan_guidance:
            context += f"\n\n## Query Analysis\n{plan_guidance}"
            plan_lines = [line.strip() for line in plan_guidance.split("\n") if line.strip()]
            self.memory.set_plan(session_id, plan_lines)

        # Get tool schemas
        tool_schemas = self.tools.get_schemas()

        for iteration in range(self.max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}")

            # Build prompt
            prompt = format_react_prompt(query, tool_schemas, context)

            # Get agent messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Add reasoning history from current session
            steps = self.memory.get_steps(session_id)
            if steps:
                history = self._format_steps_for_prompt(steps)
                messages.append({"role": "assistant", "content": history})
                messages.append({"role": "user", "content": "Continue your reasoning:"})

            # Generate response
            response = self.llm.chat_completion(
                messages,
                temperature=settings.agent_temperature,
                stop=["OBSERVATION:"],  # Stop before we inject observation
            )

            # Parse response
            parsed = self._parse_response(response)

            if parsed["type"] == "final_answer":
                # Agent is done
                logger.debug("Agent produced final answer")
                return str(parsed["content"])

            elif parsed["type"] == "action":
                # Execute tool
                thought = parsed["thought"]
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Record step
                step = self.memory.add_step(
                    session_id,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                )

                # Execute tool
                result = self._execute_tool(action, action_input, session_id)

                # Update step with observation
                observation = self._format_observation(result)
                self.memory.update_step_observation(session_id, step.step_number, observation)

                # Update context for next iteration
                context = self._build_context(session_id)

            else:
                # Couldn't parse - try to recover
                logger.warning("Failed to parse response", response=response[:100])
                # Add as thought and continue
                self.memory.add_step(session_id, thought=response, action=None)

        # Max iterations reached — return best partial response
        logger.warning("Max iterations reached", session_id=session_id)
        return self._build_partial_response(session_id)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse agent response to extract thought/action/answer.

        Args:
            response: Raw LLM response.

        Returns:
            Dict with parsed components.
        """
        response = response.strip()

        # Check for final answer
        if "FINAL_ANSWER:" in response:
            parts = response.split("FINAL_ANSWER:", 1)
            thought = ""
            if "THOUGHT:" in parts[0]:
                thought = parts[0].split("THOUGHT:", 1)[1].strip()
            return {
                "type": "final_answer",
                "thought": thought,
                "content": parts[1].strip(),
            }

        # Check for action
        if "ACTION:" in response:
            # Extract thought
            thought = ""
            if "THOUGHT:" in response:
                thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
                if thought_match:
                    thought = thought_match.group(1).strip()

            # Extract action
            action_match = re.search(r"ACTION:\s*(\w+)", response)
            if not action_match:
                return {"type": "unknown", "content": response}
            action = action_match.group(1)

            # Extract action input
            action_input = {}
            input_match = re.search(r"ACTION_INPUT:\s*(\{.+?\})", response, re.DOTALL)
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse action input", input=input_match.group(1))

            return {
                "type": "action",
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }

        # Couldn't parse - return as-is
        return {"type": "unknown", "content": response}

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        session_id: str,
    ) -> ToolResult:
        """
        Execute a tool and track papers.

        Args:
            tool_name: Name of tool to execute.
            tool_input: Tool parameters.
            session_id: Session ID for tracking.

        Returns:
            Tool execution result.
        """
        logger.debug("Executing tool", tool=tool_name, input=tool_input)

        result = self.tools.execute(tool_name, **tool_input)

        # Track retrieved papers
        if result.success and result.data:
            self._track_papers(result.data, session_id)

            # Record search to episodic memory for later recall
            if tool_name == "search_papers" and isinstance(result.data, list):
                self._record_search_to_memory(
                    query=tool_input.get("query", ""),
                    results=result.data,
                    session_id=session_id,
                )

        return result

    def _record_search_to_memory(
        self,
        query: str,
        results: list[dict[str, Any]],
        session_id: str,
    ) -> None:
        """Record a search to episodic memory."""
        try:
            from src.models.memory import EpisodicMemory

            memory = EpisodicMemory(
                query=query,
                session_id=session_id,
                action_type="search",
                result_paper_ids=[r.get("arxiv_id", "") for r in results if r.get("arxiv_id")],
                result_count=len(results),
            )

            self._run_async(self.memory_manager.episodic.store(memory))
            logger.debug("Recorded search to episodic memory", query=query[:30])
        except Exception as e:
            # Don't fail the main operation if memory recording fails
            logger.debug("Failed to record search to memory", error=str(e))

    def _track_papers(self, data: Any, session_id: str) -> None:
        """Track papers from tool results in working memory."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "arxiv_id" in item:
                    self.memory.add_paper(session_id, item["arxiv_id"])
        elif isinstance(data, dict):
            if "arxiv_id" in data:
                self.memory.add_paper(session_id, data["arxiv_id"])
            if "papers" in data and isinstance(data["papers"], list):
                for p in data["papers"]:
                    if isinstance(p, dict) and "arxiv_id" in p:
                        self.memory.add_paper(session_id, p["arxiv_id"])

    def _format_observation(self, result: ToolResult) -> str:
        """Format tool result as observation string."""
        if not result.success:
            return f"Error: {result.error}"

        data = result.data

        # Format based on data type
        if isinstance(data, str):
            return data

        if isinstance(data, list):
            if not data:
                return "No results found."
            # Format as list
            if isinstance(data[0], dict):
                items = []
                for i, item in enumerate(data[:10]):  # Limit to 10
                    if "title" in item:
                        items.append(f"{i+1}. {item['title']} ({item.get('arxiv_id', 'N/A')})")
                    else:
                        items.append(f"{i+1}. {str(item)[:100]}")
                return f"Found {len(data)} results:\n" + "\n".join(items)
            return f"Found {len(data)} items."

        if isinstance(data, dict):
            # Check for specific response types
            if "comparison" in data:
                return f"Comparison generated:\n{data['comparison'][:500]}..."
            if "summary" in data:
                return f"Summary:\n{data['summary'][:500]}..."
            if "title" in data:
                return f"Paper: {data['title']} ({data.get('arxiv_id', 'N/A')})"
            return json.dumps(data, indent=2)[:500]

        return str(data)[:500]

    def _format_steps_for_prompt(self, steps: list[AgentStep]) -> str:
        """Format reasoning steps for inclusion in prompt."""
        lines = []
        for step in steps:
            lines.append(f"THOUGHT: {step.thought}")
            if step.action:
                lines.append(f"ACTION: {step.action}")
                if step.action_input:
                    lines.append(f"ACTION_INPUT: {json.dumps(step.action_input)}")
            if step.observation:
                lines.append(f"OBSERVATION: {step.observation}")
            lines.append("")
        return "\n".join(lines)

    def _build_partial_response(self, session_id: str) -> str:
        """Build the best possible response from partial reasoning steps.

        Called when the agent hits max iterations without producing a FINAL_ANSWER.

        Args:
            session_id: Session ID.

        Returns:
            Best partial response string.
        """
        steps = self.memory.get_steps(session_id)
        if not steps:
            return (
                "I was unable to complete my analysis within the allowed number of steps. "
                "Please try rephrasing your question or breaking it into simpler parts."
            )

        # Gather all observations (tool results) as useful context
        observations = []
        last_thought = ""
        for step in steps:
            if step.thought:
                last_thought = step.thought
            if step.observation and not step.observation.startswith("Error:"):
                observations.append(step.observation)

        if observations:
            obs_text = "\n\n".join(observations)
            return (
                "I gathered some information but could not fully complete my analysis. "
                f"Here is what I found:\n\n{obs_text}\n\n"
                "You may want to refine your query for more specific results."
            )

        if last_thought:
            return (
                "I was still working through the analysis. "
                f"My last reasoning: {last_thought}\n\n"
                "Please try a more focused query."
            )

        return (
            "I was unable to complete the analysis within the allowed steps. "
            "Please try a simpler or more specific question."
        )

    def _build_context(self, session_id: str) -> str:
        """Build context string for the agent."""
        state = self.memory.get_session(session_id)

        # Get conversation history (formatted)
        history = [{"role": m.role, "content": m.content} for m in state.messages[-5:]]

        # Basic context from working memory
        basic_context = format_conversation_context(
            history=history,
            papers=state.retrieved_paper_ids,
            current_query=state.current_query,
        )

        # Try to add rich context from memory manager (beliefs, episodic)
        try:
            rich_context = self._run_async(
                self.memory_manager.build_context(
                    session_id=session_id,
                    query=state.current_query,
                    include_beliefs=True,
                    include_episodic=True,
                    max_episodic=3,
                )
            )
            formatted_rich = self.memory_manager.format_context_for_prompt(rich_context)
            if formatted_rich and formatted_rich != "No prior context.":
                basic_context += f"\n\n## User Profile\n{formatted_rich}"
        except Exception as e:
            logger.debug("Could not build rich context", error=str(e))

        return basic_context

    def _get_plan_guidance(self, query: str) -> str:
        """Use the query planner to generate guidance for the ReAct loop.

        Args:
            query: User query.

        Returns:
            Guidance string to prepend to context, or empty string.
        """
        try:
            plan = self.planner.plan(query, use_llm=False)
            if plan.intent == "search" and len(plan.steps) == 1:
                return ""

            guidance_parts = [f"Detected intent: {plan.intent}"]

            for i, step in enumerate(plan.steps):
                tool_str = f" using {step.tool}" if step.tool else ""
                guidance_parts.append(f"  Step {i + 1}: {step.task}{tool_str}")

            if plan.requires_comparison:
                guidance_parts.append("This query requires comparing papers.")
            if plan.requires_summary:
                guidance_parts.append("This query requires summarizing a paper.")

            return "\n".join(guidance_parts)
        except Exception as e:
            logger.debug("Planner guidance failed", error=str(e))
            return ""

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an async context — run in a separate thread
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = self._executor.submit(asyncio.run, coro)
            return future.result(timeout=30)
        else:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(coro)

    def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> str:
        """
        Simple chat interface.

        Args:
            message: User message.
            session_id: Session ID.

        Returns:
            Agent response string.
        """
        response = self.run(message, session_id)
        return response.response

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID.

        Returns:
            List of message dicts.
        """
        messages = self.memory.get_conversation(session_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's state."""
        self.memory.clear_session(session_id)


# Singleton instance
_agent: PaperLensAgent | None = None


def get_agent() -> PaperLensAgent:
    """Get or create the agent singleton."""
    global _agent
    if _agent is None:
        _agent = PaperLensAgent()
    return _agent


if __name__ == "__main__":
    # Quick test (requires Groq API key and indexed papers)
    import sys

    agent = PaperLensAgent()

    print("PaperLens Agent Test")
    print("=" * 50)

    # Test query
    query = sys.argv[1] if len(sys.argv) > 1 else "Find papers about transformer attention"

    print(f"\nQuery: {query}")
    print("-" * 50)

    try:
        response = agent.run(query)
        print(f"\nResponse:\n{response.response}")
        print(f"\nPapers referenced: {response.papers}")
        print(f"Steps taken: {len(response.steps)}")
    except Exception as e:
        print(f"Error: {e}")
