"""
Agentic RAG System for PaperLens.

Implements a ReAct-based agent for paper search and comparison.
"""

from src.agent.agent import AgentError, AgentResponse, PaperLensAgent, get_agent
from src.agent.planner import PlanStep, QueryIntent, QueryPlan, QueryPlanner, get_planner
from src.agent.tools import Tool, ToolRegistry, ToolResult, get_tool_registry

__all__ = [
    # Agent
    "PaperLensAgent",
    "AgentResponse",
    "AgentError",
    "get_agent",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    # Planner
    "QueryPlanner",
    "QueryPlan",
    "PlanStep",
    "QueryIntent",
    "get_planner",
]
