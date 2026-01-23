"""
Agentic RAG System for PaperLens.

Implements a ReAct-based agent for paper search and comparison.
"""

from src.agent.agent import PaperLensAgent, AgentResponse, AgentError, get_agent
from src.agent.tools import Tool, ToolResult, ToolRegistry, get_tool_registry
from src.agent.planner import QueryPlanner, QueryPlan, PlanStep, QueryIntent, get_planner

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
