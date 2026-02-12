"""
Agent Tools for PaperLens.

Defines the tools available to the ReAct agent for paper search and analysis.
Each tool has a schema for validation and structured output.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.config import settings
from src.memory.semantic import SemanticMemory, get_semantic_memory
from src.services.llm import LLMService, get_llm_service

logger = structlog.get_logger()


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool = Field(..., description="Whether the tool succeeded")
    data: Any = Field(default=None, description="Result data")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_observation(self) -> str:
        """Convert to observation string for agent."""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, list):
            return f"Found {len(self.data)} results"
        return str(self.data)


class Tool(ABC):
    """Base class for agent tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema in OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items() if v.get("required", False)],
                },
            },
        }


class SearchPapersTool(Tool):
    """Search for papers by semantic query."""

    name = "search_papers"
    description = (
        "Search for academic papers using natural language. "
        "Returns papers ranked by semantic similarity to the query. "
        "Use filters to narrow results by year or category."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Natural language search query",
            "required": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of papers to return (default: 10)",
            "required": False,
        },
        "year_from": {
            "type": "integer",
            "description": "Filter papers published from this year onwards",
            "required": False,
        },
        "year_to": {
            "type": "integer",
            "description": "Filter papers published up to this year",
            "required": False,
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ArXiv categories to filter by (e.g., ['cs.CL', 'cs.LG'])",
            "required": False,
        },
    }

    def __init__(self, semantic_memory: SemanticMemory | None = None):
        self.semantic_memory = semantic_memory or get_semantic_memory()

    def execute(  # type: ignore[override]
        self,
        query: str,
        limit: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        categories: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute paper search."""
        try:
            limit = limit or settings.search_top_k

            results = self.semantic_memory.search(
                query=query,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                categories=categories,
            )

            # Format results for agent
            formatted = []
            for r in results:
                formatted.append({
                    "arxiv_id": r.paper.arxiv_id,
                    "title": r.paper.title,
                    "year": r.paper.year,
                    "score": round(r.score, 3),
                    "abstract_preview": r.paper.abstract[:200] + "..." if len(r.paper.abstract) > 200 else r.paper.abstract,
                })

            logger.info("Search executed", query=query[:30], results=len(results))

            return ToolResult(
                success=True,
                data=formatted,
                metadata={"query": query, "total_results": len(results)},
            )
        except Exception as e:
            logger.error("Search failed", error=str(e))
            return ToolResult(success=False, error=str(e))


class GetPaperTool(Tool):
    """Retrieve full details of a specific paper."""

    name = "get_paper"
    description = (
        "Get full details of a paper by its ArXiv ID. "
        "Returns title, abstract, authors, categories, and more."
    )
    parameters = {
        "arxiv_id": {
            "type": "string",
            "description": "The ArXiv ID of the paper (e.g., '2301.12345')",
            "required": True,
        },
    }

    def __init__(self, semantic_memory: SemanticMemory | None = None):
        self.semantic_memory = semantic_memory or get_semantic_memory()

    def execute(self, arxiv_id: str, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        """Get paper details."""
        try:
            paper = self.semantic_memory.get_paper(arxiv_id)

            if not paper:
                return ToolResult(
                    success=False,
                    error=f"Paper not found: {arxiv_id}",
                )

            data = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "categories": paper.categories,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.arxiv_url,
            }

            logger.info("Paper retrieved", arxiv_id=arxiv_id)

            return ToolResult(success=True, data=data)
        except Exception as e:
            logger.error("Get paper failed", arxiv_id=arxiv_id, error=str(e))
            return ToolResult(success=False, error=str(e))


class GetRelatedPapersTool(Tool):
    """Find papers related to a given paper."""

    name = "get_related"
    description = (
        "Find papers similar to a given paper. "
        "Useful for exploring related work or finding follow-up research."
    )
    parameters = {
        "arxiv_id": {
            "type": "string",
            "description": "ArXiv ID of the source paper",
            "required": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of related papers to return (default: 5)",
            "required": False,
        },
    }

    def __init__(self, semantic_memory: SemanticMemory | None = None):
        self.semantic_memory = semantic_memory or get_semantic_memory()

    def execute(  # type: ignore[override]
        self,
        arxiv_id: str,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Find related papers."""
        try:
            limit = limit or 5

            results = self.semantic_memory.find_related(
                arxiv_id=arxiv_id,
                limit=limit,
                exclude_self=True,
            )

            if not results:
                return ToolResult(
                    success=False,
                    error=f"Could not find related papers for: {arxiv_id}",
                )

            formatted = []
            for r in results:
                formatted.append({
                    "arxiv_id": r.paper.arxiv_id,
                    "title": r.paper.title,
                    "year": r.paper.year,
                    "similarity": round(r.score, 3),
                })

            logger.info("Related papers found", source=arxiv_id, count=len(results))

            return ToolResult(
                success=True,
                data=formatted,
                metadata={"source_paper": arxiv_id},
            )
        except Exception as e:
            logger.error("Get related failed", arxiv_id=arxiv_id, error=str(e))
            return ToolResult(success=False, error=str(e))


class ComparePapersTool(Tool):
    """Compare multiple papers on specified aspects."""

    name = "compare_papers"
    description = (
        "Generate a comparison between multiple papers. "
        "Analyzes methodology, contributions, and key differences. "
        "Provide 2-5 paper IDs to compare."
    )
    parameters = {
        "paper_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of ArXiv IDs to compare (2-5 papers)",
            "required": True,
        },
        "aspects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Aspects to compare (e.g., ['methodology', 'contributions', 'results'])",
            "required": False,
        },
    }

    def __init__(
        self,
        semantic_memory: SemanticMemory | None = None,
        llm_service: LLMService | None = None,
    ):
        self.semantic_memory = semantic_memory or get_semantic_memory()
        self.llm_service = llm_service or get_llm_service()

    def execute(  # type: ignore[override]
        self,
        paper_ids: list[str],
        aspects: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Compare papers."""
        try:
            if len(paper_ids) < 2:
                return ToolResult(
                    success=False,
                    error="Need at least 2 papers to compare",
                )
            if len(paper_ids) > 5:
                return ToolResult(
                    success=False,
                    error="Can compare at most 5 papers at once",
                )

            # Retrieve papers
            papers = self.semantic_memory.get_papers(paper_ids)

            if len(papers) < 2:
                return ToolResult(
                    success=False,
                    error=f"Could only find {len(papers)} of {len(paper_ids)} papers",
                )

            # Default aspects
            aspects = aspects or ["methodology", "contributions", "key_findings"]

            # Build comparison prompt
            papers_text = "\n\n".join([
                f"Paper {i+1}: {p.title}\n"
                f"ArXiv ID: {p.arxiv_id}\n"
                f"Year: {p.year}\n"
                f"Abstract: {p.abstract}"
                for i, p in enumerate(papers)
            ])

            aspects_text = ", ".join(aspects)

            prompt = f"""Compare the following academic papers on these aspects: {aspects_text}

{papers_text}

Provide a structured comparison that:
1. Identifies key similarities
2. Highlights important differences
3. Notes the progression of ideas if papers are related
4. Summarizes the unique contribution of each paper

Be concise but thorough."""

            # Generate comparison
            messages = [
                {"role": "system", "content": "You are an expert at analyzing and comparing academic papers."},
                {"role": "user", "content": prompt},
            ]

            comparison = self.llm_service.chat_completion(messages, max_tokens=1500)

            logger.info("Comparison generated", paper_count=len(papers))

            return ToolResult(
                success=True,
                data={
                    "comparison": comparison,
                    "papers": [{"arxiv_id": p.arxiv_id, "title": p.title} for p in papers],
                    "aspects": aspects,
                },
            )
        except Exception as e:
            logger.error("Compare failed", error=str(e))
            return ToolResult(success=False, error=str(e))


class SummarizePaperTool(Tool):
    """Generate a summary of a paper."""

    name = "summarize_paper"
    description = (
        "Generate a comprehensive summary of a paper. "
        "Includes key contributions, methodology, and findings."
    )
    parameters = {
        "arxiv_id": {
            "type": "string",
            "description": "ArXiv ID of the paper to summarize",
            "required": True,
        },
        "style": {
            "type": "string",
            "enum": ["brief", "detailed", "technical"],
            "description": "Summary style: brief (1-2 paragraphs), detailed (full summary), technical (for experts)",
            "required": False,
        },
    }

    def __init__(
        self,
        semantic_memory: SemanticMemory | None = None,
        llm_service: LLMService | None = None,
    ):
        self.semantic_memory = semantic_memory or get_semantic_memory()
        self.llm_service = llm_service or get_llm_service()

    def execute(  # type: ignore[override]
        self,
        arxiv_id: str,
        style: str = "detailed",
        **kwargs: Any,
    ) -> ToolResult:
        """Summarize a paper."""
        try:
            paper = self.semantic_memory.get_paper(arxiv_id)

            if not paper:
                return ToolResult(
                    success=False,
                    error=f"Paper not found: {arxiv_id}",
                )

            # Style-specific instructions
            style_instructions = {
                "brief": "Provide a brief 1-2 paragraph summary suitable for a quick overview.",
                "detailed": "Provide a comprehensive summary covering all key aspects of the paper.",
                "technical": "Provide a technical summary focusing on methodology and implementation details.",
            }

            prompt = f"""Summarize the following academic paper.

Title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Year: {paper.year}
Categories: {', '.join(paper.categories)}

Abstract:
{paper.abstract}

{style_instructions.get(style, style_instructions['detailed'])}

Include:
1. Main problem/research question
2. Key methodology or approach
3. Major findings or contributions
4. Significance and potential impact"""

            messages = [
                {"role": "system", "content": "You are an expert at summarizing academic papers clearly and accurately."},
                {"role": "user", "content": prompt},
            ]

            max_tokens = {"brief": 500, "detailed": 1000, "technical": 1200}.get(style, 1000)
            summary = self.llm_service.chat_completion(messages, max_tokens=max_tokens)

            logger.info("Summary generated", arxiv_id=arxiv_id, style=style)

            return ToolResult(
                success=True,
                data={
                    "summary": summary,
                    "paper": {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "year": paper.year,
                    },
                    "style": style,
                },
            )
        except Exception as e:
            logger.error("Summarize failed", arxiv_id=arxiv_id, error=str(e))
            return ToolResult(success=False, error=str(e))


class RecallMemoryTool(Tool):
    """Recall past interactions from episodic memory."""

    name = "recall_memory"
    description = (
        "Search past interactions and searches from memory. "
        "Useful for finding papers searched for previously or "
        "recalling what the user looked at before."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "What to recall (e.g., 'transformers', 'papers from last week')",
            "required": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum memories to return (default: 5)",
            "required": False,
        },
    }

    def __init__(self) -> None:
        # Import here to avoid circular imports
        from src.memory.episodic import EpisodicMemoryStore, get_episodic_store
        self._episodic: EpisodicMemoryStore | None = None
        self._get_store = get_episodic_store

    @property
    def episodic(self) -> Any:
        if self._episodic is None:
            self._episodic = self._get_store()
        return self._episodic

    def execute(  # type: ignore[override]
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Recall memories."""
        import asyncio

        try:
            # Run async method synchronously
            loop = asyncio.new_event_loop()
            try:
                memories = loop.run_until_complete(
                    self.episodic.search_by_query(query, limit=limit)
                )
            finally:
                loop.close()

            if not memories:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={"message": "No matching memories found"},
                )

            formatted = []
            for mem in memories:
                formatted.append({
                    "query": mem.query,
                    "papers": mem.result_paper_ids[:5],
                    "result_count": mem.result_count,
                    "liked_papers": mem.liked_paper_ids,
                    "when": mem.created_at.strftime("%Y-%m-%d %H:%M"),
                })

            logger.info("Memory recalled", query=query[:30], count=len(memories))

            return ToolResult(
                success=True,
                data=formatted,
                metadata={"query": query, "total_found": len(memories)},
            )
        except Exception as e:
            logger.error("Recall failed", error=str(e))
            return ToolResult(success=False, error=str(e))


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("Tool registered", name=tool.name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(**kwargs)


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()

    # Register core tools
    registry.register(SearchPapersTool())
    registry.register(GetPaperTool())
    registry.register(GetRelatedPapersTool())
    registry.register(ComparePapersTool())
    registry.register(SummarizePaperTool())
    registry.register(RecallMemoryTool())

    logger.info("Tool registry created", tools=registry.list_tools())

    return registry


# Singleton registry
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the tool registry singleton."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = create_default_registry()
    return _tool_registry


if __name__ == "__main__":
    # Quick test
    registry = create_default_registry()

    print("Available tools:")
    for name in registry.list_tools():
        tool = registry.get(name)
        if tool:
            print(f"  - {name}: {tool.description[:50]}...")

    print("\nTool schemas:")
    for schema in registry.get_schemas():
        print(f"  - {schema['function']['name']}")
