"""
Query Planner for PaperLens.

Decomposes complex user queries into structured plans:
- Identifies user intent
- Breaks down into subtasks
- Determines tool requirements
- Orders operations for efficient execution
"""

import json
import re
from typing import Any

import structlog

from src.agent.prompts import QUERY_DECOMPOSITION_PROMPT
from src.services.llm import LLMService, get_llm_service

logger = structlog.get_logger()


class QueryIntent:
    """Recognized query intents."""

    SEARCH = "search"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    FIND_RELATED = "find_related"
    RECALL = "recall"
    MULTI_STEP = "multi_step"


class PlanStep:
    """A single step in an execution plan."""

    def __init__(
        self,
        task: str,
        tool: str | None = None,
        parameters: dict[str, Any] | None = None,
        depends_on: list[int] | None = None,
    ):
        """
        Initialize a plan step.

        Args:
            task: Description of the task.
            tool: Tool to use (if any).
            parameters: Tool parameters.
            depends_on: Indices of steps this depends on.
        """
        self.task = task
        self.tool = tool
        self.parameters = parameters or {}
        self.depends_on = depends_on or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "tool": self.tool,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
        }


class QueryPlan:
    """A plan for executing a user query."""

    def __init__(
        self,
        original_query: str,
        intent: str,
        steps: list[PlanStep],
        requires_comparison: bool = False,
        requires_summary: bool = False,
    ):
        """
        Initialize a query plan.

        Args:
            original_query: The original user query.
            intent: Detected intent.
            steps: List of execution steps.
            requires_comparison: Whether comparison is needed.
            requires_summary: Whether summary is needed.
        """
        self.original_query = original_query
        self.intent = intent
        self.steps = steps
        self.requires_comparison = requires_comparison
        self.requires_summary = requires_summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "steps": [s.to_dict() for s in self.steps],
            "requires_comparison": self.requires_comparison,
            "requires_summary": self.requires_summary,
        }

    def __repr__(self) -> str:
        return f"QueryPlan(intent={self.intent}, steps={len(self.steps)})"


class QueryPlanner:
    """
    Planner for decomposing user queries.

    Uses a combination of pattern matching and LLM-based planning
    for complex queries.
    """

    def __init__(self, llm_service: LLMService | None = None):
        """
        Initialize the planner.

        Args:
            llm_service: LLM service for complex planning.
        """
        self.llm = llm_service or get_llm_service()

        # Pattern matchers for common intents
        self._intent_patterns = {
            QueryIntent.COMPARE: [
                r"\bcompare\b",
                r"\bdifference[s]?\s+between\b",
                r"\bvs\.?\b",
                r"\bversus\b",
                r"\bcontrast\b",
            ],
            QueryIntent.SUMMARIZE: [
                r"\bsummar(y|ize|ise)\b",
                r"\bexplain\b.*\bpaper\b",
                r"\bwhat\s+is\b.*\babout\b",
                r"\btldr\b",
                r"\bbrief\b.*\boverview\b",
            ],
            QueryIntent.FIND_RELATED: [
                r"\brelated\s+(to|papers?)\b",
                r"\bsimilar\s+(to|papers?)\b",
                r"\blike\s+this\b",
                r"\bmore\s+like\b",
            ],
            QueryIntent.RECALL: [
                r"\blast\s+(time|week|month|search)\b",
                r"\bpreviously\b",
                r"\bbefore\b",
                r"\bhistory\b",
                r"\bremember\b",
            ],
        }

        logger.info("Query planner initialized")

    def plan(self, query: str, use_llm: bool = True) -> QueryPlan:
        """
        Create an execution plan for a query.

        Args:
            query: User query.
            use_llm: Whether to use LLM for complex queries.

        Returns:
            QueryPlan with steps to execute.
        """
        query_lower = query.lower()

        # Detect intent
        intent = self._detect_intent(query_lower)

        # Check for specific patterns
        paper_ids = self._extract_paper_ids(query)
        has_year_filter = self._detect_year_filter(query_lower)

        logger.debug(
            "Analyzing query",
            query=query[:50],
            intent=intent,
            paper_ids=paper_ids,
            has_year=has_year_filter,
        )

        # Build plan based on intent
        if intent == QueryIntent.COMPARE:
            return self._plan_comparison(query, paper_ids)
        elif intent == QueryIntent.SUMMARIZE:
            return self._plan_summary(query, paper_ids)
        elif intent == QueryIntent.FIND_RELATED:
            return self._plan_related(query, paper_ids)
        elif intent == QueryIntent.RECALL:
            return self._plan_recall(query)
        elif self._is_complex_query(query) and use_llm:
            return self._plan_with_llm(query)
        else:
            return self._plan_search(query, has_year_filter)

    def _detect_intent(self, query: str) -> str:
        """Detect primary intent from query."""
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return QueryIntent.SEARCH

    def _extract_paper_ids(self, query: str) -> list[str]:
        """Extract ArXiv IDs from query."""
        # Pattern for ArXiv IDs: YYMM.NNNNN or category/YYMMNNN
        pattern = r'\b(\d{4}\.\d{4,5}|[a-z-]+/\d{7})\b'
        return re.findall(pattern, query)

    def _detect_year_filter(self, query: str) -> dict[str, int] | None:
        """Detect year filters in query."""
        filters = {}

        # "from 2023", "since 2023", "after 2023"
        match = re.search(r'\b(from|since|after)\s+(\d{4})\b', query)
        if match:
            filters["year_from"] = int(match.group(2))

        # "before 2024", "until 2023"
        match = re.search(r'\b(before|until|to)\s+(\d{4})\b', query)
        if match:
            filters["year_to"] = int(match.group(2))

        # "in 2023", "2023 papers"
        match = re.search(r'\b(in\s+)?(\d{4})(\s+papers?)?\b', query)
        if match and not filters:
            year = int(match.group(2))
            if 2000 <= year <= 2030:
                filters["year_from"] = year
                filters["year_to"] = year

        # "recent", "latest"
        if re.search(r'\b(recent|latest|new)\b', query):
            from datetime import datetime
            filters["year_from"] = datetime.now().year - 1

        return filters if filters else None

    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex enough to need LLM planning."""
        # Multiple clauses
        if len(query.split(" and ")) > 2:
            return True
        if len(query.split(",")) > 2:
            return True

        # Long query
        if len(query.split()) > 15:
            return True

        # Multiple question marks
        if query.count("?") > 1:
            return True

        return False

    def _plan_search(
        self,
        query: str,
        year_filter: dict[str, int] | None,
    ) -> QueryPlan:
        """Create a simple search plan."""
        params = {"query": query, "limit": 10}
        if year_filter:
            params.update(year_filter)

        steps = [
            PlanStep(
                task=f"Search for papers: {query}",
                tool="search_papers",
                parameters=params,
            )
        ]

        return QueryPlan(
            original_query=query,
            intent=QueryIntent.SEARCH,
            steps=steps,
        )

    def _plan_comparison(
        self,
        query: str,
        paper_ids: list[str],
    ) -> QueryPlan:
        """Create a comparison plan."""
        steps = []

        if paper_ids and len(paper_ids) >= 2:
            # Direct comparison of specified papers
            steps.append(
                PlanStep(
                    task=f"Compare papers: {', '.join(paper_ids)}",
                    tool="compare_papers",
                    parameters={"paper_ids": paper_ids},
                )
            )
        else:
            # Need to search first, then compare
            # Extract search query (remove comparison words)
            search_query = re.sub(
                r'\b(compare|versus|vs\.?|difference|contrast)\b',
                '',
                query,
                flags=re.IGNORECASE,
            ).strip()

            steps.append(
                PlanStep(
                    task=f"Search for papers to compare: {search_query}",
                    tool="search_papers",
                    parameters={"query": search_query, "limit": 5},
                )
            )
            steps.append(
                PlanStep(
                    task="Compare top papers from search results",
                    tool="compare_papers",
                    parameters={},  # Will use results from step 0
                    depends_on=[0],
                )
            )

        return QueryPlan(
            original_query=query,
            intent=QueryIntent.COMPARE,
            steps=steps,
            requires_comparison=True,
        )

    def _plan_summary(
        self,
        query: str,
        paper_ids: list[str],
    ) -> QueryPlan:
        """Create a summary plan."""
        steps = []

        if paper_ids:
            # Summarize specific paper
            for pid in paper_ids:
                steps.append(
                    PlanStep(
                        task=f"Summarize paper {pid}",
                        tool="summarize_paper",
                        parameters={"arxiv_id": pid},
                    )
                )
        else:
            # Need to search first
            search_query = re.sub(
                r'\b(summar(y|ize|ise)|explain|overview)\b',
                '',
                query,
                flags=re.IGNORECASE,
            ).strip()

            steps.append(
                PlanStep(
                    task=f"Search for paper: {search_query}",
                    tool="search_papers",
                    parameters={"query": search_query, "limit": 1},
                )
            )
            steps.append(
                PlanStep(
                    task="Summarize the found paper",
                    tool="summarize_paper",
                    parameters={},
                    depends_on=[0],
                )
            )

        return QueryPlan(
            original_query=query,
            intent=QueryIntent.SUMMARIZE,
            steps=steps,
            requires_summary=True,
        )

    def _plan_related(
        self,
        query: str,
        paper_ids: list[str],
    ) -> QueryPlan:
        """Create a find-related plan."""
        steps = []

        if paper_ids:
            # Find related to specific paper
            steps.append(
                PlanStep(
                    task=f"Find papers related to {paper_ids[0]}",
                    tool="get_related",
                    parameters={"arxiv_id": paper_ids[0], "limit": 5},
                )
            )
        else:
            # Need to identify the reference paper first
            search_query = re.sub(
                r'\b(related|similar|like)\s+(to|papers?)?\b',
                '',
                query,
                flags=re.IGNORECASE,
            ).strip()

            steps.append(
                PlanStep(
                    task=f"Search for reference paper: {search_query}",
                    tool="search_papers",
                    parameters={"query": search_query, "limit": 1},
                )
            )
            steps.append(
                PlanStep(
                    task="Find related papers",
                    tool="get_related",
                    parameters={},
                    depends_on=[0],
                )
            )

        return QueryPlan(
            original_query=query,
            intent=QueryIntent.FIND_RELATED,
            steps=steps,
        )

    def _plan_recall(self, query: str) -> QueryPlan:
        """Create a recall plan for past interactions."""
        steps = [
            PlanStep(
                task="Recall relevant past interactions",
                tool="recall_memory",
                parameters={"query": query},
            )
        ]

        return QueryPlan(
            original_query=query,
            intent=QueryIntent.RECALL,
            steps=steps,
        )

    def _plan_with_llm(self, query: str) -> QueryPlan:
        """Use LLM to create plan for complex queries."""
        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)

        messages = [
            {
                "role": "system",
                "content": "You are a query planner. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm.chat_completion(
                messages,
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=500,
            )

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON in LLM response, falling back to search")
                return self._plan_search(query, None)

            plan_data = json.loads(json_match.group())

            # Convert to QueryPlan
            steps = []
            for subtask in plan_data.get("subtasks", []):
                steps.append(
                    PlanStep(
                        task=subtask.get("task", ""),
                        tool=subtask.get("tool"),
                        depends_on=subtask.get("depends_on", []),
                    )
                )

            if not steps:
                # Fallback to search
                return self._plan_search(query, None)

            return QueryPlan(
                original_query=query,
                intent=plan_data.get("intent", QueryIntent.MULTI_STEP),
                steps=steps,
                requires_comparison=plan_data.get("requires_comparison", False),
                requires_summary=plan_data.get("requires_summary", False),
            )

        except Exception as e:
            logger.error("LLM planning failed", error=str(e))
            return self._plan_search(query, None)

    def explain_plan(self, plan: QueryPlan) -> str:
        """Generate human-readable explanation of plan."""
        lines = [
            f"Query: {plan.original_query}",
            f"Intent: {plan.intent}",
            f"Steps ({len(plan.steps)}):",
        ]

        for i, step in enumerate(plan.steps):
            deps = f" (after step {step.depends_on})" if step.depends_on else ""
            tool = f" using {step.tool}" if step.tool else ""
            lines.append(f"  {i+1}. {step.task}{tool}{deps}")

        if plan.requires_comparison:
            lines.append("  → Will generate comparison")
        if plan.requires_summary:
            lines.append("  → Will generate summary")

        return "\n".join(lines)


# Singleton instance
_planner: QueryPlanner | None = None


def get_planner() -> QueryPlanner:
    """Get or create the query planner singleton."""
    global _planner
    if _planner is None:
        _planner = QueryPlanner()
    return _planner


if __name__ == "__main__":
    # Quick test
    planner = QueryPlanner()

    test_queries = [
        "Find papers about attention mechanisms",
        "Compare BERT and GPT",
        "Summarize paper 1706.03762",
        "Find papers related to transformers from 2023",
        "What papers did I search for last week about NLP?",
        "Compare the methodology and results of recent vision transformer papers",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        plan = planner.plan(query)
        print(planner.explain_plan(plan))
