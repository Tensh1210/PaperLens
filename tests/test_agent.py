"""
Tests for PaperLens agent system.
"""



class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_tool(self, tool_registry):
        """Test registering a tool."""
        from src.agent.tools import Tool, ToolResult

        class TestTool(Tool):
            name = "test_tool"
            description = "A test tool"
            parameters = {"param1": {"type": "string", "required": True}}

            def execute(self, **kwargs):
                return ToolResult(success=True, data="test")

        tool = TestTool()
        tool_registry.register(tool)

        assert "test_tool" in tool_registry.list_tools()
        assert tool_registry.get("test_tool") is not None

    def test_execute_tool(self, tool_registry):
        """Test executing a registered tool."""
        from src.agent.tools import Tool, ToolResult

        class TestTool(Tool):
            name = "echo_tool"
            description = "Echoes input"
            parameters = {"message": {"type": "string", "required": True}}

            def execute(self, message: str = "", **kwargs):
                return ToolResult(success=True, data=message)

        tool_registry.register(TestTool())

        result = tool_registry.execute("echo_tool", message="hello")
        assert result.success is True
        assert result.data == "hello"

    def test_execute_unknown_tool(self, tool_registry):
        """Test executing unknown tool returns error."""
        result = tool_registry.execute("nonexistent_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_get_schemas(self, tool_registry):
        """Test getting tool schemas."""
        from src.agent.tools import Tool, ToolResult

        class TestTool(Tool):
            name = "schema_test"
            description = "Test schema"
            parameters = {
                "query": {"type": "string", "description": "Search query", "required": True}
            }

            def execute(self, **kwargs):
                return ToolResult(success=True)

        tool_registry.register(TestTool())

        schemas = tool_registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "schema_test"


class TestQueryPlanner:
    """Tests for the query planner."""

    def test_detect_search_intent(self):
        """Test detecting search intent."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("Find papers about transformers", use_llm=False)

        assert plan.intent == "search"
        assert len(plan.steps) >= 1

    def test_detect_compare_intent(self):
        """Test detecting comparison intent."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("Compare BERT and GPT", use_llm=False)

        assert plan.intent == "compare"
        assert plan.requires_comparison is True

    def test_detect_summarize_intent(self):
        """Test detecting summarize intent."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("Summarize paper 1706.03762", use_llm=False)

        assert plan.intent == "summarize"
        assert plan.requires_summary is True

    def test_detect_recall_intent(self):
        """Test detecting recall/memory intent."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("What did I search for last week?", use_llm=False)

        assert plan.intent == "recall"

    def test_extract_paper_ids(self):
        """Test extracting arxiv IDs from query."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("Compare 1706.03762 and 1810.04805", use_llm=False)

        # Paper IDs should be extracted
        assert len(plan.steps) >= 1

    def test_detect_year_filter(self):
        """Test detecting year filters."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()

        # Test "from 2023"
        plan1 = planner.plan("Find papers from 2023", use_llm=False)
        assert len(plan1.steps) >= 1

        # Test "recent"
        plan2 = planner.plan("Find recent transformer papers", use_llm=False)
        assert len(plan2.steps) >= 1

    def test_plan_explanation(self):
        """Test plan explanation generation."""
        from src.agent.planner import QueryPlanner

        planner = QueryPlanner()
        plan = planner.plan("Find papers about transformers", use_llm=False)
        explanation = planner.explain_plan(plan)

        assert "Query:" in explanation
        assert "Intent:" in explanation
        assert "Steps" in explanation


class TestToolResult:
    """Tests for ToolResult."""

    def test_successful_result(self):
        """Test successful tool result."""
        from src.agent.tools import ToolResult

        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_failed_result(self):
        """Test failed tool result."""
        from src.agent.tools import ToolResult

        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_observation_success(self):
        """Test observation string for success."""
        from src.agent.tools import ToolResult

        result = ToolResult(success=True, data="Test data")
        observation = result.to_observation()
        assert observation == "Test data"

    def test_to_observation_error(self):
        """Test observation string for error."""
        from src.agent.tools import ToolResult

        result = ToolResult(success=False, error="Error message")
        observation = result.to_observation()
        assert "Error:" in observation


class TestAgentPrompts:
    """Tests for agent prompts."""

    def test_format_tools_description(self):
        """Test formatting tool descriptions."""
        from src.agent.prompts import format_tools_description

        tools = [
            {
                "function": {
                    "name": "search_papers",
                    "description": "Search for papers",
                    "parameters": {
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                                "required": True,
                            }
                        }
                    }
                }
            }
        ]

        description = format_tools_description(tools)
        assert "search_papers" in description
        assert "Search for papers" in description

    def test_format_react_prompt(self):
        """Test formatting ReAct prompt."""
        from src.agent.prompts import format_react_prompt

        tools = [
            {
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"properties": {}}
                }
            }
        ]

        prompt = format_react_prompt(
            query="Find papers about transformers",
            tools=tools,
            context="User is interested in NLP",
        )

        assert "Find papers about transformers" in prompt
        assert "test_tool" in prompt
        assert "THOUGHT:" in prompt
        assert "ACTION:" in prompt

    def test_format_conversation_context(self):
        """Test formatting conversation context."""
        from src.agent.prompts import format_conversation_context

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        papers = ["1706.03762", "1810.04805"]

        context = format_conversation_context(
            history=history,
            papers=papers,
            current_query="Find more papers",
        )

        assert "USER:" in context
        assert "ASSISTANT:" in context
        assert "1706.03762" in context

    def test_format_search_results(self):
        """Test formatting search results."""
        from src.agent.prompts import format_search_results

        results = [
            {
                "title": "Test Paper",
                "arxiv_id": "2301.12345",
                "year": 2023,
                "score": 0.95,
            }
        ]

        formatted = format_search_results(results)
        assert "Test Paper" in formatted
        assert "2301.12345" in formatted

    def test_format_search_results_empty(self):
        """Test formatting empty results."""
        from src.agent.prompts import format_search_results

        formatted = format_search_results([])
        assert "couldn't find" in formatted.lower()


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_agent_response_creation(self):
        """Test creating agent response."""
        from src.agent.agent import AgentResponse

        response = AgentResponse(
            response="Here are the papers...",
            session_id="test-session",
            steps=[],
            papers=["1706.03762"],
            metadata={"iterations": 2},
        )

        assert response.response == "Here are the papers..."
        assert response.session_id == "test-session"
        assert len(response.papers) == 1

    def test_agent_response_to_dict(self):
        """Test converting response to dict."""
        from src.agent.agent import AgentResponse

        response = AgentResponse(
            response="Test response",
            session_id="test",
            steps=[],
            papers=["1706.03762"],
        )

        d = response.to_dict()
        assert d["response"] == "Test response"
        assert d["session_id"] == "test"
        assert d["papers_referenced"] == ["1706.03762"]


class TestAgentParseResponse:
    """Tests for agent response parsing."""

    def test_parse_action_response(self):
        """Test parsing action response."""
        from src.agent.agent import PaperLensAgent

        agent = PaperLensAgent.__new__(PaperLensAgent)

        response = """
THOUGHT: I need to search for papers about transformers.
ACTION: search_papers
ACTION_INPUT: {"query": "transformers", "limit": 5}
"""

        parsed = agent._parse_response(response)
        assert parsed["type"] == "action"
        assert parsed["action"] == "search_papers"
        assert parsed["action_input"]["query"] == "transformers"

    def test_parse_final_answer(self):
        """Test parsing final answer."""
        from src.agent.agent import PaperLensAgent

        agent = PaperLensAgent.__new__(PaperLensAgent)

        response = """
THOUGHT: I have found the papers and can now answer.
FINAL_ANSWER: Here are the top papers about transformers...
"""

        parsed = agent._parse_response(response)
        assert parsed["type"] == "final_answer"
        assert "top papers" in parsed["content"]

    def test_parse_unknown_format(self):
        """Test parsing unknown response format."""
        from src.agent.agent import PaperLensAgent

        agent = PaperLensAgent.__new__(PaperLensAgent)

        response = "This is just some random text without proper format."

        parsed = agent._parse_response(response)
        assert parsed["type"] == "unknown"
