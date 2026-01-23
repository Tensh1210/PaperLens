"""
Agent Prompts for PaperLens.

Contains prompt templates for the ReAct agent including:
- System prompt defining agent persona
- ReAct reasoning template
- Tool selection instructions
- Task-specific prompts
"""

SYSTEM_PROMPT = """You are PaperLens, an intelligent research assistant specialized in finding, analyzing, and comparing machine learning papers.

Your capabilities:
1. **Search**: Find papers using semantic search, with filters for year and category
2. **Retrieve**: Get full details of specific papers by ArXiv ID
3. **Compare**: Analyze similarities and differences between multiple papers
4. **Summarize**: Generate clear summaries of individual papers
5. **Relate**: Find papers related to a given paper

You have access to a database of ML papers from ArXiv. When users ask about papers, you should:
- Use your tools to search and retrieve relevant information
- Provide accurate, well-structured responses
- Cite specific papers with their ArXiv IDs
- Be honest when you cannot find relevant papers

Always think step by step and use the appropriate tools to gather information before responding."""


REACT_PROMPT = """You are a reasoning agent that follows the ReAct (Reasoning + Acting) pattern.

For each user query, you will:
1. **THINK**: Analyze what the user wants and plan your approach
2. **ACT**: Use a tool to gather information
3. **OBSERVE**: Process the tool's result
4. **REPEAT** or **RESPOND**: Either continue with more actions or provide a final answer

## Output Format

Your response must follow this exact format:

THOUGHT: [Your reasoning about what to do next]
ACTION: [tool_name]
ACTION_INPUT: [JSON parameters for the tool]

OR, when you have enough information to answer:

THOUGHT: [Your final reasoning]
FINAL_ANSWER: [Your response to the user]

## Rules

1. Always start with a THOUGHT
2. Only use one ACTION per turn
3. ACTION_INPUT must be valid JSON
4. Use FINAL_ANSWER only when you have gathered sufficient information
5. Be concise but thorough in your reasoning
6. If a tool returns an error, try an alternative approach

## Available Tools

{tools_description}

## Example

User: "Find papers about attention mechanisms in transformers"

THOUGHT: The user wants papers about attention mechanisms in transformers. I should search for papers on this topic.
ACTION: search_papers
ACTION_INPUT: {{"query": "attention mechanisms transformers", "limit": 5}}

[After receiving results]

THOUGHT: I found 5 relevant papers. Let me provide a summary to the user.
FINAL_ANSWER: I found several papers about attention mechanisms in transformers:

1. **Attention Is All You Need** (1706.03762) - The original transformer paper introducing self-attention
2. ...

## Current Context

{context}

## User Query

{query}

Begin your response:"""


TOOL_SELECTION_PROMPT = """Based on the user's request, select the most appropriate tool:

**User Request**: {query}

**Available Tools**:
{tools_list}

**Guidelines**:
- Use `search_papers` for finding papers on a topic
- Use `get_paper` when you have a specific ArXiv ID
- Use `compare_papers` when comparing 2+ papers
- Use `summarize_paper` for detailed paper summaries
- Use `get_related` to find similar papers

Select the tool that best addresses the user's immediate need."""


COMPARE_PROMPT = """Compare the following academic papers on these aspects: {aspects}

{papers_text}

Provide a structured comparison that:
1. Identifies key similarities
2. Highlights important differences
3. Notes the progression of ideas if papers are related
4. Summarizes the unique contribution of each paper

Be concise but thorough. Use bullet points for clarity."""


SUMMARY_PROMPT = """Summarize the following academic paper:

**Title**: {title}
**Authors**: {authors}
**Year**: {year}
**Categories**: {categories}

**Abstract**:
{abstract}

{style_instructions}

Structure your summary to include:
1. **Problem/Motivation**: What problem does this paper address?
2. **Approach/Methodology**: How do the authors tackle the problem?
3. **Key Contributions**: What are the main contributions?
4. **Results/Findings**: What did they discover or achieve?
5. **Significance**: Why is this work important?"""


QUERY_DECOMPOSITION_PROMPT = """Analyze this user query and break it down into subtasks:

**Query**: {query}

Identify:
1. What information the user needs
2. What tools would be helpful
3. The logical order of operations

Output a JSON plan:
{{
    "intent": "brief description of user intent",
    "subtasks": [
        {{"task": "description", "tool": "tool_name", "depends_on": []}},
        ...
    ],
    "requires_comparison": true/false,
    "requires_summary": true/false
}}"""


CONVERSATION_CONTEXT_PROMPT = """## Conversation History

{history}

## Retrieved Papers This Session

{papers}

## Current State

- Messages: {message_count}
- Papers viewed: {paper_count}
- Current query: {current_query}"""


MEMORY_RECALL_PROMPT = """You have access to memories from past interactions:

**Recent Queries**:
{recent_queries}

**User Preferences**:
{preferences}

Use this context to provide more personalized and relevant responses."""


def format_tools_description(tools: list[dict]) -> str:
    """Format tool schemas for the prompt."""
    lines = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})

        param_strs = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            required = param_info.get("required", False)
            req_str = " (required)" if required else ""
            param_strs.append(f"    - {param_name} ({param_type}){req_str}: {param_desc}")

        params_text = "\n".join(param_strs) if param_strs else "    (no parameters)"
        lines.append(f"### {name}\n{desc}\n**Parameters**:\n{params_text}\n")

    return "\n".join(lines)


def format_react_prompt(
    query: str,
    tools: list[dict],
    context: str = "",
) -> str:
    """Format the ReAct prompt with query and tools."""
    tools_desc = format_tools_description(tools)
    return REACT_PROMPT.format(
        tools_description=tools_desc,
        context=context or "No prior context.",
        query=query,
    )


def format_conversation_context(
    history: list[dict],
    papers: list[str],
    current_query: str | None = None,
) -> str:
    """Format conversation context for the agent."""
    # Format history
    history_lines = []
    for msg in history[-10:]:  # Last 10 messages
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:200]  # Truncate long messages
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "No previous messages."

    # Format papers
    papers_text = ", ".join(papers[-10:]) if papers else "None retrieved yet."

    return CONVERSATION_CONTEXT_PROMPT.format(
        history=history_text,
        papers=papers_text,
        message_count=len(history),
        paper_count=len(papers),
        current_query=current_query or "None",
    )


def format_compare_prompt(
    papers: list[dict],
    aspects: list[str] | None = None,
) -> str:
    """Format the comparison prompt."""
    aspects = aspects or ["methodology", "contributions", "results"]

    papers_text = "\n\n".join([
        f"**Paper {i+1}: {p['title']}**\n"
        f"ArXiv ID: {p['arxiv_id']}\n"
        f"Year: {p.get('year', 'Unknown')}\n"
        f"Abstract: {p['abstract']}"
        for i, p in enumerate(papers)
    ])

    return COMPARE_PROMPT.format(
        aspects=", ".join(aspects),
        papers_text=papers_text,
    )


def format_summary_prompt(
    paper: dict,
    style: str = "detailed",
) -> str:
    """Format the summary prompt."""
    style_instructions = {
        "brief": "Provide a brief 1-2 paragraph summary suitable for a quick overview.",
        "detailed": "Provide a comprehensive summary covering all key aspects.",
        "technical": "Provide a technical summary focusing on methodology and implementation.",
    }

    return SUMMARY_PROMPT.format(
        title=paper.get("title", "Unknown"),
        authors=", ".join(paper.get("authors", [])[:5]),
        year=paper.get("year", "Unknown"),
        categories=", ".join(paper.get("categories", [])),
        abstract=paper.get("abstract", "No abstract available."),
        style_instructions=style_instructions.get(style, style_instructions["detailed"]),
    )


# Response templates for consistent formatting
RESPONSE_TEMPLATES = {
    "search_results": """I found {count} papers related to your query:

{papers_list}

Would you like me to summarize any of these papers or compare them?""",

    "paper_details": """**{title}**

- **ArXiv ID**: {arxiv_id}
- **Year**: {year}
- **Authors**: {authors}
- **Categories**: {categories}

**Abstract**:
{abstract}

[PDF]({pdf_url}) | [ArXiv]({arxiv_url})""",

    "no_results": """I couldn't find any papers matching your query "{query}".

Try:
- Using different keywords
- Broadening your search terms
- Removing year filters if you specified any""",

    "error": """I encountered an issue: {error}

Please try rephrasing your request or let me know if you need help with something else.""",
}


def format_search_results(results: list[dict]) -> str:
    """Format search results for display."""
    if not results:
        return RESPONSE_TEMPLATES["no_results"].format(query="your search")

    papers_list = "\n".join([
        f"{i+1}. **{r['title']}** ({r['arxiv_id']}, {r.get('year', 'N/A')}) - Score: {r['score']:.2f}"
        for i, r in enumerate(results)
    ])

    return RESPONSE_TEMPLATES["search_results"].format(
        count=len(results),
        papers_list=papers_list,
    )


def format_paper_details(paper: dict) -> str:
    """Format paper details for display."""
    return RESPONSE_TEMPLATES["paper_details"].format(
        title=paper.get("title", "Unknown"),
        arxiv_id=paper.get("arxiv_id", "Unknown"),
        year=paper.get("year", "Unknown"),
        authors=", ".join(paper.get("authors", [])[:5]) or "Unknown",
        categories=", ".join(paper.get("categories", [])) or "Unknown",
        abstract=paper.get("abstract", "No abstract available."),
        pdf_url=paper.get("pdf_url", "#"),
        arxiv_url=paper.get("arxiv_url", "#"),
    )
