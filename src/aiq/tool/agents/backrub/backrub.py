"""
Backrub Research Agent

A specialized research agent that can be called by other agents for research tasks.
"""

import logging
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from ..base_config import BaseAgentConfig

logger = logging.getLogger(__name__)


class BackrubConfig(BaseAgentConfig, name="backrub"):
    """Configuration for Backrub Research Agent."""

    system_prompt: str = Field(
        "You are Backrub, a specialized research agent. You excel at finding information, "
        "conducting research, and gathering data. Always provide thorough, well-researched responses.",
        description="System prompt for the Backrub research agent.",
    )


@register_function(
    config_type=BackrubConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def backrub_agent(config: BackrubConfig, builder: Builder):
    """
    Backrub Research Agent function.

    This agent handles research tasks and can be called by other agents.
    """
    pass

    async def _backrub(query: str) -> str:
        """
        Conduct research on the given query.

        Args:
            query: The research query to investigate

        Returns:
            Comprehensive research results
        """
        logger.info(f"Backrub is researching: {query}")

        # Get LLM when needed
        llm = await builder.get_llm(
            config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        # Get web search tools
        try:
            web_search = builder.get_tool("webpage_query")
            wiki_search = builder.get_tool("wikipedia_search")
        except Exception as e:
            logger.warning(f"Could not get search tools: {e}")
            web_search = None
            wiki_search = None

        # Determine if this research requires financial expertise
        query_lower = query.lower()
        financial_keywords = [
            "financial",
            "market",
            "investment",
            "stock",
            "economy",
            "trading",
            "budget",
            "money",
            "finance",
        ]

        if any(keyword in query_lower for keyword in financial_keywords):
            logger.info(
                "Backrub detecting financial research - will coordinate with Anorvis"
            )

            # Instead of calling Anorvis directly, provide comprehensive research
            # and suggest coordination for financial analysis
            research_response = f"""
**Research Results for: {query}**

I've conducted comprehensive research on this topic. Since this involves financial analysis, 
I recommend coordinating with our financial specialist (Warren) for detailed financial insights.

**Research Summary:**
"""

            # Add research context using LLM
            messages = [
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Provide comprehensive research findings for: {query}. "
                        "Focus on factual information, trends, and data."
                    ),
                },
            ]
            research_content = await llm.ainvoke(messages)

            return (
                research_response
                + research_content.content
                + "\n\n**Note:** For detailed financial analysis and recommendations, please ask Anorvis to coordinate with Warren."
            )

        # For non-financial research, use web search tools
        research_results = []

        # Try web search first
        if web_search:
            try:
                web_result = await web_search.ainvoke(query)
                research_results.append(f"**Web Search Results:**\n{web_result}")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        # Try Wikipedia search
        if wiki_search:
            try:
                wiki_result = await wiki_search.ainvoke(query)
                research_results.append(f"**Wikipedia Results:**\n{wiki_result}")
            except Exception as e:
                logger.warning(f"Wikipedia search failed: {e}")

        # If no search tools available, use LLM for research
        if not research_results:
            messages = [
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Provide comprehensive research findings for: {query}. "
                        "Focus on factual information, trends, and data."
                    ),
                },
            ]
            research_content = await llm.ainvoke(messages)
            research_results.append(
                f"**Research Analysis:**\n{research_content.content}"
            )

        # Combine all research results
        combined_response = f"""
**Comprehensive Research Results for: {query}**

{chr(10).join(research_results)}

**Research Summary:**
This analysis combines multiple sources to provide you with comprehensive information on your query.
"""

        return combined_response

    yield _backrub
