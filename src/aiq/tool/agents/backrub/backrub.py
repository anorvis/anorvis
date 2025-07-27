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
            web_search = builder.get_tool("internet_search", LLMFrameworkEnum.LANGCHAIN)
            wiki_search = builder.get_tool(
                "wikipedia_search", LLMFrameworkEnum.LANGCHAIN
            )
        except Exception as e:
            logger.warning(f"Could not get search tools: {e}")
            web_search = None
            wiki_search = None

        # Use LLM to analyze if financial expertise would add value
        analysis_prompt = f"""
You are analyzing a research query to determine if financial expertise would add value to the analysis.

Query: {query}

Think through this reasoning framework:

1. **Financial Context**: Does this query involve financial markets, investments, companies, 
   economic indicators, or financial decision-making?

2. **Value Addition**: Would having financial analysis (risk assessment, investment advice, 
   market analysis) provide significant additional value beyond the research?

3. **User Intent**: Is the user likely looking for actionable financial insights, 
   investment recommendations, or financial planning advice?

Respond with ONLY:
- 'financial' if financial expertise would add significant value
- 'general' if this is general research without strong financial context

Reasoning: Focus on user value, not just keywords.
"""

        # Get LLM analysis
        analysis_messages = [
            {"role": "system", "content": "You are a research analysis expert."},
            {"role": "user", "content": analysis_prompt},
        ]
        financial_analysis_needed = await llm.ainvoke(analysis_messages)
        financial_analysis_needed = financial_analysis_needed.content.strip().lower()

        research_results = []

        # Try web search first for current information
        if web_search:
            try:
                logger.info("Performing web search for current information")
                web_results = await web_search.ainvoke(query)
                if web_results:
                    research_results.append(
                        f"**Current Web Search Results:**\n{web_results}"
                    )
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        # Add Wikipedia search for background information
        if wiki_search:
            try:
                logger.info("Performing Wikipedia search for background information")
                wiki_results = await wiki_search.ainvoke(query)
                if wiki_results:
                    research_results.append(
                        f"**Wikipedia Background:**\n{wiki_results}"
                    )
            except Exception as e:
                logger.warning(f"Wikipedia search failed: {e}")

        # Combine results with friendly formatting
        if research_results:
            combined_research = "\n\n---\n\n".join(research_results)

            # Simple, direct response
            final_response = f"Hey! I just looked into '{query}' for you. Here's what I found:\n\n{combined_research}"
        else:
            final_response = f"Hey! I tried to look into '{query}' for you, but I couldn't find much current info. You might want to try a different search or ask me something else!"

        # Check if financial analysis would add value
        if "financial" in financial_analysis_needed:
            logger.info(
                "Backrub detecting financial context - suggesting coordination with Warren"
            )
            final_response += "\n\nThis seems like something where some financial advice could really help you out. Want me to connect you with my friend who knows that stuff?"
        else:
            final_response += (
                "\n\nHope that helps! Let me know if you need anything else."
            )

        return final_response

    yield _backrub
