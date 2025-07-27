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


@register_function(config_type=BackrubConfig)
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

        # For other research, provide comprehensive response using the LLM
        logger.info("Backrub conducting research using LLM")

        messages = [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": f"Please research the following topic thoroughly: {query}",
            },
        ]

        response = await llm.ainvoke(messages)
        return response.content

    yield _backrub
