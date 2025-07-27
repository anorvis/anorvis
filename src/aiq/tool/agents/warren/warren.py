"""
Warren Finance Agent

A specialized finance agent that can be called by other agents for financial tasks.
"""

import logging
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from ..base_config import BaseAgentConfig

logger = logging.getLogger(__name__)


class WarrenConfig(BaseAgentConfig, name="warren"):
    """Configuration for Warren Finance Agent."""

    system_prompt: str = Field(
        "You are Warren, a specialized finance agent. You excel at financial analysis, "
        "budgeting, investments, and financial planning. Always provide professional, "
        "actionable financial advice.",
        description="System prompt for the Warren finance agent.",
    )


@register_function(config_type=WarrenConfig)
async def warren_agent(config: WarrenConfig, builder: Builder):
    """
    Warren Finance Agent function.

    This agent handles financial tasks and can call other agents for coordination.
    """
    pass

    async def _warren(query: str) -> str:
        """
        Handle financial queries and analysis.

        Args:
            query: The financial query to analyze

        Returns:
            Financial analysis and recommendations
        """
        logger.info(f"Warren is analyzing: {query}")

        # Get LLM when needed
        llm = await builder.get_llm(
            config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        query_lower = query.lower()

        # Research-heavy financial queries - suggest coordination
        research_keywords = [
            "market",
            "trends",
            "industry",
            "research",
            "analysis",
            "latest",
            "study",
            "investigation",
        ]

        if any(keyword in query_lower for keyword in research_keywords):
            logger.info(
                "Warren detecting research-heavy financial query - suggesting coordination"
            )

            # Provide financial analysis and suggest research coordination
            financial_response = f"""
**Financial Analysis for: {query}**

I've provided financial analysis for this topic. Since this involves significant research 
and market analysis, I recommend coordinating with our research specialist (Backrub) 
for comprehensive market research and trends.

**Financial Recommendations:**
"""

            # Add financial recommendations using LLM
            messages = [
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Provide financial analysis and recommendations for: {query}. "
                        "Focus on financial implications, risks, and opportunities."
                    ),
                },
            ]
            financial_recommendations = await llm.ainvoke(messages)

            return (
                financial_response
                + financial_recommendations.content
                + "\n\n**Note:** For comprehensive market research and trends analysis, "
                "please ask Anorvis to coordinate with Backrub."
            )

        # For other financial queries, provide general financial advice using the LLM
        logger.info("Warren providing financial advice using LLM")

        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": f"Provide financial advice for: {query}"},
        ]

        response = await llm.ainvoke(messages)
        return response.content

    yield _warren
