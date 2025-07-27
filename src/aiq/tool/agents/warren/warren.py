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


@register_function(
    config_type=WarrenConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
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

        # Check if this is a final analysis call (contains research results)
        if "research that has been done" in query or "Based on this research" in query:
            logger.info("Warren providing final analysis based on provided research")

            advice_prompt = f"""
You are a confident financial advisor. {query}

Provide specific, actionable financial advice. Be confident and direct. Give concrete recommendations and explain your reasoning.

If this is about stocks or investments, provide specific stock recommendations with reasoning.
If this is about financial planning, give specific steps and strategies.
If this is about budgeting, provide concrete budgeting advice.

Be helpful, confident, and specific!
"""

            messages = [
                {"role": "system", "content": "You are a confident financial advisor."},
                {"role": "user", "content": advice_prompt},
            ]

            response = await llm.ainvoke(messages)
            return response.content

        # Use LLM to analyze if research would add value
        analysis_prompt = f"""
You are analyzing a financial query to determine if additional research would add value to the analysis.

Query: {query}

Think through this reasoning framework:

1. **Research Context**: Does this query require current market data, recent events, 
   industry trends, or up-to-date information that might not be in your knowledge base?

2. **Value Addition**: Would having current research (market trends, recent news, 
   industry analysis) provide significant additional value to the financial analysis?

3. **User Intent**: Is the user likely looking for actionable insights based on 
   current market conditions, recent developments, or timely information?

4. **Information Currency**: Does this query benefit from the most current 
   information rather than general financial principles?

Respond with ONLY:
- 'research' if current research would add significant value
- 'general' if this can be answered with general financial knowledge

Reasoning: Focus on whether current, researched information would improve the analysis.
"""

        # Get LLM analysis
        analysis_messages = [
            {"role": "system", "content": "You are a financial analysis expert."},
            {"role": "user", "content": analysis_prompt},
        ]
        research_needed = await llm.ainvoke(analysis_messages)
        research_needed = research_needed.content.strip().lower()

        if "research" in research_needed:
            logger.info(
                "Warren detecting need for current research - suggesting coordination with Backrub"
            )

            # Provide friendly financial advice and suggest research coordination
            financial_response = f"""
Hey! I'd love to help you with some financial advice, but I think we'd both benefit from getting some current market info first.

What I can help you with:
- Understanding investment strategies
- Risk assessment and portfolio planning
- Financial analysis and recommendations

My suggestion: Let me connect you with my research buddy who can get us the latest market data and trends. Then I can give you some solid financial advice based on what's actually happening right now.

Sound good? Just ask Anorvis to coordinate us and we'll get you sorted!
"""
            return financial_response

        # For other financial queries, provide friendly financial advice using the LLM
        logger.info("Warren providing financial advice using LLM")

        advice_prompt = f"""
You are a confident financial advisor. The user is asking: {query}

Provide specific, actionable financial advice. Be confident and direct. Give concrete recommendations and explain your reasoning.

If this is about stocks or investments, provide specific stock recommendations with reasoning.
If this is about financial planning, give specific steps and strategies.
If this is about budgeting, provide concrete budgeting advice.

Be helpful, confident, and specific!
"""

        messages = [
            {"role": "system", "content": "You are a confident financial advisor."},
            {"role": "user", "content": advice_prompt},
        ]

        response = await llm.ainvoke(messages)
        return response.content

    yield _warren
