"""
Anorvis Orchestrator Agent

The main orchestrator agent that routes requests to specialized agents using StateGraph orchestration.
"""

import logging
from typing import TypedDict
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from ..base_config import BaseAgentConfig

logger = logging.getLogger(__name__)


class AnorvisState(TypedDict):
    """State for the Anorvis orchestration workflow."""

    messages: list[BaseMessage]
    current_agent: str
    coordination_needed: bool
    suggested_agent: str
    research_results: str
    financial_analysis: str
    final_response: str
    original_query: str


class AnorvisConfig(BaseAgentConfig, name="anorvis"):
    """Configuration for Anorvis Orchestrator Agent."""

    system_prompt: str = Field(
        "You are Anorvis, an intelligent AI assistant that coordinates specialized agents.\n"
        "You analyze user requests and route them to the appropriate specialist agent.\n"
        "You can also coordinate between agents when they suggest it.\n"
        "You only handle general queries and time/date requests directly when no specialist agent is available.\n"
        "IMPORTANT: If you are handling a query directly (not routing to specialists), be honest about your limitations.\n"
        "Do not claim to have information from other agents unless you actually called them.\n"
        "For research, news, or information gathering requests, you should route to Backrub instead of handling them directly.",
        description="System prompt for the Anorvis orchestrator agent.",
    )
    backrub: FunctionRef = Field(description="Reference to the Backrub research agent.")
    warren: FunctionRef = Field(description="Reference to the Warren finance agent.")


@register_function(config_type=AnorvisConfig)
async def anorvis_agent(config: AnorvisConfig, builder: Builder):
    """
    Anorvis Orchestrator Agent function using StateGraph for advanced coordination.
    """
    pass

    async def _anorvis(query: str) -> str:
        """
        Orchestrate user requests using StateGraph for advanced agent coordination.
        """
        logger.info(f"Anorvis received query: {query}")

        # Initialize state
        initial_state = AnorvisState(
            messages=[HumanMessage(content=query)],
            current_agent="",
            coordination_needed=False,
            suggested_agent="",
            research_results="",
            financial_analysis="",
            final_response="",
            original_query=query,
        )

        # Build the orchestration graph
        workflow = StateGraph(AnorvisState)

        # Add nodes
        workflow.add_node("classify", classify_query)
        workflow.add_node("route_to_backrub", route_to_backrub)
        workflow.add_node("route_to_warren", route_to_warren)
        workflow.add_node("coordinate_agents", coordinate_agents)
        workflow.add_node("handle_general", handle_general)
        workflow.add_node("finalize_response", finalize_response)

        # Add conditional edges from classify
        workflow.add_conditional_edges(
            "classify",
            route_decision,
            {
                "backrub": "route_to_backrub",
                "warren": "route_to_warren",
                "coordinate": "coordinate_agents",
                "general": "handle_general",
            },
        )

        # Add edges from routing nodes to finalize
        workflow.add_edge("route_to_backrub", "finalize_response")
        workflow.add_edge("route_to_warren", "finalize_response")
        workflow.add_edge("coordinate_agents", "finalize_response")
        workflow.add_edge("handle_general", "finalize_response")
        workflow.add_edge("finalize_response", END)

        # Set entry point
        workflow.set_entry_point("classify")

        # Compile and run
        app = workflow.compile()

        # Execute the workflow
        final_state = await app.ainvoke(initial_state)

        return final_state["final_response"]

    # Node functions
    async def classify_query(state: AnorvisState) -> AnorvisState:
        """Classify the query using LLM-based intent classification."""
        query = state["messages"][-1].content

        # Get LLM for classification
        llm = await builder.get_llm(
            config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        # Create intelligent routing prompt
        router_prompt = """
You are an intelligent query classifier. Your job is to analyze user requests and determine which specialist agent(s) would be best suited to handle them.

Think through the following reasoning framework:

1. **Research Needs**: Does this query require gathering information, facts, data, or current events?
   - Examples: "what is", "how does", "find", "research", "latest", "current", "news", "data"

2. **Financial Analysis**: Does this query involve money, investments, markets, budgeting, or financial planning?
   - Examples: "invest", "buy", "market", "budget", "financial", "cost", "profit", "portfolio"

3. **Coordination Required**: Does this query benefit from BOTH research AND financial analysis?
   - **CRITICAL**: Investment-related queries (stocks, investments, buying, etc.) almost always need coordination
   - Examples: "promising stocks to buy", "investment opportunities", "market analysis", "what should I invest in"
   - Think: Does the user want actionable advice that requires current data AND financial expertise?

4. **General Queries**: Simple questions, greetings, or requests that don't fit the above categories.

**Your Task**: 
Analyze the user query and respond with ONLY one category:
- 'backrub' (research needed)
- 'warren' (financial analysis needed) 
- 'coordinate' (both research and financial analysis would be valuable)
- 'general' (simple query, no specialist needed)

**IMPORTANT**: When in doubt about investment queries, choose 'coordinate'. 
Investment advice without current research is often outdated or incomplete.

User query: {input}
Classification:"""

        messages = [
            SystemMessage(content=router_prompt),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        classification = response.content.strip().lower()

        # Clean up the classification - remove any extra text
        if "classification:" in classification:
            classification = classification.split("classification:")[-1].strip()
        if ":" in classification:
            classification = classification.split(":")[-1].strip()
        # Remove quotes if present
        classification = classification.strip("'\" ")

        logger.info(f"LLM classified query as: '{classification}'")

        # Set the current agent based on classification
        if classification == "coordinate":
            state["current_agent"] = "coordinator"
        elif classification == "backrub":
            state["current_agent"] = "backrub"
        elif classification == "warren":
            state["current_agent"] = "warren"
        else:
            state["current_agent"] = "general"

        return state

    async def route_decision(state: AnorvisState) -> str:
        """Decide which route to take based on classification."""
        logger.info(f"Route decision - current_agent: '{state['current_agent']}'")

        if state["current_agent"] == "coordinator":
            logger.info("Routing to coordinate_agents")
            return "coordinate"
        elif state["current_agent"] == "backrub":
            logger.info("Routing to route_to_backrub")
            return "backrub"
        elif state["current_agent"] == "warren":
            logger.info("Routing to route_to_warren")
            return "warren"
        else:
            logger.info("Routing to handle_general")
            return "general"

    async def route_to_backrub(state: AnorvisState) -> AnorvisState:
        """Route query to Backrub research agent."""
        query = state["messages"][-1].content
        logger.info("Routing to Backrub for research")

        # Get Backrub function
        backrub = builder.get_function(config.backrub)
        response = await backrub.ainvoke(query)

        # Check if Backrub suggests coordination
        if "coordinate" in response.lower() or "warren" in response.lower():
            state["coordination_needed"] = True
            state["suggested_agent"] = "warren"

        state["research_results"] = response
        state["messages"].append(AIMessage(content=f"Backrub response: {response}"))

        return state

    async def route_to_warren(state: AnorvisState) -> AnorvisState:
        """Route query to Warren finance agent."""
        query = state["messages"][-1].content
        logger.info("Routing to Warren for financial analysis")

        # Get Warren function
        warren = builder.get_function(config.warren)
        response = await warren.ainvoke(query)

        # Check if Warren suggests coordination
        if "coordinate" in response.lower() or "backrub" in response.lower():
            state["coordination_needed"] = True
            state["suggested_agent"] = "backrub"

        state["financial_analysis"] = response
        state["messages"].append(AIMessage(content=f"Warren response: {response}"))

        return state

    async def coordinate_agents(state: AnorvisState) -> AnorvisState:
        """Coordinate between multiple agents intelligently."""
        logger.info("Starting intelligent coordination between agents")

        original_query = state["original_query"]

        # Start with Backrub for research
        logger.info("Starting with Backrub for initial research...")
        backrub = builder.get_function(config.backrub)
        research_response = await backrub.ainvoke(original_query)
        logger.info(f"Backrub response received, length: {len(research_response)}")

        # Check if Backrub suggests financial analysis is needed
        if (
            "warren" in research_response.lower()
            or "financial" in research_response.lower()
        ):
            logger.info("Backrub suggested financial analysis, calling Warren...")
            warren = builder.get_function(config.warren)
            financial_response = await warren.ainvoke(original_query)
            logger.info(f"Warren response received, length: {len(financial_response)}")

            # Now let Warren analyze the research results and provide final advice
            logger.info("Getting Warren's final analysis based on research...")
            final_analysis_prompt = f"""
You are a financial advisor. The user is asking: {original_query}

Here is the research that has been done: {research_response[:1000]}

Based on this research, provide a final, confident answer with:
1. Specific stock recommendations based on the research
2. Clear investment advice and reasoning
3. Risk considerations
4. Actionable next steps

Be confident, specific, and give concrete advice. The user is waiting for your expert financial opinion.
"""

            final_analysis = await warren.ainvoke(final_analysis_prompt)
            logger.info(f"Final analysis received, length: {len(final_analysis)}")

            state["research_results"] = research_response
            state["financial_analysis"] = final_analysis
            state["messages"].append(AIMessage(content="Full coordination completed"))
            logger.info(
                "Full coordination completed - both agents finished conversation"
            )
        else:
            # Backrub didn't suggest financial analysis, so just use research
            logger.info(
                "Backrub didn't suggest financial analysis, using research only"
            )
            state["research_results"] = research_response
            state["financial_analysis"] = ""
            state["messages"].append(
                AIMessage(content="Research-only coordination completed")
            )
            logger.info("Research-only coordination completed")

        return state

    async def handle_general(state: AnorvisState) -> AnorvisState:
        """Handle general queries directly."""
        query = state["messages"][-1].content

        # Get LLM for general queries
        llm = await builder.get_llm(
            config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )

        messages = [
            SystemMessage(content=config.system_prompt),
            HumanMessage(content=query),
        ]

        response = await llm.ainvoke(messages)
        state["final_response"] = response.content
        state["messages"].append(AIMessage(content=response.content))

        return state

    async def finalize_response(state: AnorvisState) -> AnorvisState:
        """Finalize the response by combining results if needed."""
        if (
            state["current_agent"] == "coordinator"
            and state["research_results"]
            and state["financial_analysis"]
        ):
            # Get LLM to create a friendly summary
            llm = await builder.get_llm(
                config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
            )

            summary_prompt = f"""
You are a helpful friend who just completed research and got final financial advice. Create a short, friendly summary for the user.

Research results: {state["research_results"][:1000]}
Final financial advice: {state["financial_analysis"][:800]}

Write a brief, confident response that:
1. Starts with "Hey! I looked into that for you..."
2. Gives specific stock recommendations based on the research
3. Provides clear, actionable advice
4. Ends with a friendly reminder about doing your own research

Be confident and specific - the agents have finished their analysis and are ready to give you a final answer.
Keep it under 200 words and make it sound like a friend giving you solid advice!
"""

            messages = [
                SystemMessage(content="You are a helpful research assistant."),
                HumanMessage(content=summary_prompt),
            ]

            summary_response = await llm.ainvoke(messages)
            state["final_response"] = summary_response.content
        elif (
            state["current_agent"] == "coordinator"
            and state["research_results"]
            and not state["financial_analysis"]
        ):
            # Research-only coordination
            state["final_response"] = f"""
Hey! I found some info for you:

{state["research_results"][:500]}...

Want me to get some financial advice to go with this research?
"""
        elif state["research_results"]:
            state["final_response"] = f"""
Hey! Here's what I found:

{state["research_results"][:300]}...
"""
        elif state["financial_analysis"]:
            state["final_response"] = f"""
Hey! Here's what my financial expert friend had to say:

{state["financial_analysis"][:300]}...
"""
        elif state["final_response"]:
            # Already set by handle_general
            pass
        else:
            state["final_response"] = (
                "Sorry, I'm having trouble with that right now. Can you try again?"
            )

        return state

    yield _anorvis
