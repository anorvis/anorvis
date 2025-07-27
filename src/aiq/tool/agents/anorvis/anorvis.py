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
        "You only handle general queries and time/date requests directly when no specialist agent is available.",
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
        Given the user input below, classify it as one of the following categories:
        
        - 'backrub': Research, information gathering, data analysis, finding facts, 
          exploring topics, academic queries, "what is", "how does", "tell me about", etc.
        - 'warren': Financial analysis, budgeting, investments, stock market, economic analysis, 
          cost analysis, financial planning, money management, etc.
        - 'coordinate': Queries that require BOTH research AND financial analysis 
          (e.g., "research AI and analyze financial impact", "study market trends and provide investment advice")
        - 'general': General questions, greetings, time/date requests, or anything that doesn't fit the above categories
        
        Respond with ONLY the category name (backrub, warren, coordinate, or general).
        
        User query: {input}
        Classification:"""

        # Create routing chain
        routing_chain = (
            {"input": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(router_prompt)
            | llm
            | StrOutputParser()
        )

        # Get classification
        classification = await routing_chain.ainvoke(query)
        classification = classification.strip().lower()

        logger.info(f"LLM classified query as: {classification}")

        # Set state based on classification
        if classification == "coordinate":
            state["coordination_needed"] = True
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
        if state["coordination_needed"]:
            return "coordinate"
        elif state["current_agent"] == "backrub":
            return "backrub"
        elif state["current_agent"] == "warren":
            return "warren"
        else:
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
        """Coordinate between multiple agents."""
        logger.info("Coordinating between agents")

        # Get both agents
        backrub = builder.get_function(config.backrub)
        warren = builder.get_function(config.warren)

        original_query = state["original_query"]

        # Get both research and financial analysis
        logger.info("Getting both research and financial analysis")
        research_response = await backrub.ainvoke(original_query)
        financial_response = await warren.ainvoke(original_query)

        state["research_results"] = research_response
        state["financial_analysis"] = financial_response
        state["messages"].append(AIMessage(content="Coordination completed"))

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
            state["coordination_needed"]
            and state["research_results"]
            and state["financial_analysis"]
        ):
            # Combine research and financial analysis
            combined_response = f"""
**Comprehensive Analysis for: {state["original_query"]}**

**Research Findings:**
{state["research_results"]}

**Financial Analysis:**
{state["financial_analysis"]}

**Summary:**
This analysis combines thorough research with professional financial insights to provide you with a complete picture.
"""
            state["final_response"] = combined_response
        elif state["research_results"]:
            state["final_response"] = state["research_results"]
        elif state["financial_analysis"]:
            state["final_response"] = state["financial_analysis"]
        elif state["final_response"]:
            # Already set by handle_general
            pass
        else:
            state["final_response"] = (
                "I apologize, but I wasn't able to process your request properly."
            )

        return state

    yield _anorvis
