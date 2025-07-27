"""
Base Configuration for Anorvis Agents

Provides common configuration and functionality that all agents in the Anorvis system should have.
"""

import logging
from typing import Any
from pydantic import Field

from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef, LLMRef

logger = logging.getLogger(__name__)


class BaseAgentConfig(FunctionBaseConfig):
    """
    Base configuration class for all Anorvis agents.

    This class provides common configuration fields that all agents should have:
    - LLM reference for language model access
    - Anorvis orchestrator reference for coordination
    - Foundational tools that all agents should have access to
    """

    # Common LLM configuration - all agents need an LLM
    llm_name: LLMRef = Field(description="The LLM model to use with this agent.")

    # Anorvis orchestrator for coordination - all agents can call Anorvis (optional for orchestrator itself)
    anorvis: FunctionRef | None = Field(
        default=None, description="Reference to Anorvis orchestrator for coordination."
    )

    # Foundational tools that all agents should have access to
    current_datetime: FunctionRef | None = Field(
        default=None, description="Reference to the current datetime tool."
    )

    # Add more foundational tools here as needed
    # web_search: FunctionRef | None = Field(
    #     default=None,
    #     description="Reference to web search tool."
    # )
    # file_operations: FunctionRef | None = Field(
    #     default=None,
    #     description="Reference to file operations tool."
    # )
    # calculator: FunctionRef | None = Field(
    #     default=None,
    #     description="Reference to calculator tool."
    # )


def get_common_agent_config_fields() -> dict[str, str]:
    """
    Get the common configuration fields that should be included in all agent configs.

    Returns:
        Dictionary of common field definitions for agent configurations
    """
    return {
        "llm_name": "anorvis_llm",
        "anorvis": "anorvis",
        "current_datetime": "current_datetime",
        # Add more foundational tools as they become available
        # "web_search": "web_search",
        # "file_operations": "file_operations",
        # "calculator": "calculator",
    }


def create_agent_config_template(
    agent_name: str, system_prompt: str, **additional_fields: Any
) -> dict[str, Any]:
    """
    Create a template configuration for a new agent.

    Args:
        agent_name: The name of the agent (e.g., "backrub", "warren")
        system_prompt: The system prompt for the agent
        **additional_fields: Additional configuration fields specific to this agent

    Returns:
        Dictionary template for the agent configuration
    """
    base_config = get_common_agent_config_fields()

    config = {
        "_type": f"{agent_name}_agent",
        "llm_name": base_config["llm_name"],
        "anorvis": base_config["anorvis"],
        "system_prompt": system_prompt,
        **additional_fields,
    }

    # Add optional foundational tools if they exist
    if base_config.get("current_datetime"):
        config["current_datetime"] = base_config["current_datetime"]

    return config
