# Anorvis Agent System

This directory contains the Anorvis multi-agent system built using AIQ Toolkit functions.

## Architecture

The system uses a base configuration class (`BaseAgentConfig`) that provides common functionality for all agents:

- **LLM Access**: All agents automatically get `llm_name` field
- **Anorvis Coordination**: All agents can call Anorvis for coordination via `anorvis` field
- **Foundational Tools**: Extensible system for common tools like `current_datetime`

## Current Agents

### Anorvis (Orchestrator)
- **Purpose**: Routes requests to appropriate specialized agents
- **Location**: `anorvis/anorvis.py`
- **Special Features**: Agent-specific references to Backrub and Warren

### Backrub (Research Agent)
- **Purpose**: Conducts research and information gathering
- **Location**: `backrub/backrub.py`
- **Special Features**: Coordinates with Anorvis for research

### Warren (Finance Agent)
- **Purpose**: Handles financial analysis and advice
- **Location**: `warren/warren.py`
- **Special Features**: Coordinates with Anorvis for research-heavy financial queries

## Adding a New Agent

To add a new agent, follow these steps:

### 1. Create Agent Configuration
```python
from ..base_config import BaseAgentConfig
from pydantic import Field

class NewAgentConfig(BaseAgentConfig, name="new_agent"):
    """Configuration for New Agent."""
    
    system_prompt: str = Field(
        "You are New Agent, a specialized agent for...",
        description="System prompt for the new agent.",
    )
    # Add any agent-specific fields here
```

### 2. Create Agent Function
```python
from aiq.cli.register_workflow import register_function
from aiq.builder.builder import Builder

@register_function(config_type=NewAgentConfig)
async def new_agent_function(config: NewAgentConfig, builder: Builder):
    """New Agent function."""
    pass
    
    async def _new_agent(query: str) -> str:
        # Get LLM and functions when needed
        llm = await builder.get_llm(
            config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
        )
        
        # Get Anorvis if available
        anorvis = None
        if config.anorvis:
            anorvis = builder.get_function(config.anorvis)
        
        # Your agent logic here
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": query},
        ]
        response = await llm.ainvoke(messages)
        return response.content
    
    yield _new_agent
```

### 3. Add to Configuration
```yaml
# In config/anorvis.yml
functions:
  new_agent:
    _type: new_agent
    anorvis: anorvis  # Automatically available
    llm_name: anorvis_llm  # Automatically available
    system_prompt: "Your agent's system prompt..."
```

### 4. Update Imports
Add your agent to `src/aiq/tool/agents/__init__.py`:
```python
from .new_agent.new_agent import new_agent_function

__all__ = [
    # ... existing agents
    "new_agent_function",
]
```

## Foundational Tools

The base configuration includes support for foundational tools that all agents can use:

- `current_datetime`: Get current time and date
- `web_search`: Web search capabilities (planned)
- `file_operations`: File system operations (planned)
- `calculator`: Mathematical calculations (planned)

To add a new foundational tool:

1. Add the field to `BaseAgentConfig` in `base_config.py`
2. Update `get_common_agent_config_fields()` function
3. Add the tool to the configuration file
4. Update agent logic to use the tool when available

## Testing

Test the system with various query types:

```bash
# General query
aiq run --config_file config/anorvis.yml --input "hello"

# Research query
aiq run --config_file config/anorvis.yml --input "research quantum computing"

# Financial query
aiq run --config_file config/anorvis.yml --input "show me my budget"

# Time/date query
aiq run --config_file config/anorvis.yml --input "what time is it"
```

## Benefits

- **Consistency**: All agents have the same base functionality
- **Easy Setup**: New agents just inherit `BaseAgentConfig`
- **Centralized Management**: Foundational tools defined once
- **Scalability**: Easy to add new tools and agents
- **Coordination**: All agents can call Anorvis for coordination 