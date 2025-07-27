<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, Anorvis. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Anorvis AI Agent

Anorvis is an intelligent AI agent built on top of the NVIDIA NeMo Agent toolkit, designed to provide powerful, flexible, and customizable AI capabilities for your applications.

> [!NOTE]
> This project is based on the [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) (previously known as Agent Intelligence (AIQ) toolkit). The original project provides a flexible, lightweight, and unifying library that allows you to easily connect existing enterprise agents to data sources and tools across any framework.

## Key Features

- **Framework Agnostic**: Works with existing agentic frameworks without replatforming
- **Reusable Components**: Build once, use everywhere with composable agents and tools
- **Rapid Development**: Start with pre-built components and customize for your needs
- **Profiling & Monitoring**: Track performance and debug workflows effectively
- **API Ready**: Built-in REST API for easy frontend integration
- **Observability**: Monitor and debug with OpenTelemetry-compatible tools

## Quick Start

### Prerequisites

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Python (3.11 or 3.12)](https://www.python.org/downloads/)
- NVIDIA API Key (get one at [build.nvidia.com](https://build.nvidia.com/))
- OpenAI API Key (get one at [platform.openai.com](https://platform.openai.com/))

### Installation

1. Clone the Anorvis repository:
   ```bash
   git clone https://github.com/your-username/anorvis.git
   cd anorvis
   ```

2. Set up the environment:
   ```bash
   uv venv --seed .venv
   source .venv/bin/activate
   uv sync
   ```

3. Set your API key:
   ```bash
   export OPENAI_API_KEY=<your_api_key>
   ```

### Running Anorvis

#### As a CLI Tool
```bash
# Run a simple query
aiq run --config_file config/anorvis.yml --input "What can you help me with?"
```

#### As an API Server
```bash
# Start the API server
aiq serve --config_file config/anorvis.yml

# Then call from your frontend
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Anorvis!"}'
```

## Development

### Creating Custom Agents
```bash
# Create a new domain agent
aiq workflow create my_domain_agent --workflow-dir examples/anorvis/agents --description "Custom domain agent"
```

### Adding Tools
```bash
# See available tools
aiq info components -t function

# Add tools to your configuration
```

## API Endpoints

When running as a server, Anorvis provides these endpoints:

- `POST /chat` - Send a message to Anorvis
- `GET /health` - Check server health
- `GET /docs` - Interactive API documentation

## Architecture

Anorvis is built as a multi-agent system with the following components:

### Core Agents

- **Anorvis**: The main orchestrator agent that routes requests to specialized agents
- **Backrub**: A specialized research agent for information gathering and analysis
- **Warren**: A finance agent for financial analysis, budgeting, and investment advice

### Agent Communication

Agents can communicate with each other through the Anorvis orchestrator, enabling complex workflows:

- `Anorvis → Backrub` for research tasks
- `Anorvis → Warren` for financial analysis
- `Anorvis → Backrub → Anorvis → Warren` for coordinated research and financial analysis

### StateGraph Orchestration

The system uses LangGraph's StateGraph for advanced agent coordination, allowing for:

- Intelligent query classification using LLM-based intent recognition
- Conditional routing based on query content
- Multi-agent coordination for complex tasks
- Automatic response synthesis from multiple agents

## Configuration

The main configuration file is `config/anorvis.yml`, which defines:

- LLM providers and models
- Agent configurations and system prompts
- Workflow definitions
- Tool assignments

### Example Configuration

```yaml
llms:
  anorvis_llm:
    _type: openai
    model_name: gpt-3.5-turbo
    temperature: 0.1
    max_tokens: 1000

functions:
  anorvis:
    _type: anorvis
    llm_name: anorvis_llm
    backrub: backrub
    warren: warren
    system_prompt: |
      You are Anorvis, an intelligent AI assistant that coordinates specialized agents.
      You analyze user requests and route them to the appropriate specialist agent.

  backrub:
    _type: backrub
    llm_name: anorvis_llm
    system_prompt: |
      You are Backrub, a specialized research agent. You excel at finding information, 
      conducting research, and gathering data.

  warren:
    _type: warren
    llm_name: anorvis_llm
    system_prompt: |
      You are Warren, a specialized finance agent. You excel at financial analysis, 
      budgeting, investments, and financial planning.

workflow:
  _type: tool_calling_agent
  llm_name: anorvis_llm
  tool_names: [anorvis]
  system_prompt: |
    You are an AI assistant that helps users by routing their requests to the appropriate specialist agent.
    Use the anorvis tool to handle all user requests.
```

## Extending Anorvis

### Adding New Agents

1. Create a new agent file in `src/aiq/tool/agents/<agent_name>/<agent_name>.py`
2. Inherit from `BaseAgentConfig` for common functionality
3. Register the agent using the `@register_function` decorator
4. Add the agent to your configuration file

### Adding New Tools

1. Create tool functions using the `@register_function` decorator
2. Add tools to agent configurations as needed
3. Update the base configuration for foundational tools

### Example Agent Template

```python
from aiq.cli.register_workflow import register_function
from aiq.builder.builder import Builder
from ..base_config import BaseAgentConfig

class MyAgentConfig(BaseAgentConfig, name="my_agent"):
    system_prompt: str = Field(
        "You are MyAgent, a specialized agent for...",
        description="System prompt for MyAgent."
    )

@register_function(config_type=MyAgentConfig)
async def my_agent(config: MyAgentConfig, builder: Builder):
    async def _my_agent(query: str) -> str:
        # Agent logic here
        return response
    
    yield _my_agent
```

## Testing

Test the multi-agent system with various query types:

```bash
# Research query
aiq run --config_file config/anorvis.yml --input "What are the latest developments in AI?"

# Financial query
aiq run --config_file config/anorvis.yml --input "How should I invest $10,000?"

# Coordinated query
aiq run --config_file config/anorvis.yml --input "Research the AI market and provide investment advice"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:

- Check the [documentation](docs/)
- Open an issue on GitHub
- Join our community discussions

## Acknowledgments

- Built on the [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
- Uses OpenAI's GPT models for natural language processing
- Leverages LangGraph for agent orchestration
