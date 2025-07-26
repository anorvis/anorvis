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
   export NVIDIA_API_KEY=<your_api_key>
   ```

### Running Anorvis

#### As a CLI Tool
```bash
# Run a simple query
aiq run --config_file examples/anorvis/configs/anorvis_config.yml --input "What can you help me with?"
```

#### As an API Server
```bash
# Start the API server
aiq serve --config_file examples/anorvis/configs/anorvis_config.yml

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
- `POST /chat` - Chat interface
- `POST /run` - Execute workflows
- `GET /health` - Health check
- `GET /docs` - API documentation

## Attribution

This project is based on the [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit), which is licensed under the Apache License, Version 2.0. See the [original repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit) for more information about the underlying technology.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md) for details.
