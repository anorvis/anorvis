# SPDX-FileCopyrightText: Copyright (c) 2024-2025, Anorvis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field
from pydantic import ConfigDict

from aiq.builder.builder import Builder
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.retry_mixin import RetryMixin


class OllamaModelConfig(LLMBaseConfig, RetryMixin, name="ollama"):
    """An Ollama LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=())

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API server.",
    )
    model_name: str = Field(
        description="The name of the Ollama model to use (e.g., 'llama2', 'mistral', 'codellama')."
    )
    temperature: float = Field(
        default=0.0, description="Sampling temperature in [0, 1]."
    )
    top_p: float = Field(default=1.0, description="Top-p for distribution sampling.")
    max_tokens: int = Field(
        default=300, description="Maximum number of tokens to generate."
    )
    timeout: int = Field(default=120, description="Request timeout in seconds.")


@register_llm_provider(config_type=OllamaModelConfig)
async def ollama_llm(config: OllamaModelConfig, builder: Builder):
    """Register Ollama LLM provider."""
    yield LLMProviderInfo(
        config=config, description="An Ollama model for use with an LLM client."
    )
