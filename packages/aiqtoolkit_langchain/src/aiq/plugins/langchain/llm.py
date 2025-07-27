# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.data_models.retry_mixin import RetryMixin
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig
from aiq.llm.ollama_llm import OllamaModelConfig
from aiq.utils.exception_handlers.automatic_retries import patch_with_retry


@register_llm_client(
    config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def nim_langchain(llm_config: NIMModelConfig, builder: Builder):
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(**llm_config.model_dump(exclude={"type"}, by_alias=True))

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    yield client


@register_llm_client(
    config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def openai_langchain(llm_config: OpenAIModelConfig, builder: Builder):
    from langchain_openai import ChatOpenAI

    # Default kwargs for OpenAI to include usage metadata in the response. If the user has set stream_usage to False, we
    # will not include this.
    default_kwargs = {"stream_usage": True}

    kwargs = {
        **default_kwargs,
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
    }

    client = ChatOpenAI(**kwargs)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    yield client


@register_llm_client(
    config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def aws_bedrock_langchain(llm_config: AWSBedrockModelConfig, builder: Builder):
    from langchain_aws import ChatBedrockConverse

    client = ChatBedrockConverse(
        **llm_config.model_dump(exclude={"type", "context_size"}, by_alias=True)
    )

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    yield client


@register_llm_client(
    config_type=OllamaModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN
)
async def ollama_langchain(llm_config: OllamaModelConfig, builder: Builder):
    """Register Ollama LangChain client."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama is required for Ollama support. "
            "Install it with: pip install langchain-ollama"
        )

    # Convert AIQ config to LangChain Ollama config - use explicit field mapping
    ollama_config = {
        "base_url": llm_config.base_url,
        "model": llm_config.model_name,
        "temperature": llm_config.temperature,
        "top_p": llm_config.top_p,
        "num_predict": llm_config.max_tokens,
        "timeout": llm_config.timeout,
    }

    client = ChatOllama(**ollama_config)

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )

    yield client
