"""
OpenAI Provider
"""

import asyncio
from typing import Any

from openai import AsyncOpenAI

from ..llm_request import LLMRequest
from ..llm_response import LLMResponse
from .llm_provider_base import LLMProviderBase
from .llm_provider_error import LLMProviderError
from .llm_provider_timeout_error import LLMProviderTimeoutError


class OpenAIProvider(LLMProviderBase):
    """OpenAI LLM Provider"""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

        # Available models
        self.models = {
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }

        # Default model
        self.default_model = "gpt-3.5-turbo"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using OpenAI"""
        try:
            self._validate_request(request)

            # Select model
            model_name = request.model or self.default_model
            if model_name not in self.models:
                model_name = self.default_model

            # Prepare messages
            messages = []

            # Add system message if provided
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})

            # Add user message
            messages.append({"role": "user", "content": request.prompt})

            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens or 2048,
                "stream": request.stream
            }

            # Generate with timeout
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**request_params),
                    timeout=60.0
                )
            except TimeoutError as e:
                raise LLMProviderTimeoutError(
                    "Request timed out",
                    provider="openai"
                ) from e

            # Parse response
            if not response.choices or not response.choices[0].message.content:
                raise LLMProviderError(
                    "Empty response from OpenAI",
                    provider="openai"
                )

            # Extract usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return LLMResponse(
                text=response.choices[0].message.content,
                provider="openai",
                model=model_name,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                request_id=None,
                metadata={
                    "response_id": response.id,
                    "model": response.model,
                    "object": response.object
                }
            )

        except Exception as e:
            if isinstance(e, LLMProviderError | LLMProviderTimeoutError):
                raise
            raise LLMProviderError(
                f"Generation failed: {str(e)}",
                provider="openai"
            ) from e

    async def health_check(self) -> bool:
        """Check if OpenAI is accessible"""
        try:
            # Simple health check with minimal request
            test_request = LLMRequest(
                prompt="Hello",
                provider="openai",
                model="gpt-3.5-turbo",
                max_tokens=10
            )

            # Set timeout for health check
            response = await asyncio.wait_for(
                self.generate(test_request),
                timeout=10.0
            )

            return bool(response.text)

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get available OpenAI models"""
        return list(self.models.keys())
