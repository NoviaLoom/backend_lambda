"""
Base LLM Provider Interface
"""

from abc import ABC, abstractmethod
from typing import Any

from ..llm_request import LLMRequest
from ..llm_response import LLMResponse


class LLMProviderBase(ABC):
    """Base class for LLM providers"""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initialize provider with API key"""
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using the LLM provider

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLM response with generated text and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """
        Get list of available models for this provider

        Returns:
            List of model names
        """
        pass

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate request parameters"""
        if not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if request.temperature < 0 or request.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if request.max_tokens is not None and request.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
