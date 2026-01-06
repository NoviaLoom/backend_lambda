"""
LLM Service - Main service for LLM operations
"""

import logging
from typing import Any

from .llm_factory import LLMFactory
from .llm_request import LLMRequest
from .llm_response import LLMResponse

logger = logging.getLogger(__name__)


class LLMService:
    """Main LLM service for handling generation requests"""

    def __init__(self):
        """Initialize LLM service"""
        self._providers: dict[str, Any] = {}
        self._default_provider = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using specified provider

        Args:
            request: LLM request with prompt and parameters

        Returns:
            LLM response with generated text

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            # Get or create provider
            provider = await self._get_provider(request.provider)

            # Generate response
            logger.info(f"Generating with provider: {request.provider}, model: {request.model}")

            response = await provider.generate(request)

            logger.info(f"Generated {len(response.text)} characters")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def _get_provider(self, provider_name: str):
        """Get or create provider instance"""
        if provider_name not in self._providers:
            try:
                self._providers[provider_name] = LLMFactory.create_provider(provider_name)
                logger.info(f"Created provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to create provider {provider_name}: {str(e)}")
                raise

        return self._providers[provider_name]

    async def health_check(self, provider_name: str | None = None) -> dict[str, bool]:
        """
        Check health of providers

        Args:
            provider_name: Specific provider to check (optional)

        Returns:
            Dictionary with provider health status
        """
        results = {}

        if provider_name:
            # Check specific provider
            try:
                provider = await self._get_provider(provider_name)
                results[provider_name] = await provider.health_check()
            except Exception:
                results[provider_name] = False
        else:
            # Check all available providers
            for name in LLMFactory.get_available_providers():
                try:
                    provider = await self._get_provider(name)
                    results[name] = await provider.health_check()
                except Exception:
                    results[name] = False

        return results

    async def get_available_models(self, provider_name: str | None = None) -> dict[str, list[str]]:
        """
        Get available models for providers

        Args:
            provider_name: Specific provider (optional)

        Returns:
            Dictionary with provider models
        """
        models = {}

        if provider_name:
            # Get models for specific provider
            try:
                provider = await self._get_provider(provider_name)
                models[provider_name] = provider.get_available_models()
            except Exception:
                models[provider_name] = []
        else:
            # Get models for all providers
            for name in LLMFactory.get_available_providers():
                try:
                    provider = await self._get_provider(name)
                    models[name] = provider.get_available_models()
                except Exception:
                    models[name] = []

        return models

    def get_available_providers(self) -> list[str]:
        """Get list of available providers"""
        return LLMFactory.get_available_providers()
