"""
LLM Factory for creating provider instances
"""

import os
import sys
from typing import Any

# Add parent directory to path to import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from .providers.google_provider import GoogleProvider
from .providers.llm_provider_base import LLMProviderBase
from .providers.openai_provider import OpenAIProvider


class LLMFactory:
    """Factory for creating LLM provider instances"""

    # Registry of available providers
    _providers: dict[str, type[LLMProviderBase]] = {
        "google": GoogleProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str, **kwargs: Any) -> LLMProviderBase:
        """
        Create a provider instance

        Args:
            provider_name: Name of the provider (google, openai)
            **kwargs: Additional configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unsupported provider '{provider_name}'. Available: {available}")

        provider_class = cls._providers[provider_name]

        # Get API key from environment or kwargs
        api_key = kwargs.get("api_key")
        if not api_key:
            api_key = cls._get_api_key(provider_name)

        if not api_key:
            raise ValueError(f"API key not found for provider '{provider_name}'")

        # Get enable_mock_llm from settings if not provided in kwargs
        if "enable_mock_llm" not in kwargs:
            try:
                from shared.config.settings import get_core_settings
                settings = get_core_settings()
                kwargs["enable_mock_llm"] = settings.enable_mock_llm
            except Exception:
                # Fallback to environment variable if settings not available
                env_value = os.getenv("ENABLE_MOCK_LLM")
                if env_value is not None:
                    kwargs["enable_mock_llm"] = env_value.lower() == "true"
                else:
                    kwargs["enable_mock_llm"] = False

        return provider_class(api_key=api_key, **kwargs)

    @classmethod
    def _get_api_key(cls, provider_name: str) -> str:
        """Get API key from environment variables"""
        key_mapping = {
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        env_var = key_mapping.get(provider_name)
        if not env_var:
            return ""

        return os.getenv(env_var, "")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProviderBase]):
        """
        Register a new provider

        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name] = provider_class

    @classmethod
    def create_default_provider(cls) -> LLMProviderBase:
        """
        Create default provider (Google Gemini)

        Returns:
            Default provider instance
        """
        return cls.create_provider("google")
