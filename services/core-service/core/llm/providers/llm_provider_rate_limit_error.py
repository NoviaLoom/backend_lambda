"""
Rate limit error for LLM providers
"""

from .llm_provider_error import LLMProviderError


class LLMProviderRateLimitError(LLMProviderError):
    """Rate limit error for LLM providers"""
    pass

