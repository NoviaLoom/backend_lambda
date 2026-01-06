"""
Quota exceeded error for LLM providers
"""

from .llm_provider_error import LLMProviderError


class LLMProviderQuotaExceededError(LLMProviderError):
    """Quota exceeded error for LLM providers"""
    pass

