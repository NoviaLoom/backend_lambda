"""
Timeout error for LLM providers
"""

from .llm_provider_error import LLMProviderError


class LLMProviderTimeoutError(LLMProviderError):
    """Timeout error for LLM providers"""
    pass

