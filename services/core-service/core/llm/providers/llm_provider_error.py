"""
Base exception for LLM provider errors
"""



class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""

    def __init__(self, message: str, provider: str, error_code: str | None = None):
        self.message = message
        self.provider = provider
        self.error_code = error_code
        super().__init__(f"[{provider}] {message}")

