"""
Dependencies for Core Service API

Uses shared package for consistent dependency injection patterns.
"""

import os
import sys
from functools import lru_cache

# Add parent directory to path to import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from shared.config.settings import get_core_settings
from shared.dependencies.service_factory import ServiceFactory

from core.llm.llm_service import LLMService

# Initialize settings
settings = get_core_settings()

# Create LLM service factory
# Note: LLMService doesn't need API keys as arguments - it reads them from environment
# via LLMFactory which uses os.getenv()
_llm_service_factory = ServiceFactory(LLMService)


@lru_cache
def get_llm_service() -> LLMService:
    """
    Get LLM service instance (singleton).

    The service is configured with API keys from settings.
    Uses ServiceFactory for consistent instantiation.

    Returns:
        LLMService: Configured LLM service instance
    """
    return _llm_service_factory.get_instance()
