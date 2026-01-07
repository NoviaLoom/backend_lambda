"""
Settings specific to Core Service.
"""

from typing import Optional
from pydantic import Field, field_validator

from .base_service_settings import BaseServiceSettings


class CoreServiceSettings(BaseServiceSettings):
    """Settings specific to Core Service."""

    service_name: str = "core-service"
    port: int = Field(default=8001, description="Port to run Core Service on")

    # LLM API Keys
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    @field_validator("google_api_key", "openai_api_key")
    @classmethod
    def validate_api_keys(cls, v: Optional[str]) -> Optional[str]:
        """Validate API keys are not empty strings."""
        if v == "":
            return None
        return v

