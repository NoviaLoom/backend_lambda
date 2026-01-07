"""
LLM Request Model
"""

from typing import Any

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Request model for LLM generation"""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    provider: str = Field(..., description="LLM provider (google, openai)")
    model: str | None = Field(None, description="Specific model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens to generate")
    system_message: str | None = Field(None, description="System message for context")
    messages: list[dict[str, str]] | None = Field(None, description="Chat messages format")
    stream: bool = Field(False, description="Whether to stream the response")

    # Google Search support
    use_search: bool = Field(
        False,
        description="Enable Google Search grounding (Gemini only)"
    )

    # Google Maps support
    use_maps: bool = Field(
        False,
        description="Enable Google Maps grounding (Gemini only)"
    )

    # Advanced grounding configuration
    grounding_config: dict[str, Any] | None = Field(
        None,
        description="Advanced grounding configuration"
    )

    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Find information about Decathlon Lyon store",
                "provider": "google",
                "model": "gemini-1.5-pro",
                "temperature": 0.1,
                "max_tokens": 32000,
                "use_search": True,  # Enable Google Search
                "use_maps": True,    # Enable Google Maps
                "system_message": "You are a retail data analyst"
            }
        }
