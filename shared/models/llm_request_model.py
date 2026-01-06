"""
Request model for LLM generation.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMRequest(BaseModel):
    """Request model for LLM generation."""

    prompt: str = Field(min_length=1, description="Prompt text")
    provider: str = Field(default="google", description="LLM provider (google, openai)")
    model: Optional[str] = Field(default=None, description="Specific model (auto-select if None)")
    use_search: bool = Field(default=False, description="Enable Google Search grounding")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Max output tokens")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Analyze this store data...",
                "provider": "google",
                "model": "gemini-2.5-flash-lite",
                "use_search": True,
                "temperature": 0.3
            }
        }
    )

