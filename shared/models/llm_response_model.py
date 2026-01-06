"""
Response model for LLM generation.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMResponse(BaseModel):
    """Response model for LLM generation."""

    text: str = Field(description="Generated text")
    provider: str = Field(description="Provider used")
    model: str = Field(description="Model used")
    tokens_input: Optional[int] = Field(default=None, description="Input tokens")
    tokens_output: Optional[int] = Field(default=None, description="Output tokens")
    execution_time_seconds: Optional[float] = Field(default=None, description="Execution time")
    search_used: bool = Field(default=False, description="Whether search grounding was used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Analysis results...",
                "provider": "google",
                "model": "gemini-2.5-flash-lite",
                "tokens_input": 250,
                "tokens_output": 500,
                "execution_time_seconds": 2.4,
                "search_used": True
            }
        }
    )

