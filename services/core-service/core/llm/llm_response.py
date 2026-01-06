"""
LLM Response Model
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Response model for LLM generation"""

    text: str = Field(..., description="Generated text content")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used for generation")
    usage: dict[str, int] | None = Field(None, description="Token usage statistics")
    finish_reason: str | None = Field(None, description="Reason for completion")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str | None = Field(None, description="Request identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Based on the store data analysis, here are the key insights...",
                "provider": "google",
                "model": "gemini-1.5-flash",
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 300,
                    "total_tokens": 450
                },
                "finish_reason": "stop",
                "created_at": "2025-01-10T10:30:00Z"
            }
        }
