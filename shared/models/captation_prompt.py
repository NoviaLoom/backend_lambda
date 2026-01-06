"""
Individual captation prompt configuration.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CaptationPrompt(BaseModel):
    """Individual captation prompt configuration."""

    prompt_number: int = Field(ge=1, le=10, description="Prompt sequence number")
    prompt_title: str = Field(min_length=1, max_length=200, description="Prompt title")
    prompt_content: str = Field(min_length=10, description="Prompt content/template")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Variables for template")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt_number": 1,
                "prompt_title": "General Information",
                "prompt_content": "Extract general info about {{store_name}}",
                "variables": {"store_name": "Decathlon Paris"}
            }
        }
    )

