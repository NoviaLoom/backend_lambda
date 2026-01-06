"""
Request model for captation execution.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

from .captation_prompt import CaptationPrompt


class CaptationRequest(BaseModel):
    """Request model for captation execution."""

    session_id: Optional[str] = Field(default=None, description="Optional session ID to resume")
    store_id: str = Field(description="Store ID to capture data for")
    client_id: str = Field(description="Client ID (multi-tenant)")
    theme_id: str = Field(description="Theme ID to use for captation")
    prompts: List[CaptationPrompt] = Field(description="List of prompts to execute")
    use_search: bool = Field(default=True, description="Enable Google Search grounding")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Global variables")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store_id": "store-123",
                "client_id": "client-456",
                "theme_id": "theme-789",
                "prompts": [],
                "use_search": True,
                "variables": {"store_name": "Decathlon Paris"}
            }
        }
    )

