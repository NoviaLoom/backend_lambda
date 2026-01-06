"""
Request model for analysis execution.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class AnalysisRequest(BaseModel):
    """Request model for analysis execution."""

    session_id: Optional[str] = Field(default=None, description="Optional session ID to resume")
    store_id: str = Field(description="Store ID to analyze")
    client_id: str = Field(description="Client ID (multi-tenant)")
    captation_session_id: str = Field(description="Captation session to analyze")
    theme_id: str = Field(description="Theme ID for analysis prompts")
    use_search: bool = Field(default=False, description="Enable search (usually False for analysis)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store_id": "store-123",
                "client_id": "client-456",
                "captation_session_id": "captation-789",
                "theme_id": "theme-789",
                "use_search": False
            }
        }
    )

