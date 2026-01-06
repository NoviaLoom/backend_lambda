"""
Response model for captation execution.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .service_status import ServiceStatus


class CaptationResponse(BaseModel):
    """Response model for captation execution."""

    session_id: str = Field(description="Captation session ID")
    status: ServiceStatus = Field(description="Current status")
    progress_percentage: float = Field(ge=0, le=100, description="Progress percentage")
    prompts_completed: int = Field(ge=0, description="Number of prompts completed")
    prompts_total: int = Field(ge=0, description="Total number of prompts")
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Captured raw data")
    processing_time_seconds: Optional[float] = Field(default=None, description="Processing time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "session-123",
                "status": "completed",
                "progress_percentage": 100.0,
                "prompts_completed": 7,
                "prompts_total": 7,
                "processing_time_seconds": 45.2
            }
        }
    )

