"""
Response model for analysis execution.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .service_status import ServiceStatus


class AnalysisResponse(BaseModel):
    """Response model for analysis execution."""

    session_id: str = Field(description="Analysis session ID")
    status: ServiceStatus = Field(description="Current status")
    progress_percentage: float = Field(ge=0, le=100, description="Progress percentage")
    processors_completed: int = Field(ge=0, description="Number of processors completed")
    processors_total: int = Field(ge=0, description="Total number of processors")
    analysis_data: Optional[Dict[str, Any]] = Field(default=None, description="Analysis results")
    processing_time_seconds: Optional[float] = Field(default=None, description="Processing time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "analysis-123",
                "status": "completed",
                "progress_percentage": 100.0,
                "processors_completed": 6,
                "processors_total": 6,
                "processing_time_seconds": 32.1
            }
        }
    )

