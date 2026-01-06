"""
Status model for batch report generation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .service_status import ServiceStatus


class BatchReportStatus(BaseModel):
    """Status model for batch report generation."""

    batch_id: str = Field(description="Batch report ID")
    status: ServiceStatus = Field(description="Current status")
    progress_percentage: float = Field(ge=0, le=100, description="Progress percentage")
    total_stores: int = Field(ge=0, description="Total stores to process")
    completed_stores: int = Field(ge=0, description="Stores completed")
    failed_stores: int = Field(ge=0, description="Stores failed")
    current_store: Optional[str] = Field(default=None, description="Currently processing store")
    estimated_time_remaining_seconds: Optional[float] = Field(default=None, description="ETA")
    report_urls: Optional[List[str]] = Field(default=None, description="Download URLs when complete")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "batch_id": "batch-123",
                "status": "in_progress",
                "progress_percentage": 42.0,
                "total_stores": 10,
                "completed_stores": 4,
                "failed_stores": 0,
                "current_store": "store-456"
            }
        }
    )

