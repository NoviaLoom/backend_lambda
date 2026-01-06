"""
Request model for batch report generation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .output_format_enum import OutputFormat


class BatchReportRequest(BaseModel):
    """Request model for batch report generation."""

    network_id: str = Field(description="Network ID to generate reports for")
    client_id: str = Field(description="Client ID (multi-tenant)")
    theme_id: str = Field(description="Theme ID to use")
    report_types: List[str] = Field(default=["standard"], description="Report types to generate")
    output_format: OutputFormat = Field(default=OutputFormat.HTML, description="Output format")
    store_ids: Optional[List[str]] = Field(default=None, description="Specific stores (or all if None)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "network_id": "decathlon-france",
                "client_id": "client-123",
                "theme_id": "theme-456",
                "report_types": ["standard"],
                "output_format": "html"
            }
        }
    )

