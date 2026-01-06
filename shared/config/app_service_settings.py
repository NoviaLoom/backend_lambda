"""
Settings specific to App Service.
"""

from pydantic import Field

from .base_service_settings import BaseServiceSettings


class AppServiceSettings(BaseServiceSettings):
    """Settings specific to App Service."""

    service_name: str = "app-service"
    port: int = Field(default=8002, description="Port to run App Service on")

    # Storage
    reports_storage_path: str = Field(
        default="/app/volumes/reports",
        description="Path for storing generated reports"
    )

    reports_base_url: str = Field(
        default="http://localhost:8002",
        description="Base URL for accessing reports"
    )

    # Processing limits
    max_parallel_stores: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of stores to process in parallel (MVP: 1)"
    )

    captation_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=600,
        description="Timeout for captation process per store"
    )

    analysis_timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=600,
        description="Timeout for analysis process per store"
    )

