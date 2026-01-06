"""
Health status response model.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class HealthStatus(BaseModel):
    """Health status response model."""

    service: str
    version: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    uptime_seconds: Optional[float] = None
    dependencies: Optional[Dict[str, Any]] = None

