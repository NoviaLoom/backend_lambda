"""
Service authentication token payload.
"""

from datetime import datetime
from pydantic import BaseModel


class ServiceToken(BaseModel):
    """Service authentication token payload."""

    service: str  # Service name (e.g., "app-service")
    iat: datetime  # Issued at
    exp: datetime  # Expires at

