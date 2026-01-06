"""
Settings specific to Gateway Service.
"""

from pydantic import Field

from .base_service_settings import BaseServiceSettings


class GatewayServiceSettings(BaseServiceSettings):
    """Settings specific to Gateway Service."""

    service_name: str = "gateway-service"
    port: int = Field(default=8000, description="Port to run Gateway Service on")

    # JWT Token settings
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Access token expiry time in minutes"
    )

    jwt_refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Refresh token expiry time in days"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    # Security
    enable_service_auth: bool = Field(
        default=True,
        description="Enable service-to-service authentication"
    )

