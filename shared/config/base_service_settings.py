"""
Base settings shared by all services.
"""

from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """
    Base settings shared by all services.

    All services inherit from this class and add their specific settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )

    # Service identification
    service_name: str = Field(
        default="unknown-service",
        description="Name of the service"
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    # JWT Authentication
    jwt_secret: Optional[str] = Field(
        default=None,
        min_length=32,
        description="Secret key for JWT token signing (min 32 chars, required for Gateway Service)"
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@postgres:5432/novialoom",
        description="PostgreSQL connection URL (async)"
    )

    # Service URLs
    core_service_url: str = Field(
        default="http://core-service:8001",
        description="URL of the Core Service"
    )

    # DEPRECATED: app-service migré vers microservices spécialisés
    # app_service_url: str = Field(
    #     default="http://app-service:8002",
    #     description="URL of the App Service"
    # )

    # DEPRECATED: gateway-service remplacé par b2b-gateway-service et b2c-gateway-service
    # gateway_service_url: str = Field(
    #     default="http://gateway-service:8000",
    #     description="URL of the Gateway Service"
    # )
    
    # New Gateway Services
    b2b_gateway_url: str = Field(
        default="http://b2b-gateway-service:8013",
        description="URL of the B2B Gateway Service"
    )
    
    b2c_gateway_url: str = Field(
        default="http://b2c-gateway-service:8011",
        description="URL of the B2C Gateway Service"
    )

    # CORS
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:4200",
        description="Comma-separated list of allowed CORS origins"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v_upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v_lower

    def get_allowed_origins_list(self) -> List[str]:
        """
        Parse allowed origins from comma-separated string.

        Returns:
            List of allowed origin URLs
        """
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

