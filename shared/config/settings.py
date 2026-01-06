"""
Centralized configuration management using Pydantic Settings.

This module provides type-safe configuration with validation,
replacing scattered os.getenv() calls throughout the codebase.

Features:
- Type-safe configuration with Pydantic
- Automatic .env file loading
- Environment variable validation
- Default values
- Configuration inheritance for different services
"""

from .base_service_settings import BaseServiceSettings
from .core_service_settings import CoreServiceSettings
from .app_service_settings import AppServiceSettings
from .gateway_service_settings import GatewayServiceSettings


# Factory functions for creating settings

def get_core_settings() -> CoreServiceSettings:
    """
    Get Core Service settings.

    Returns:
        CoreServiceSettings instance

    Example:
        >>> settings = get_core_settings()
        >>> print(settings.google_api_key)
    """
    return CoreServiceSettings()


def get_app_settings() -> AppServiceSettings:
    """
    Get App Service settings.

    Returns:
        AppServiceSettings instance
    """
    return AppServiceSettings()


def get_gateway_settings() -> GatewayServiceSettings:
    """
    Get Gateway Service settings.

    Returns:
        GatewayServiceSettings instance
    """
    return GatewayServiceSettings()


# Generic settings getter (auto-detects service from env)

def get_settings() -> BaseServiceSettings:
    """
    Auto-detect service and return appropriate settings.

    Reads SERVICE_NAME from environment and returns the corresponding settings class.

    Returns:
        Service-specific settings instance

    Raises:
        ValueError: If SERVICE_NAME is unknown
    """
    import os
    service_name = os.getenv("SERVICE_NAME", "unknown")

    settings_map = {
        "core-service": get_core_settings,
        "app-service": get_app_settings,
        "gateway-service": get_gateway_settings
    }

    settings_factory = settings_map.get(service_name)
    if settings_factory is None:
        # Fallback to base settings
        return BaseServiceSettings()

    return settings_factory()
