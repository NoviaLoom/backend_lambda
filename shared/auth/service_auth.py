"""
Service-to-service authentication using JWT tokens.

This module provides secure authentication for inter-service communication,
replacing hardcoded tokens with proper JWT-based authentication.

Features:
- JWT token generation for service identity
- Token verification with signature validation
- Automatic token expiry
- Service name validation
- FastAPI dependency for easy integration
"""

from typing import Optional
from fastapi import Header, HTTPException, status

from .service_token import ServiceToken
from .service_authenticator import ServiceAuthenticator


# Global authenticator instance (initialized once)
_authenticator: Optional[ServiceAuthenticator] = None


def get_service_authenticator() -> ServiceAuthenticator:
    """
    Get or create the global service authenticator instance.

    Returns:
        ServiceAuthenticator instance
    """
    global _authenticator
    if _authenticator is None:
        _authenticator = ServiceAuthenticator()
    return _authenticator


# FastAPI dependencies for service authentication

async def verify_service_token_header(
    x_service_token: str = Header(..., description="Service authentication token")
) -> ServiceToken:
    """
    FastAPI dependency to verify service token from header.

    Args:
        x_service_token: JWT token from X-Service-Token header

    Returns:
        ServiceToken if valid

    Raises:
        HTTPException: 401 if token is invalid

    Example:
        >>> @router.post("/internal/endpoint")
        >>> async def internal_endpoint(
        ...     service: ServiceToken = Depends(verify_service_token_header)
        ... ):
        ...     return {"called_by": service.service}
    """
    try:
        authenticator = get_service_authenticator()
        return authenticator.verify_service_token(x_service_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid service token: {str(e)}"
        )


def require_service(allowed_services: list[str]):
    """
    Create a FastAPI dependency that requires specific services.

    Args:
        allowed_services: List of allowed service names

    Returns:
        FastAPI dependency function

    Example:
        >>> require_app_service = require_service(["app-service"])
        >>>
        >>> @router.post("/internal/captation")
        >>> async def captation_endpoint(
        ...     service: ServiceToken = Depends(require_app_service)
        ... ):
        ...     return {"message": "Only app-service can call this"}
    """
    async def verify_specific_service(
        x_service_token: str = Header(..., description="Service authentication token")
    ) -> ServiceToken:
        try:
            authenticator = get_service_authenticator()
            return authenticator.verify_service(x_service_token, allowed_services)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )

    return verify_specific_service
