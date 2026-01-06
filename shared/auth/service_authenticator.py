"""
Service-to-service authentication using JWT tokens.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import jwt
from jwt.exceptions import InvalidTokenError

from .service_token import ServiceToken


class ServiceAuthenticator:
    """
    Handles service-to-service authentication using JWT.

    This class provides methods to generate and verify service tokens,
    ensuring secure communication between microservices.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize service authenticator.

        Args:
            secret_key: Secret key for signing tokens. If None, reads from JWT_SECRET env var
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET")
        if not self.secret_key:
            raise ValueError(
                "JWT_SECRET environment variable must be set for service authentication"
            )

        self.algorithm = "HS256"

    def generate_service_token(
        self,
        service_name: str,
        expiry_hours: int = 1
    ) -> str:
        """
        Generate a JWT token for service-to-service authentication.

        Args:
            service_name: Name of the calling service (e.g., "app-service", "gateway-service")
            expiry_hours: Token expiry time in hours (default: 1 hour)

        Returns:
            JWT token string

        Example:
            >>> auth = ServiceAuthenticator()
            >>> token = auth.generate_service_token("app-service")
            >>> # Use token in HTTP headers: {"X-Service-Token": token}
        """
        now = datetime.utcnow()
        payload = {
            "service": service_name,
            "iat": now.timestamp(),
            "exp": (now + timedelta(hours=expiry_hours)).timestamp()
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_service_token(self, token: str) -> ServiceToken:
        """
        Verify and decode a service token.

        Args:
            token: JWT token to verify

        Returns:
            ServiceToken with decoded payload

        Raises:
            ValueError: If token is invalid, expired, or malformed

        Example:
            >>> auth = ServiceAuthenticator()
            >>> try:
            ...     service_token = auth.verify_service_token(token)
            ...     print(f"Request from service: {service_token.service}")
            ... except ValueError as e:
            ...     print(f"Invalid token: {e}")
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            return ServiceToken(
                service=payload["service"],
                iat=datetime.fromtimestamp(payload["iat"]),
                exp=datetime.fromtimestamp(payload["exp"])
            )

        except InvalidTokenError as e:
            raise ValueError(f"Invalid service token: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Malformed service token: missing field {str(e)}")

    def verify_service(
        self,
        token: str,
        allowed_services: Optional[list[str]] = None
    ) -> ServiceToken:
        """
        Verify token and check if service is allowed.

        Args:
            token: JWT token to verify
            allowed_services: List of allowed service names. If None, any service is allowed

        Returns:
            ServiceToken if valid and allowed

        Raises:
            ValueError: If token is invalid or service not allowed
        """
        service_token = self.verify_service_token(token)

        if allowed_services and service_token.service not in allowed_services:
            raise ValueError(
                f"Service '{service_token.service}' is not authorized for this endpoint"
            )

        return service_token


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

