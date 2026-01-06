"""
Base HTTP client with automatic service authentication.
"""

from typing import Optional
import httpx

from .service_authenticator import get_service_authenticator


class ServiceHttpClient:
    """
    Base HTTP client with automatic service authentication.

    This class wraps httpx.AsyncClient and automatically adds service tokens
    to all outgoing requests.
    """

    def __init__(self, service_name: str, base_url: str):
        """
        Initialize service HTTP client.

        Args:
            service_name: Name of this service (for token generation)
            base_url: Base URL of the target service
        """
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.authenticator = get_service_authenticator()

    def _get_headers(self) -> dict:
        """Generate headers with service token."""
        token = self.authenticator.generate_service_token(self.service_name)
        return {
            "X-Service-Token": token,
            "Content-Type": "application/json"
        }

    async def post(self, path: str, json: dict, timeout: int = 30):
        """
        POST request with service authentication.

        Args:
            path: Endpoint path (e.g., "/captation/execute")
            json: Request body
            timeout: Request timeout in seconds

        Returns:
            Response from httpx
        """
        url = f"{self.base_url}{path}"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=json, headers=headers)
            response.raise_for_status()
            return response

    async def get(self, path: str, params: Optional[dict] = None, timeout: int = 30):
        """
        GET request with service authentication.

        Args:
            path: Endpoint path
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Response from httpx
        """
        url = f"{self.base_url}{path}"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response

