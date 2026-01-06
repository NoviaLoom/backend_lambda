"""
Dependency health check wrapper.
"""

from typing import Callable, Dict, Any


class DependencyCheck:
    """Wrapper for dependency health checks."""

    def __init__(self, name: str, check_fn: Callable):
        """
        Initialize dependency check.

        Args:
            name: Name of the dependency (e.g., "database", "redis", "core-service")
            check_fn: Async function that returns True if healthy, raises exception if not
        """
        self.name = name
        self.check_fn = check_fn

    async def check(self) -> Dict[str, Any]:
        """
        Execute the health check.

        Returns:
            Dict with status and optional error message
        """
        try:
            result = await self.check_fn()
            if isinstance(result, bool):
                return {"status": "healthy" if result else "unhealthy"}
            elif isinstance(result, dict):
                return result
            else:
                return {"status": "healthy", "details": str(result)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

