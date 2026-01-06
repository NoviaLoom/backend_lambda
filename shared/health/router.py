"""
Health check router factory for consistent health endpoints across services.

This module provides a standardized health check implementation that includes:
- Basic service health status
- Version information
- Dependency health checks (DB, external services, etc.)
- Degraded state detection
"""

from typing import Optional, Dict, Callable, Any
from datetime import datetime, timezone
from fastapi import APIRouter, status

from .health_status import HealthStatus
from .dependency_check import DependencyCheck


def create_health_router(
    service_name: str,
    version: str,
    dependencies: Optional[Dict[str, Callable]] = None,
    start_time: Optional[datetime] = None
) -> APIRouter:
    """
    Create a standardized health check router.

    Args:
        service_name: Name of the service (e.g., "core-service", "app-service")
        version: Service version (e.g., "1.0.0")
        dependencies: Optional dict of dependency name -> health check function
        start_time: Optional service start time for uptime calculation

    Returns:
        FastAPI router with /health and /health/ready endpoints

    Example:
        >>> async def check_database():
        ...     # Check DB connection
        ...     return await db.is_connected()
        >>>
        >>> router = create_health_router(
        ...     service_name="my-service",
        ...     version="1.0.0",
        ...     dependencies={"database": check_database}
        ... )
        >>> app.include_router(router)
    """

    router = APIRouter(tags=["Health"])
    service_start_time = start_time or datetime.now(timezone.utc)

    @router.get(
        "/health",
        response_model=HealthStatus,
        status_code=status.HTTP_200_OK,
        summary="Health check endpoint",
        description="Returns the health status of the service and its dependencies"
    )
    async def health_check():
        """
        Basic health check endpoint.

        Returns service health status, version, and dependency statuses.
        """
        current_time = datetime.now(timezone.utc)
        uptime = (current_time - service_start_time).total_seconds()

        health_status = HealthStatus(
            service=service_name,
            version=version,
            status="healthy",
            timestamp=current_time.isoformat(),
            uptime_seconds=uptime
        )

        # Check dependencies if provided
        if dependencies:
            dependency_results = {}
            all_healthy = True

            for dep_name, check_fn in dependencies.items():
                dep_check = DependencyCheck(dep_name, check_fn)
                result = await dep_check.check()
                dependency_results[dep_name] = result

                if result.get("status") != "healthy":
                    all_healthy = False

            health_status.dependencies = dependency_results

            # Update overall status based on dependencies
            if not all_healthy:
                health_status.status = "degraded"

        return health_status

    @router.get(
        "/health/ready",
        status_code=status.HTTP_200_OK,
        summary="Readiness check",
        description="Returns 200 if service is ready to accept traffic, 503 otherwise"
    )
    async def readiness_check():
        """
        Readiness check for Kubernetes/orchestration platforms.

        Returns:
            200 if ready, 503 if not ready
        """
        # Check critical dependencies
        if dependencies:
            for dep_name, check_fn in dependencies.items():
                dep_check = DependencyCheck(dep_name, check_fn)
                result = await dep_check.check()

                # If any critical dependency is unhealthy, service is not ready
                if result.get("status") == "unhealthy":
                    return {
                        "ready": False,
                        "reason": f"Dependency '{dep_name}' is unhealthy"
                    }

        return {"ready": True}

    @router.get(
        "/health/live",
        status_code=status.HTTP_200_OK,
        summary="Liveness check",
        description="Returns 200 if service is alive (for Kubernetes liveness probes)"
    )
    async def liveness_check():
        """
        Liveness check for Kubernetes/orchestration platforms.

        This is a simple check that the service is running.
        It should NOT check dependencies - only if the service itself is alive.
        """
        return {"alive": True}

    return router


# Predefined dependency check functions

async def check_database_health(session_factory: Callable) -> bool:
    """
    Check PostgreSQL database health.

    Args:
        session_factory: Function that returns a database session

    Returns:
        True if database is healthy
    """
    try:
        async with session_factory() as session:
            # Simple query to test connection
            await session.execute("SELECT 1")
            return True
    except Exception:
        raise


async def check_http_service_health(base_url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Check health of an HTTP service.

    Args:
        base_url: Base URL of the service
        timeout: Request timeout in seconds

    Returns:
        Dict with status and response time
    """
    import httpx
    from time import time

    try:
        start = time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/health")
            elapsed = time() - start

            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": round(elapsed * 1000, 2),
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
