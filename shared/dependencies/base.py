"""
Base dependency injection utilities.

This module provides standardized patterns for dependency injection
across all services, ensuring consistency and testability.
"""

from functools import lru_cache
from typing import TypeVar, Callable
from contextlib import asynccontextmanager

from .dependency_container import DependencyContainer

T = TypeVar('T')


def cached_dependency(func: Callable[[], T]) -> Callable[[], T]:
    """
    Decorator to cache dependency results.

    This is a convenience wrapper around lru_cache for FastAPI dependencies.

    Example:
        >>> @cached_dependency
        >>> def get_my_service() -> MyService:
        ...     return MyService()
        >>>
        >>> # In FastAPI route:
        >>> @app.get("/endpoint")
        >>> async def endpoint(service: MyService = Depends(get_my_service)):
        ...     pass
    """
    return lru_cache(maxsize=1)(func)


@asynccontextmanager
async def get_async_session(session_factory: Callable):
    """
    Async context manager for database sessions.

    Ensures proper session lifecycle with automatic commit/rollback.

    Args:
        session_factory: Function that returns an async session

    Yields:
        Database session

    Example:
        >>> async with get_async_session(AsyncSessionLocal) as session:
        ...     result = await session.execute(query)
        ...     # Session automatically committed/rolled back
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Global container instance
_global_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """
    Get the global dependency container.

    Returns:
        Global DependencyContainer instance
    """
    return _global_container
