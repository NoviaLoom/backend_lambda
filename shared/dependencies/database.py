"""
Database dependency utilities.

Provides standardized database session management for FastAPI.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool


class DatabaseSessionManager:
    """
    Manages database connections and sessions.

    This class handles the lifecycle of database connections,
    ensuring proper cleanup and connection pooling.
    """

    def __init__(self, database_url: str, echo: bool = False, schema: str = None):
        """
        Initialize database session manager.

        Args:
            database_url: PostgreSQL connection URL (async format)
            echo: Whether to log SQL queries
            schema: Default schema to use (e.g., 'auth' or 'business')
        """
        self.database_url = database_url
        self.echo = echo
        self.schema = schema
        self._engine = None
        self._session_factory = None

    def init(self) -> None:
        """Initialize database engine and session factory."""
        if self._engine is not None:
            return  # Already initialized

        # Prepare connect_args with schema if specified
        connect_args = {}
        if self.schema:
            # Set search_path to prioritize the specified schema
            connect_args["server_settings"] = {
                "search_path": f"{self.schema},public"
            }

        self._engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            poolclass=NullPool,  # Use NullPool for async to avoid connection issues
            pool_pre_ping=True,  # Verify connections before using
            connect_args=connect_args,  # Add schema-specific settings
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    async def close(self) -> None:
        """Close database engine and cleanup resources."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.

        Yields:
            AsyncSession instance

        Example:
            >>> async for session in manager.get_session():
            ...     result = await session.execute(query)
        """
        if self._session_factory is None:
            self.init()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global session manager (initialized per service)
_session_manager: DatabaseSessionManager = None


def init_database(database_url: str, echo: bool = False, schema: str = None) -> DatabaseSessionManager:
    """
    Initialize global database session manager.

    Call this once at service startup.

    Args:
        database_url: PostgreSQL connection URL
        echo: Whether to log SQL queries
        schema: Default schema to use (e.g., 'auth' or 'business')

    Returns:
        DatabaseSessionManager instance

    Example:
        >>> # In main.py startup (Gateway)
        >>> manager = init_database(settings.database_url, schema="auth")
        >>> # In main.py startup (App)
        >>> manager = init_database(settings.database_url, schema="business")
    """
    global _session_manager
    _session_manager = DatabaseSessionManager(database_url, echo, schema)
    _session_manager.init()
    return _session_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Use this as a dependency in route handlers.

    Yields:
        AsyncSession instance

    Example:
        >>> @router.get("/stores")
        >>> async def get_stores(session: AsyncSession = Depends(get_db_session)):
        ...     result = await session.execute(select(Store))
        ...     return result.scalars().all()
    """
    if _session_manager is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() at startup."
        )

    async for session in _session_manager.get_session():
        yield session


async def close_database() -> None:
    """
    Close database connections.

    Call this at service shutdown.

    Example:
        >>> # In main.py shutdown
        >>> await close_database()
    """
    global _session_manager
    if _session_manager:
        await _session_manager.close()
        _session_manager = None
