"""
Structured logging configuration using structlog.

This module provides a standardized logging setup that outputs structured JSON logs
for better parsing, searching, and analysis in production environments.

Features:
- JSON output for production
- Pretty console output for development
- Automatic context injection (service name, request ID, etc.)
- Log level filtering
- Performance-optimized
"""

import os
import sys
import logging
from typing import Optional
import structlog


def configure_logging(
    service_name: str,
    log_level: Optional[str] = None,
    json_logs: Optional[bool] = None
) -> structlog.BoundLogger:
    """
    Configure structured logging for a service.

    Args:
        service_name: Name of the service (e.g., "core-service")
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON. If False, pretty console. If None, auto-detect from env

    Returns:
        Configured logger instance

    Example:
        >>> logger = configure_logging("my-service", "INFO")
        >>> logger.info("service_started", port=8000)
    """

    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Determine output format based on environment
    if json_logs is None:
        environment = os.getenv("ENVIRONMENT", "development")
        json_logs = environment == "production"

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level)
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Shared processors for all environments
    shared_processors = [
        # Add log level
        structlog.stdlib.add_log_level,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add context variables
        structlog.contextvars.merge_contextvars,
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
    ]

    # Choose renderer based on environment
    if json_logs:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Pretty console output with colors
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create logger and bind service context
    logger = structlog.get_logger()
    logger = logger.bind(service=service_name)

    return logger


def get_logger() -> structlog.BoundLogger:
    """
    Get a logger instance.

    Returns:
        Logger instance (must call configure_logging first)
    """
    return structlog.get_logger()


def bind_context(**kwargs) -> None:
    """
    Bind context variables to all subsequent log messages in this context.

    This is useful for adding request-scoped data like request_id, user_id, etc.

    Args:
        **kwargs: Key-value pairs to add to log context

    Example:
        >>> bind_context(request_id="abc-123", user_id="user-456")
        >>> logger.info("processing_request")
        # Output includes: request_id="abc-123", user_id="user-456"
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all context variables.

    Call this at the end of request processing to avoid context leaking.
    """
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """
    Remove specific context variables.

    Args:
        *keys: Keys to remove from context

    Example:
        >>> unbind_context("request_id", "user_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


class LoggerMiddleware:
    """
    FastAPI middleware for automatic request logging and context injection.

    This middleware:
    - Logs all incoming requests
    - Adds request_id to all logs
    - Logs response status and duration
    - Clears context after request
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import uuid
        import time

        request_id = str(uuid.uuid4())
        method = scope["method"]
        path = scope["path"]

        # Bind request context
        bind_context(
            request_id=request_id,
            method=method,
            path=path
        )

        logger = get_logger()
        
        # Skip verbose logging for health checks and polling endpoints (called frequently)
        is_health_check = path in ["/health", "/health/ready", "/health/live"] or path.startswith("/health")
        is_polling_endpoint = path.endswith("/status") or path.endswith("/logs")
        
        if not is_health_check and not is_polling_endpoint:
            logger.info("request_started")

        start_time = time.time()

        async def send_wrapper(message):
            """Intercept response to log completion."""
            if message["type"] == "http.response.start":
                duration_ms = round((time.time() - start_time) * 1000, 2)
                status_code = message["status"]

                # Only log health checks if they fail or take too long
                if is_health_check:
                    if status_code != 200 or duration_ms > 1000:
                        logger.warning(
                            "health_check_issue",
                            status_code=status_code,
                            duration_ms=duration_ms
                        )
                    # Skip normal health check logs
                elif is_polling_endpoint:
                    # Log polling endpoints only at DEBUG level or if there's an issue
                    if status_code != 200 or duration_ms > 500:
                        logger.warning(
                            "polling_endpoint_issue",
                            status_code=status_code,
                            duration_ms=duration_ms
                        )
                    # Skip normal polling endpoint logs (too frequent)
                else:
                    logger.info(
                        "request_completed",
                        status_code=status_code,
                        duration_ms=duration_ms
                    )

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                "request_failed",
                error=str(e),
                duration_ms=duration_ms,
                exc_info=True
            )
            raise
        finally:
            # Clear context to avoid leaking between requests
            clear_context()


def setup_logging_middleware(app) -> None:
    """
    Setup logging middleware for FastAPI application.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> setup_logging_middleware(app)
    """
    app.add_middleware(LoggerMiddleware)


class FilteredAccessLogger:
    """
    Access logger pour Uvicorn qui filtre les health checks et les endpoints de polling.
    """
    def __init__(self, original_logger):
        self.original_logger = original_logger
    
    def __call__(self, scope, message):
        """Filtre les logs d'accès pour les health checks et les endpoints de polling."""
        if scope.get("type") == "http":
            path = scope.get("path", "")
            # Ne pas logger les health checks
            if path.startswith("/health"):
                return
            # Ne pas logger les endpoints de polling (trop fréquents)
            if path.endswith("/status") or path.endswith("/logs"):
                return
        # Logger les autres requêtes normalement
        if self.original_logger:
            self.original_logger(scope, message)
