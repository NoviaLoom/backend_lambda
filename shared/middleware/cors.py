"""
CORS middleware configuration shared across all services.

This module provides a centralized CORS configuration that ensures
security and consistency across all microservices.
"""

import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None
) -> None:
    """
    Configure CORS middleware with appropriate security settings.

    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins. If None, reads from env or uses defaults
        allow_credentials: Whether to allow credentials (cookies, auth headers)
        allowed_methods: List of allowed HTTP methods. If None, uses safe defaults
        allowed_headers: List of allowed headers. If None, allows all

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> configure_cors(app, allowed_origins=["http://localhost:3000"])
    """

    # Get environment
    environment = os.getenv("ENVIRONMENT", "development")

    # Determine allowed origins based on environment
    if allowed_origins is None:
        if environment == "production":
            # In production, use specific origins from env
            origins_env = os.getenv("ALLOWED_ORIGINS", "")
            allowed_origins = [
                origin.strip()
                for origin in origins_env.split(",")
                if origin.strip()
            ]

            # Default production origins if not specified
            if not allowed_origins:
                allowed_origins = [
                    "https://app.novialoom.com",
                    "https://www.novialoom.com"
                ]
        else:
            # Development: Allow localhost on various ports
            # Also allow wildcard if specified in env
            origins_env = os.getenv("ALLOWED_ORIGINS", "")
            if origins_env == "*":
                allowed_origins = ["*"]
            else:
                allowed_origins = [
                    "http://localhost:3000",
                    "http://localhost:4200",
                    "http://localhost:4201",
                    "http://localhost:8000",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:4200",
                    "http://127.0.0.1:4201",
                ]
                if origins_env:
                    allowed_origins.extend([o.strip() for o in origins_env.split(",") if o.strip()])

    # Default allowed methods (safe subset)
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

    # Default allowed headers
    if allowed_headers is None:
        allowed_headers = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
    )


def configure_cors_permissive(app: FastAPI) -> None:
    """
    Configure permissive CORS for development/testing only.

    WARNING: This allows all origins. NEVER use in production!

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
