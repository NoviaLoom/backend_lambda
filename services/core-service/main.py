"""
Core Service - LLM Infrastructure
Port: 8001

This service provides LLM abstraction layer for Google Gemini and OpenAI.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI

# Add parent directory to path to import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Shared imports
from shared.config.settings import get_core_settings
from shared.health.router import create_health_router
from shared.log_config.config import configure_logging, setup_logging_middleware
from shared.middleware.cors import configure_cors

# Local imports
from api.llm_router import router as llm_router
from api.embeddings_router import router as embeddings_router

# Initialize settings
settings = get_core_settings()

# Initialize structured logging
logger = configure_logging(
    service_name=settings.service_name,
    log_level=settings.log_level
)

# Service start time for uptime tracking
start_time = datetime.now(UTC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown.

    This replaces the old @app.on_event("startup") and @app.on_event("shutdown")
    which are deprecated in FastAPI.
    """
    # Startup
    logger.info(
        "service_starting",
        service=settings.service_name,
        environment=settings.environment,
        port=settings.port,
        google_api_configured=settings.google_api_key is not None,
        openai_api_configured=settings.openai_api_key is not None,
        bedrock_configured=True  # Bedrock uses IAM role
    )

    yield

    # Shutdown
    logger.info(
        "service_stopping",
        service=settings.service_name
    )


# Create FastAPI app
app = FastAPI(
    title="Novialoom Core Service",
    description="LLM Infrastructure Service - Google Gemini & OpenAI abstraction layer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS (environment-aware, secure)
configure_cors(app)

# Setup logging middleware for request tracking
setup_logging_middleware(app)

# Health check with LLM provider dependency checks
async def check_llm_providers():
    """Check if LLM providers are properly configured."""
    has_google = settings.google_api_key is not None
    has_openai = settings.openai_api_key is not None
    has_bedrock = True  # Bedrock uses IAM role, always available in Lambda

    if not has_google and not has_openai and not has_bedrock:
        return {
            "status": "unhealthy",
            "error": "No LLM providers configured"
        }

    providers_status = {
        "google": "available" if has_google else "not_configured",
        "openai": "available" if has_openai else "not_configured",
        "bedrock": "available",  # Always available with IAM role
    }

    return {
        "status": "healthy",
        "providers": providers_status
    }


health_router = create_health_router(
    service_name=settings.service_name,
    version="1.0.0",
    dependencies={"llm_providers": check_llm_providers},
    start_time=start_time
)

# Include routers
app.include_router(health_router)
app.include_router(llm_router, prefix="/llm", tags=["LLM"])
app.include_router(embeddings_router)  # Prefix /embed already in router


@app.get("/", response_model=None)
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "environment": settings.environment,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.is_development,  # Auto-reload only in development
        access_log=False  # Désactiver les logs d'accès Uvicorn (géré par notre middleware)
    )


# ============================================
# Lambda Handler (for AWS Lambda deployment)
# ============================================
try:
    from mangum import Mangum
    # lifespan="off" prevents Lambda timeout issues with startup/shutdown events
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Mangum not installed, skip Lambda handler (local dev)
    handler = None  # type: ignore
