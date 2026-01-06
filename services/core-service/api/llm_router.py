"""
LLM router for Core Service

Handles LLM generation requests with Google Gemini and OpenAI providers.
Uses shared package for consistent logging and responses.
"""

import os
import sys
from typing import Any

from fastapi import APIRouter, Depends

# Add parent directory to path to import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from shared.api.responses import error_response, server_error_response, success_response
from shared.log_config.config import get_logger

from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.llm_service import LLMService
from core.llm.providers.llm_provider_error import LLMProviderError
from core.llm.providers.llm_provider_timeout_error import LLMProviderTimeoutError

from .dependencies import get_llm_service

router = APIRouter()
logger = get_logger()


@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: LLMRequest,
    llm_service: LLMService = Depends(get_llm_service)
) -> LLMResponse:
    """
    Generate text using LLM provider.

    Supports Google Gemini and OpenAI providers with optional Google Search grounding.

    Args:
        request: LLM generation request with prompt, provider, model, etc.
        llm_service: Injected LLM service

    Returns:
        LLMResponse with generated text and metadata
    """
    try:
        logger.info(
            "llm_request_received",
            provider=request.provider,
            model=request.model,
            use_search=request.use_search,
            prompt_length=len(request.prompt)
        )

        response = await llm_service.generate(request)

        # Extract token usage from usage dict if available
        usage = response.usage or {}
        tokens_input = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        tokens_output = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        logger.info(
            "llm_request_completed",
            provider=response.provider,
            model=response.model,
            response_length=len(response.text),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            total_tokens=usage.get("total_tokens", tokens_input + tokens_output),
            finish_reason=response.finish_reason
        )

        return response

    except LLMProviderTimeoutError as e:
        logger.warning(
            "llm_request_timeout",
            provider=request.provider,
            error=str(e)
        )
        return error_response(
            message=f"Request timed out: {e.message}",
            error_code="LLM_TIMEOUT",
            status_code=408
        )

    except LLMProviderError as e:
        logger.error(
            "llm_provider_error",
            provider=request.provider,
            error=str(e),
            exc_info=True
        )
        return error_response(
            message=f"LLM provider error: {e.message}",
            error_code="LLM_PROVIDER_ERROR",
            status_code=502
        )

    except ValueError as e:
        logger.warning(
            "llm_request_validation_error",
            provider=request.provider,
            error=str(e)
        )
        return error_response(
            message=str(e),
            error_code="VALIDATION_ERROR",
            status_code=400
        )

    except Exception as e:
        logger.error(
            "llm_request_failed",
            provider=request.provider,
            error=str(e),
            exc_info=True
        )
        return server_error_response(
            message="Failed to generate text with LLM"
        )


@router.get("/providers")
async def get_providers(
    llm_service: LLMService = Depends(get_llm_service)
) -> dict[str, Any]:
    """
    Get list of available LLM providers.

    Returns:
        List of provider names (google, openai, etc.)
    """
    try:
        providers = llm_service.get_available_providers()

        logger.debug("providers_requested", providers_count=len(providers))

        return success_response(
            data={"providers": providers},
            message=f"{len(providers)} providers available"
        )

    except Exception as e:
        logger.error("get_providers_failed", error=str(e), exc_info=True)
        return server_error_response("Failed to get available providers")


@router.get("/models")
async def get_models(
    provider: str = None,
    llm_service: LLMService = Depends(get_llm_service)
) -> dict[str, Any]:
    """
    Get available models for LLM providers.

    Args:
        provider: Specific provider to get models for (optional, returns all if None)

    Returns:
        Dict mapping provider names to list of available models
    """
    try:
        models = await llm_service.get_available_models(provider)

        logger.debug(
            "models_requested",
            provider=provider or "all",
            models_count=sum(len(m) for m in models.values())
        )

        return success_response(
            data={"models": models},
            message=f"Models retrieved for {provider or 'all providers'}"
        )

    except ValueError as e:
        logger.warning("get_models_validation_error", provider=provider, error=str(e))
        return error_response(
            message=str(e),
            error_code="INVALID_PROVIDER",
            status_code=400
        )

    except Exception as e:
        logger.error("get_models_failed", provider=provider, error=str(e), exc_info=True)
        return server_error_response("Failed to get available models")


@router.get("/health")
async def llm_health_check(
    llm_service: LLMService = Depends(get_llm_service)
) -> dict[str, Any]:
    """
    Check health of LLM providers.

    Returns health status for each configured provider.
    """
    try:
        health_status = await llm_service.health_check()

        all_healthy = all(health_status.values())
        status = "healthy" if all_healthy else "degraded"

        logger.debug("llm_health_check", status=status, providers=health_status)

        return success_response(
            data={
                "status": status,
                "providers": health_status
            }
        )

    except Exception as e:
        logger.error("llm_health_check_failed", error=str(e), exc_info=True)
        return error_response(
            message="Health check failed",
            error_code="HEALTH_CHECK_ERROR",
            status_code=503,
            details={"error": str(e)}
        )
