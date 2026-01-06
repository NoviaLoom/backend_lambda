"""
Standardized API response utilities.

This module provides consistent response formats and helpers for
returning data from API endpoints.
"""

from typing import Optional, Any, Dict
from datetime import datetime
from fastapi import status
from fastapi.responses import JSONResponse

from .api_response import APIResponse


# Response helper functions

def success_response(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK
) -> JSONResponse:
    """
    Create a successful API response.

    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code (default: 200)

    Returns:
        JSONResponse with standardized format

    Example:
        >>> @app.get("/stores")
        >>> async def get_stores():
        ...     stores = await fetch_stores()
        ...     return success_response(stores, "Stores retrieved successfully")
    """
    response = APIResponse(
        success=True,
        data=data,
        message=message
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(mode='json', exclude_none=True)
    )


def error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create an error API response.

    Args:
        message: Error message
        error_code: Optional error code for client handling
        status_code: HTTP status code (default: 400)
        details: Optional additional error details

    Returns:
        JSONResponse with standardized error format

    Example:
        >>> @app.get("/stores/{store_id}")
        >>> async def get_store(store_id: str):
        ...     store = await fetch_store(store_id)
        ...     if not store:
        ...         return error_response(
        ...             "Store not found",
        ...             error_code="STORE_NOT_FOUND",
        ...             status_code=404
        ...         )
        ...     return success_response(store)
    """
    response_data = {
        "success": False,
        "error": message,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        response_data["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


def created_response(
    data: Any,
    message: Optional[str] = None
) -> JSONResponse:
    """
    Create a 201 Created response.

    Use this when a new resource has been created.

    Args:
        data: Created resource data
        message: Optional success message

    Returns:
        JSONResponse with 201 status code

    Example:
        >>> @app.post("/stores")
        >>> async def create_store(store_data: StoreCreate):
        ...     store = await create_new_store(store_data)
        ...     return created_response(store, "Store created successfully")
    """
    return success_response(
        data=data,
        message=message or "Resource created successfully",
        status_code=status.HTTP_201_CREATED
    )


def no_content_response() -> JSONResponse:
    """
    Create a 204 No Content response.

    Use this for successful operations that don't return data (e.g., DELETE).

    Returns:
        JSONResponse with 204 status code

    Example:
        >>> @app.delete("/stores/{store_id}")
        >>> async def delete_store(store_id: str):
        ...     await remove_store(store_id)
        ...     return no_content_response()
    """
    return JSONResponse(
        status_code=status.HTTP_204_NO_CONTENT,
        content=None
    )


def not_found_response(
    resource_type: str,
    resource_id: str
) -> JSONResponse:
    """
    Create a 404 Not Found response.

    Args:
        resource_type: Type of resource (e.g., "Store", "User")
        resource_id: ID of the resource

    Returns:
        JSONResponse with 404 status code

    Example:
        >>> return not_found_response("Store", store_id)
    """
    return error_response(
        message=f"{resource_type} not found",
        error_code=f"{resource_type.upper()}_NOT_FOUND",
        status_code=status.HTTP_404_NOT_FOUND,
        details={"resource_id": resource_id}
    )


def unauthorized_response(
    message: str = "Authentication required"
) -> JSONResponse:
    """
    Create a 401 Unauthorized response.

    Args:
        message: Error message

    Returns:
        JSONResponse with 401 status code
    """
    return error_response(
        message=message,
        error_code="UNAUTHORIZED",
        status_code=status.HTTP_401_UNAUTHORIZED
    )


def forbidden_response(
    message: str = "Insufficient permissions"
) -> JSONResponse:
    """
    Create a 403 Forbidden response.

    Args:
        message: Error message

    Returns:
        JSONResponse with 403 status code
    """
    return error_response(
        message=message,
        error_code="FORBIDDEN",
        status_code=status.HTTP_403_FORBIDDEN
    )


def conflict_response(
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a 409 Conflict response.

    Use this when there's a conflict with existing data (e.g., duplicate email).

    Args:
        message: Error message
        details: Optional conflict details

    Returns:
        JSONResponse with 409 status code

    Example:
        >>> return conflict_response(
        ...     "Email already exists",
        ...     details={"email": user_email}
        ... )
    """
    return error_response(
        message=message,
        error_code="CONFLICT",
        status_code=status.HTTP_409_CONFLICT,
        details=details
    )


def validation_error_response(
    errors: Dict[str, Any]
) -> JSONResponse:
    """
    Create a 422 Validation Error response.

    Args:
        errors: Validation errors dictionary

    Returns:
        JSONResponse with 422 status code

    Example:
        >>> return validation_error_response({
        ...     "email": "Invalid email format",
        ...     "age": "Must be >= 18"
        ... })
    """
    return error_response(
        message="Validation failed",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"errors": errors}
    )


def server_error_response(
    message: str = "Internal server error"
) -> JSONResponse:
    """
    Create a 500 Internal Server Error response.

    Args:
        message: Error message

    Returns:
        JSONResponse with 500 status code
    """
    return error_response(
        message=message,
        error_code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def service_unavailable_response(
    service_name: str
) -> JSONResponse:
    """
    Create a 503 Service Unavailable response.

    Use this when a dependent service is down.

    Args:
        service_name: Name of the unavailable service

    Returns:
        JSONResponse with 503 status code

    Example:
        >>> return service_unavailable_response("core-service")
    """
    return error_response(
        message=f"{service_name} is currently unavailable",
        error_code="SERVICE_UNAVAILABLE",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        details={"service": service_name}
    )
