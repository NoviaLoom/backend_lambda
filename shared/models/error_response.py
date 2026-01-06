"""
Error response model.
"""

from typing import Optional, Dict, Any
from pydantic import Field, ConfigDict

from .base_response import BaseResponse


class ErrorResponse(BaseResponse):
    """Error response model."""

    success: bool = Field(default=False)
    error_code: Optional[str] = Field(default=None, description="Error code for client handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "message": "Store not found",
                "error_code": "STORE_NOT_FOUND",
                "details": {"store_id": "store-123"},
                "timestamp": "2025-11-05T12:00:00Z"
            }
        }
    )

