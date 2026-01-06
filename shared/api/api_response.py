"""
Standard API response wrapper.
"""

from typing import Optional, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field


T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """
    Standard API response wrapper.

    All API endpoints should return responses in this format for consistency.
    """

    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[T] = Field(default=None, description="Response data")
    message: Optional[str] = Field(default=None, description="Optional message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(default=None, description="Error code for client handling")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

