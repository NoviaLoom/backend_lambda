"""
Status enum for various processes.
"""

from enum import Enum


class ServiceStatus(str, Enum):
    """Status enum for various processes."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

