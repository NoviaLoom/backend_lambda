"""
Request priority enumeration for LLM queue service
"""

from enum import Enum


class RequestPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1

