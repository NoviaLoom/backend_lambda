"""
Supported output formats for reports enumeration.
"""

from enum import Enum


class OutputFormat(str, Enum):
    """Supported output formats for reports."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"

