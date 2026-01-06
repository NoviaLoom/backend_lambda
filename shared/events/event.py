"""
Base event class for the event bus.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .event_priority import EventPriority


@dataclass
class Event:
    """
    Base event class for the event bus.

    All events should use this structure to ensure consistency
    across services.
    """

    name: str  # Event name (e.g., "captation.started", "batch.completed")
    data: Dict[str, Any]  # Event payload
    source_service: str  # Service that published the event
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None  # For tracing related events

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "name": self.name,
            "data": self.data,
            "source_service": self.source_service,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            name=data["name"],
            data=data["data"],
            source_service=data["source_service"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=EventPriority(data.get("priority", "normal")),
            correlation_id=data.get("correlation_id")
        )

