"""
Event bus for decoupled inter-service communication.

This module provides an event-driven architecture pattern that allows
services to communicate without tight coupling. Services can publish
events and subscribe to events without knowing about each other.

Features:
- Async event publishing and handling
- Multiple subscribers per event
- Error isolation (one handler failure doesn't affect others)
- Structured logging of events
- Support for both local and remote event handlers
"""

from typing import Callable, Dict, List, Any, Optional, Awaitable
import asyncio

from .event_priority import EventPriority
from .event import Event

# Type alias for event handler functions
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Event bus for publishing and subscribing to events.

    This is a simple in-memory event bus. For production with multiple
    instances, consider using Redis Pub/Sub or a message queue.

    Example:
        >>> bus = EventBus("app-service")
        >>>
        >>> @bus.subscribe("captation.completed")
        >>> async def handle_captation_completed(event: Event):
        ...     print(f"Captation completed for store: {event.data['store_id']}")
        >>>
        >>> await bus.publish(Event(
        ...     name="captation.completed",
        ...     data={"store_id": "store-123"},
        ...     source_service="app-service"
        ... ))
    """

    def __init__(self, service_name: str):
        """
        Initialize event bus.

        Args:
            service_name: Name of this service (for logging and tracing)
        """
        self.service_name = service_name
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._logger: Optional[Any] = None

    def set_logger(self, logger: Any) -> None:
        """
        Set logger for event bus.

        Args:
            logger: Logger instance (structlog or standard logging)
        """
        self._logger = logger

    def subscribe(self, event_name: str, handler: Optional[EventHandler] = None):
        """
        Subscribe to an event.

        Can be used as a decorator or called directly.

        Args:
            event_name: Name of the event to subscribe to
            handler: Optional handler function

        Returns:
            Decorator if handler is None, otherwise None

        Example as decorator:
            >>> @bus.subscribe("captation.started")
            >>> async def on_captation_started(event: Event):
            ...     pass

        Example direct call:
            >>> async def my_handler(event: Event):
            ...     pass
            >>> bus.subscribe("captation.started", my_handler)
        """
        def decorator(func: EventHandler) -> EventHandler:
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(func)

            if self._logger:
                self._logger.debug(
                    "event_handler_registered",
                    event_name=event_name,
                    handler=func.__name__
                )

            return func

        if handler is None:
            return decorator
        else:
            return decorator(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from an event.

        Args:
            event_name: Name of the event
            handler: Handler function to remove
        """
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
                if self._logger:
                    self._logger.debug(
                        "event_handler_unregistered",
                        event_name=event_name,
                        handler=handler.__name__
                    )
            except ValueError:
                pass  # Handler not in list

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Handlers are executed concurrently. If a handler fails, it doesn't
        affect other handlers.

        Args:
            event: Event to publish

        Example:
            >>> await bus.publish(Event(
            ...     name="captation.completed",
            ...     data={"store_id": "store-123", "duration": 45.2},
            ...     source_service="app-service"
            ... ))
        """
        handlers = self._handlers.get(event.name, [])

        if not handlers:
            if self._logger:
                self._logger.debug(
                    "event_published_no_handlers",
                    event_name=event.name,
                    source=event.source_service
                )
            return

        if self._logger:
            self._logger.info(
                "event_published",
                event_name=event.name,
                source=event.source_service,
                handlers_count=len(handlers),
                correlation_id=event.correlation_id
            )

        # Execute all handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._execute_handler(event, handler))
            tasks.append(task)

        # Wait for all handlers to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(self, event: Event, handler: EventHandler) -> None:
        """
        Execute a single event handler with error handling.

        Args:
            event: Event to handle
            handler: Handler function
        """
        try:
            await handler(event)

            if self._logger:
                self._logger.debug(
                    "event_handler_success",
                    event_name=event.name,
                    handler=handler.__name__
                )

        except Exception as e:
            if self._logger:
                self._logger.error(
                    "event_handler_failed",
                    event_name=event.name,
                    handler=handler.__name__,
                    error=str(e),
                    exc_info=True
                )

    def list_subscriptions(self) -> Dict[str, List[str]]:
        """
        List all event subscriptions.

        Returns:
            Dict mapping event names to list of handler names
        """
        return {
            event_name: [handler.__name__ for handler in handlers]
            for event_name, handlers in self._handlers.items()
        }


# Global event bus instances (one per service)
_event_buses: Dict[str, EventBus] = {}


def get_event_bus(service_name: str) -> EventBus:
    """
    Get or create event bus for a service.

    Args:
        service_name: Name of the service

    Returns:
        EventBus instance for this service

    Example:
        >>> bus = get_event_bus("app-service")
        >>> await bus.publish(Event(...))
    """
    if service_name not in _event_buses:
        _event_buses[service_name] = EventBus(service_name)

    return _event_buses[service_name]


# Event data helpers

def create_progress_event(
    event_name: str,
    session_id: str,
    progress_percentage: float,
    source_service: str,
    **extra_data
) -> Event:
    """
    Create a standardized progress event.

    Args:
        event_name: Name of the event
        session_id: Session identifier
        progress_percentage: Progress (0-100)
        source_service: Source service name
        **extra_data: Additional data fields

    Returns:
        Event instance
    """
    data = {
        "session_id": session_id,
        "progress_percentage": progress_percentage,
        **extra_data
    }

    return Event(
        name=event_name,
        data=data,
        source_service=source_service
    )


def create_completion_event(
    event_name: str,
    session_id: str,
    source_service: str,
    duration_seconds: Optional[float] = None,
    **extra_data
) -> Event:
    """
    Create a standardized completion event.

    Args:
        event_name: Name of the event
        session_id: Session identifier
        source_service: Source service name
        duration_seconds: Processing duration
        **extra_data: Additional data fields

    Returns:
        Event instance
    """
    data = {
        "session_id": session_id,
        **extra_data
    }

    if duration_seconds is not None:
        data["duration_seconds"] = duration_seconds

    return Event(
        name=event_name,
        data=data,
        source_service=source_service
    )


def create_error_event(
    event_name: str,
    session_id: str,
    error_message: str,
    source_service: str,
    **extra_data
) -> Event:
    """
    Create a standardized error event.

    Args:
        event_name: Name of the event
        session_id: Session identifier
        error_message: Error description
        source_service: Source service name
        **extra_data: Additional data fields

    Returns:
        Event instance
    """
    data = {
        "session_id": session_id,
        "error_message": error_message,
        **extra_data
    }

    return Event(
        name=event_name,
        data=data,
        source_service=source_service,
        priority=EventPriority.HIGH
    )
