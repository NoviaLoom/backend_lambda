"""
Generic factory for creating and caching service instances.
"""

from typing import TypeVar, Generic, Optional

T = TypeVar('T')


class ServiceFactory(Generic[T]):
    """
    Generic factory for creating and caching service instances.

    This factory ensures that services are created once and reused,
    improving performance and maintaining state consistency.

    Example:
        >>> class MyService:
        ...     def __init__(self, config: str):
        ...         self.config = config
        >>>
        >>> factory = ServiceFactory(MyService, config="production")
        >>> instance1 = factory.get_instance()
        >>> instance2 = factory.get_instance()
        >>> assert instance1 is instance2  # Same instance
    """

    def __init__(self, service_class: type[T], **kwargs):
        """
        Initialize service factory.

        Args:
            service_class: Class to instantiate
            **kwargs: Constructor arguments for the service
        """
        self.service_class = service_class
        self.kwargs = kwargs
        self._instance: Optional[T] = None

    def get_instance(self) -> T:
        """
        Get or create service instance (singleton pattern).

        Returns:
            Service instance
        """
        if self._instance is None:
            self._instance = self.service_class(**self.kwargs)
        return self._instance

    def reset(self) -> None:
        """Reset the cached instance (useful for testing)."""
        self._instance = None

