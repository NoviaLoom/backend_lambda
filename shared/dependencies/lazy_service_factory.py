"""
Lazy factory that creates a new instance on each call.
"""

from typing import TypeVar, Generic

T = TypeVar('T')


class LazyServiceFactory(Generic[T]):
    """
    Lazy factory that creates a new instance on each call.

    Use this when you don't want singleton behavior and need
    a fresh instance each time.

    Example:
        >>> factory = LazyServiceFactory(MyService, config="test")
        >>> instance1 = factory.get_instance()
        >>> instance2 = factory.get_instance()
        >>> assert instance1 is not instance2  # Different instances
    """

    def __init__(self, service_class: type[T], **kwargs):
        """
        Initialize lazy factory.

        Args:
            service_class: Class to instantiate
            **kwargs: Constructor arguments for the service
        """
        self.service_class = service_class
        self.kwargs = kwargs

    def get_instance(self) -> T:
        """
        Create a new service instance.

        Returns:
            New service instance
        """
        return self.service_class(**self.kwargs)

