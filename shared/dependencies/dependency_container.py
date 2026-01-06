"""
Simple dependency injection container.
"""

from typing import TypeVar, Optional, Any

from .service_factory import ServiceFactory
from .lazy_service_factory import LazyServiceFactory

T = TypeVar('T')


class DependencyContainer:
    """
    Simple dependency injection container.

    This container manages service lifecycles and resolves dependencies
    in a type-safe manner.

    Example:
        >>> container = DependencyContainer()
        >>> container.register("database", DatabaseService, url="postgres://...")
        >>> container.register("cache", CacheService, redis_url="redis://...")
        >>>
        >>> db = container.resolve("database")
        >>> cache = container.resolve("cache")
    """

    def __init__(self):
        """Initialize empty container."""
        self._factories: dict[str, ServiceFactory] = {}
        self._instances: dict[str, Any] = {}

    def register(
        self,
        name: str,
        service_class: type[T],
        singleton: bool = True,
        **kwargs
    ) -> None:
        """
        Register a service in the container.

        Args:
            name: Service name/identifier
            service_class: Class to instantiate
            singleton: If True, use singleton pattern. If False, create new instance each time
            **kwargs: Constructor arguments
        """
        if singleton:
            factory = ServiceFactory(service_class, **kwargs)
        else:
            factory = LazyServiceFactory(service_class, **kwargs)

        self._factories[name] = factory

    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register an existing instance.

        Args:
            name: Service name/identifier
            instance: Pre-created instance
        """
        self._instances[name] = instance

    def resolve(self, name: str) -> Any:
        """
        Resolve a service by name.

        Args:
            name: Service name/identifier

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
        """
        # Check for pre-registered instances first
        if name in self._instances:
            return self._instances[name]

        # Check for factories
        if name in self._factories:
            return self._factories[name].get_instance()

        raise KeyError(f"Service '{name}' not registered in container")

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset cached instances.

        Args:
            name: Specific service to reset, or None to reset all
        """
        if name:
            if name in self._factories:
                self._factories[name].reset()
            if name in self._instances:
                del self._instances[name]
        else:
            # Reset all factories
            for factory in self._factories.values():
                if isinstance(factory, ServiceFactory):
                    factory.reset()
            # Clear all instances
            self._instances.clear()

    def list_services(self) -> list[str]:
        """
        List all registered service names.

        Returns:
            List of service names
        """
        return list(set(self._factories.keys()) | set(self._instances.keys()))

