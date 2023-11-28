from __future__ import annotations

import abc
import collections.abc as c
import typing as t
from dataclasses import dataclass

import torch


class Named(abc.ABC):
    """A named class. This is used to register operations, targets and implementations
    without relying on the __name__ attribute of the class, but rather on an explicit
    name() method.
    """

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the class. Can be different from __name__ attribute if needed.

        Returns:
            str: A string representing the name of the class, should be unique.
        """
        pass


A = t.TypeVar("A")
"""Generic type variable for anything."""
B = t.TypeVar("B")
"""Generic type variable for anything."""

# Type alias for Named types
N = t.TypeVar("N", bound=Named)


class RegistryMeta(abc.ABCMeta, t.Generic[N]):
    """Metaclass to register Named classes concrete implementations, and to retrieve
    them by name.
    """

    _registry: dict[str, N] = {}

    def __new__(cls, name, bases, attrs):
        new_cls: N = super().__new__(cls, name, bases, attrs)  # type: ignore
        if all(
            not getattr(getattr(new_cls, name, None), "__isabstractmethod__", False)
            for name in dir(new_cls)
        ):
            cls._registry[new_cls.name()] = new_cls
        return new_cls

    @classmethod
    def get(cls, name: str) -> N:
        """Get a class by name, if it exists.

        Args:
            name (str): The name of the class to retrieve.

        Returns:
            N: The class with the given name.
        """
        return cls._registry[name]


class TypedMap(c.Mapping[str, A]):
    """A `Mapping` that contains objects of different types. In addition to the usual
    mapping methods, it also provides a `by_type` method that returns a mapping of all
    the objects of a given type. This is useful to avoid explicit type casting when
    retrieving objects from the map.
    """

    def __init__(self, data: c.Mapping[str, A]) -> None:
        """Constructor. Simply wraps an existing mapping.

        Args:
            data (c.Mapping[str, A]): The mapping to wrap.
        """
        self._data = data

        _by_type: dict[type[A], dict[str, A]] = {}
        for k, v in data.items():
            _by_type.setdefault(type(v), {})[k] = v

        self._by_type: t.Mapping[A, t.Mapping[str, A]] = _by_type  # type: ignore

    def by_type(self, cls: type[B]) -> c.Mapping[str, B]:
        """Get a mapping of all the objects of a given type, if any.

        Args:
            cls (type[B]): The type of the objects to retrieve.

        Returns:
            c.Mapping[str, B]: A mapping of all the objects of the given type.
        """
        return self._by_type.get(cls, {})  # type: ignore

    def __getitem__(self, key: str) -> A:
        return self._data[key]

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


# Stub class for type checking
class Operation(Named):  # type: ignore
    pass


# Stub class for type checking
class Target(Named):  # type: ignore
    pass


# Stub class for type checking
class Implementation(Named):  # type: ignore
    pass


class OperationMeta(RegistryMeta[Operation]):
    """Metaclass for `Operation` hierarchy."""

    _registry: dict[str, type[Operation]] = {}


class TargetMeta(RegistryMeta[Target]):
    """Metaclass for `Target` hierarchy."""

    _registry: dict[str, type[Target]] = {}


class ImplementationMeta(RegistryMeta[Implementation]):
    """Metaclass for `Implementation` hierarchy."""

    _registry: dict[str, type[Implementation]] = {}


class Operation(Named, metaclass=OperationMeta):
    """Base class for operations. Operations are classes that represent a computation
    (i.e. a function) to be performed on a target.

    Operation classes should only define the parameters required to perform the
    operation, and should not contain any logic. The logic is instead implemented in
    `Implementation` classes, which are automatically registered and retrieved when
    needed.

    The main advantage of this approach is that it allows to scale the number of both
    operations and targets without having to modify the code of the operations
    themselves. This is crucial because both operations and targets are unknown at
    design time, and are instead defined by the user.
    """

    def apply_single(self, target: T) -> T:
        """Apply this operation to a target. This method is just a shortcut for
        `apply_single(op, target)`. The implementation is automatically looked up based
        on the operation and target types, inferred from the arguments.

        Args:
            target (T): The target to apply the operation to.

        Returns:
            T: The result of the operation. Guaranteed to be of the same type as the
            input target.
        """
        return apply_single(self, target)

    def apply(self, **targets: Target) -> TypedMap:
        """Apply this operation to multiple targets. This method is just a shortcut for
        `apply(op, **targets)`. The implementation is automatically looked up based on
        the operation and target types, inferred from the arguments.

        Args:
            **targets (Target): The targets to apply the operation to.

        Returns:
            TypedMap: A typed map containing the results of the operation. The keys are
            the same as the keys of the input targets, and the values are the results
            of the operation. The types of the values are the same as the types of the
            input targets.
        """
        return apply(self, **targets)


class Target(Named, metaclass=TargetMeta):
    """Base class for targets. Targets are classes that represent the objects that
    operations are applied to. Concrete targets should define the attributes required
    by the operations that can be applied to them, usually a tensor or a collection of
    tensors.

    Targets should not contain any operation logic, which is instead implemented in
    `Implementation` classes.
    """

    pass


O = t.TypeVar("O", bound=Operation)
"""Generic type variable for operations."""
T = t.TypeVar("T", bound=Target)
"""Generic type variable for targets."""


class Implementation(Named, t.Generic[O, T], metaclass=ImplementationMeta):
    """Base class for implementations. Implementations are classes that contain the
    logic to perform an operation on a target. Implementations are automatically
    registered and retrieved when needed.

    Simply defining an implementation class is enough to register it. When applying an
    operation to a target, the implementation is automatically retrieved based on the
    operation and target types, inferred from the arguments, and the `impl` method is
    called.

    To define an implementation, simply subclass this class and implement:
        - `operation`: the name of the operation this implementation is for
        - `target`: the name of the target this implementation is for
        - `impl`: the implementation of the operation on the target
    """

    @classmethod
    @abc.abstractmethod
    def operation(cls) -> str:
        """The name of the operation this implementation is for.

        Returns:
            str: The name of the operation.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def target(cls) -> str:
        """The name of the target this implementation is for.

        Returns:
            str: The name of the target.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def impl(cls, operation: O, target: T) -> T:
        """How this type of operation is implemented on this type of target.

        Args:
            operation (O): The operation to perform.
            target (T): The target to perform the operation on.

        Returns:
            T: The result of the operation, of the same type as the input target.
        """
        pass

    @classmethod
    def name(cls) -> str:
        """The name of this implementation. The name is automatically inferred from the
        `operation` and `target` class methods.

        Returns:
            str: The name of this implementation.
        """
        return cls.name_from(cls.operation(), cls.target())

    @staticmethod
    def name_from(operation: str, target: str) -> str:
        """Static method to generate the name of an implementation from the operation
        and target names.

        Args:
            operation (str): The name of the operation.
            target (str): The name of the target.

        Returns:
            str: The name of the implementation.
        """
        return f"{operation}_{target}"


def apply_single(operation, target: A) -> A:
    """Functional version of `Operation.apply_single`. This function will automatically
    look up the implementation based on the operation and target types, inferred from
    the arguments, and call the `impl` method.

    Args:
        operation: The operation to perform.
        target (A): The target to perform the operation on.

    Returns:
        A: The result of the operation, of the same type as the input target.
    """
    impl_name = Implementation.name_from(operation.name(), target.name())  # type: ignore
    return Implementation.get(impl_name).impl(operation, target)  # type: ignore


def apply(operation, **targets: t.Any) -> TypedMap[Target]:
    """Functional version of `Operation.apply`. This function will automatically look
    up the implementation based on the operation and target types, inferred from the
    arguments, and call the `impl` method.

    Args:
        operation: The operation to perform.

    Returns:
        TypedMap[Target]: A typed map containing the results of the operation. The keys
        are the same as the keys of the input targets, and the values are the results
        of the operation. The types of the values are the same as the types of the
        input targets.
    """

    return TypedMap(
        {name: apply_single(operation, target) for name, target in targets.items()}
    )


def _make_type(name: str, Base: type[t.Any]) -> t.Callable[[type[A]], type[A]]:
    def decorator(cls: type[A]) -> type[A]:
        name_ = name or cls.__name__

        class _Op(Base, cls):
            @classmethod
            def name(cls) -> str:
                return name_

        return _Op

    return decorator


def operation(name: str = "") -> t.Callable[[type[A]], type[A]]:
    """Transform any class into an `Operation`, for the lazy ones. Just decorate any
    class with `@operation()` and you're done. You can also pass a name to the
    decorator to override the default name, which is the name of the class.

    Args:
        name (str, optional): The operation name, if empty, class name is automatically
        used. Defaults to "".

    Returns:
        t.Callable[[type[A]], type[A]]: The decorated class.
    """
    return _make_type(name, Operation)


def target(name: str = "") -> t.Callable[[type[A]], type[A]]:
    """Transform any class into a `Target`, for the lazy ones. Just decorate any class
    with `@target()` and you're done. You can also pass a name to the decorator to
    override the default name, which is the name of the class.

    Args:
        name (str, optional): The target name, if empty, class name is automatically
        used. Defaults to "".

    Returns:
        t.Callable[[type[A]], type[A]]: The decorated class.
    """
    return _make_type(name, Target)


def impl(operation: str = "", target: str = ""):
    """Transform any function into an `Implementation`, for the lazy ones. Just decorate
    any function with `@impl()` and you're done. You can also pass the operation and
    target names to the decorator to override the default names, which are inferred
    from the function annotations.

    The only requirements for the function are:
        - The first argument must be the operation and be named `operation`
        - The second argument must be the target and be named `target`
        - The return type must be the same as the target type

    Signature:
        def fn(operation: A, target: B) -> B:
            ...

    Args:
        operation (str, optional): The operation name, if the operation argument is
        not annotated. If empty, it will be automatically inferred. Defaults to "".
        target (str, optional): The target name, if the target argument is not
        annotated. If empty, it will be automatically inferred. Defaults to "".
    """

    def decorator(fn: t.Callable[[A, B], B]) -> t.Callable[[A, B], B]:
        nonlocal operation, target
        if operation == "":
            annotation = fn.__annotations__["operation"]
            assert issubclass(annotation, Operation)
            operation = annotation.name()
        if target == "":
            annotation = fn.__annotations__["target"]
            assert issubclass(annotation, Target)
            target = annotation.name()

        class _Impl(Implementation[A, B]):  # type: ignore
            @classmethod
            def operation(cls) -> str:
                return operation

            @classmethod
            def target(cls) -> str:
                return target  # type: ignore

            @classmethod
            def impl(cls, operation: A, target: B) -> B:
                return fn(operation, target)

        return fn

    return decorator
