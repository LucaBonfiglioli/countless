from __future__ import annotations

import typing as t

from countless.core import Implementation, Operation, Target, TypedMap

A = t.TypeVar("A")
"""Generic type variable for anything."""
B = t.TypeVar("B")
"""Generic type variable for anything."""


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
    op, tgt = t.cast(Operation, operation), t.cast(Target, target)
    return t.cast(A, op.apply_single(tgt))


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

    op, tgts = t.cast(Operation, operation), t.cast(t.Mapping[str, Target], targets)
    return op.apply(**tgts)


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
