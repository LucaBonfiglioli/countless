from __future__ import annotations

import collections.abc as c
import typing as t
import warnings
from abc import ABC, ABCMeta, abstractmethod

import torch


class Named(ABC):
    """A named class. This is used to register operations, targets and implementations
    without relying on the __name__ attribute of the class, but rather on an explicit
    name() method.
    """

    @classmethod
    @abstractmethod
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

N = t.TypeVar("N", bound=Named)
"""Generic type variable for Named."""

ANYTHING = "__anything__"
"""Special value to match anything."""


class Deviced:
    def _deviced_fields(self) -> list[Deviced]:
        return [v for k, v in self.__dict__.items() if isinstance(v, Deviced)]

    def _tensor_fields(self) -> list[torch.Tensor]:
        return [v for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)]

    @property
    def device(self) -> str:
        device = "cpu"
        deviced_fields = self._deviced_fields()
        tensor_fields = self._tensor_fields()
        if len(deviced_fields) > 0:
            device = deviced_fields[0].device
        elif len(tensor_fields) > 0:
            device = tensor_fields[0].device.type

        return device

    @device.setter
    def device(self, device: str) -> None:
        for field in self._deviced_fields():
            field.device = device
        for field in self._tensor_fields():
            field = field.to(device)


D = t.TypeVar("D", bound=Deviced)
"""Generic type variable for Deviced."""


class Batched(ABC, t.Generic[D]):
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def unbatch(self) -> c.Iterable[D]:
        pass

    @classmethod
    def batch(cls, *unbatched: D) -> Batched[D]:
        assert len(unbatched) > 0, "Cannot batch empty list"
        for op in unbatched[1:]:
            op.device = unbatched[0].device

        return cls._batch(*unbatched)

    @classmethod
    @abstractmethod
    def _batch(cls, *unbatched: D) -> Batched[D]:
        pass


class RegistryMeta(ABCMeta, t.Generic[N]):
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
    def get(cls, name: str) -> t.Optional[N]:
        """Get a class by name, if it exists.

        Args:
            name (str): The name of the class to retrieve.

        Returns:
            Optional[N]: The class with the given name.
        """
        return cls._registry.get(name)


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
class Operation(Named, Deviced):  # type: ignore
    pass


# Stub class for type checking
class Target(Named, Deviced):  # type: ignore
    pass


# Stub class for type checking
class Implementation(Named, Deviced):  # type: ignore
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


class Target(Named, Deviced, metaclass=TargetMeta):
    """Base class for targets. Targets are classes that represent the objects that
    operations are applied to. Concrete targets should define the attributes required
    by the operations that can be applied to them, usually a tensor or a collection of
    tensors.

    Targets should not contain any operation logic, which is instead implemented in
    `Implementation` classes.
    """

    pass


T = t.TypeVar("T", bound=Target)
"""Generic type variable for targets."""


class BatchedTarget(Batched[T], Target):
    pass


BT = t.TypeVar("BT", bound=BatchedTarget)


class Operation(Named, Deviced, metaclass=OperationMeta):
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

    def _apply_single_only_target_batched(self, target: BT) -> BT:
        return target.batch(*[self.apply_single(t) for t in target.unbatch()])

    def apply_single(self, target: T) -> T:
        """Apply this operation to a target.

        Args:
            target (T): The target to apply the operation to.

        Returns:
            T: The result of the operation. Guaranteed to be of the same type as the
            input target.
        """
        impl_name = Implementation.name_from(self.name(), target.name())  # type: ignore
        impl = Implementation.get(impl_name)

        if impl is not None:
            return impl.impl(self, target)  # type: ignore

        # Check if there is a generic implementation for this operation
        impl_name = Implementation.name_from(self.name(), ANYTHING)  # type: ignore
        impl = Implementation.get(impl_name)

        if impl is not None:
            return impl.impl(self, target)  # type: ignore

        # Check if there is a generic implementation for this target
        impl_name = Implementation.name_from(ANYTHING, target.name())  # type: ignore
        impl = Implementation.get(impl_name)

        if impl is not None:
            return impl.impl(self, target)  # type: ignore

        # Check if the target is batched and the operation is not
        if isinstance(target, BatchedTarget) and not isinstance(self, BatchedOperation):
            return self._apply_single_only_target_batched(target)  # type: ignore

        raise NotImplementedError(
            f"No implementation found for op {self.name()} on target {target.name()}"  # type: ignore
        )

    def apply(self, **targets: Target) -> TypedMap:
        """Apply this operation to multiple targets.

        Args:
            **targets (Target): The targets to apply the operation to.

        Returns:
            TypedMap: A typed map containing the results of the operation. The keys are
            the same as the keys of the input targets, and the values are the results
            of the operation. The types of the values are the same as the types of the
            input targets.
        """
        return TypedMap(
            {name: self.apply_single(target) for name, target in targets.items()}
        )


O = t.TypeVar("O", bound=Operation)
"""Generic type variable for operations."""


class BatchedOperation(Batched[O], Operation):
    def _default_both_batched(self, target: BT) -> BT:
        warnings.warn(
            f"Using the default implementation for batched operation '{self.name()}' on"
            f" batched target '{target.name()}'. This may be slower than a custom"
            f" batched implementation.",
        )
        assert target.size() == self.size()

        unbatched_t = target.unbatch()
        unbatched_o = self.unbatch()

        unbatched_r = [o.apply_single(t) for o, t in zip(unbatched_o, unbatched_t)]

        return target.batch(*unbatched_r)

    def _default_only_operation_batched(self, target: T) -> T:
        first_op = next(iter(self.unbatch()))
        return first_op.apply_single(target)

    def apply_single(self, target: T) -> T:
        try:
            return super().apply_single(target)
        except NotImplementedError:
            if isinstance(target, BatchedTarget):
                return self._default_both_batched(target)
            else:
                return self._default_only_operation_batched(target)


class Sequential(Operation):
    """A sequential operation that applies a list of operations in order."""

    def __init__(self, *operations: Operation):
        self.operations = operations

    @classmethod
    def name(cls) -> str:
        return "sequential"

    def apply_single(self, target: T) -> T:
        for op in self.operations:
            target = op.apply_single(target)
        return target


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
    @abstractmethod
    def operation(cls) -> str:
        """The name of the operation this implementation is for.

        Returns:
            str: The name of the operation.
        """
        pass

    @classmethod
    @abstractmethod
    def target(cls) -> str:
        """The name of the target this implementation is for.

        Returns:
            str: The name of the target.
        """
        pass

    @classmethod
    @abstractmethod
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
