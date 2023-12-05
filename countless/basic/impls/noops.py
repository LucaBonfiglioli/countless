from countless.basic.ops import NoOp
from countless.basic.targets import PlainData
from countless.core import ANYTHING, Operation, Target
from countless.decorators import impl


@impl(target=ANYTHING)
def noop_anything(operation: NoOp, target: Target) -> Target:
    return target


@impl(operation=ANYTHING)
def anything_plain_data(operation: Operation, target: PlainData) -> PlainData:
    return target
