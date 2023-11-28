from dataclasses import dataclass

import torch

from countless.core import Implementation, Operation, Target


@dataclass
class MyImage(Target):
    image: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "my_image"


@dataclass
class MyPoints(Target):
    xy: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "my_points"


@dataclass
class MyCrop(Operation):
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def name(cls) -> str:
        return "my_crop"


class MyCropImageImpl(Implementation):
    @classmethod
    def operation(cls) -> str:
        return MyCrop.name()

    @classmethod
    def target(cls) -> str:
        return MyImage.name()

    @classmethod
    def impl(cls, operation: MyCrop, target: MyImage) -> MyImage:
        x, y, w, h = operation.x, operation.y, operation.w, operation.h
        return MyImage(target.image[..., x : x + w, y : y + h])


class MyCropPointImpl(Implementation):
    @classmethod
    def operation(cls) -> str:
        return MyCrop.name()

    @classmethod
    def target(cls) -> str:
        return MyPoints.name()

    @classmethod
    def impl(cls, operation: MyCrop, target: MyPoints) -> MyPoints:
        dxy = torch.tensor([operation.x, operation.y], device=target.xy.device)
        return MyPoints(target.xy - dxy.unsqueeze(0))


image = MyImage(torch.rand(3, 100, 100))
points = MyPoints(torch.tensor([[50, 50], [20, 30]]))

op = MyCrop(10, 20, 80, 80)

out = op.apply(image=image, points=points)
image = out.by_type(MyImage)["image"].image
points = out.by_type(MyPoints)["points"].xy

print("Images:", image.shape)
print("Points:", points)
