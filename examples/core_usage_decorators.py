from dataclasses import dataclass

import torch

from countless.decorators import apply, impl, operation, target


@target()
@dataclass
class MyImage:
    image: torch.Tensor


@target()
@dataclass
class MyPoints:
    xy: torch.Tensor


@operation()
@dataclass
class MyCrop:
    x: int
    y: int
    w: int
    h: int


@impl()
def crop_image(operation: MyCrop, target: MyImage) -> MyImage:
    x, y, w, h = operation.x, operation.y, operation.w, operation.h
    return MyImage(target.image[..., x : x + w, y : y + h])


@impl()
def crop_point(operation: MyCrop, target: MyPoints) -> MyPoints:
    dxy = torch.tensor([operation.x, operation.y], device=target.xy.device)
    return MyPoints(target.xy - dxy.unsqueeze(0))


image = MyImage(torch.rand(3, 100, 100))
points = MyPoints(torch.tensor([[50, 50], [20, 30]]))
op = MyCrop(10, 20, 80, 80)
out = apply(op, image=image, points=points)
image = out.by_type(MyImage)["image"].image
points = out.by_type(MyPoints)["points"].xy

print("Images:", image.shape)
print("Points:", points)
