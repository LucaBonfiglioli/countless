from dataclasses import dataclass

import torch

from countless.core import Operation


@dataclass
class NoOp(Operation):
    @classmethod
    def name(cls) -> str:
        return "noop"


@dataclass
class Crop(Operation):
    xy: torch.Tensor
    wh: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "crop"


# class Pad(Operation):
#     pass


# class IsotropicResize(Operation):
#     pass


# class Resize(Operation):
#     pass


# class Blend(Operation):
#     pass


# class LinearFilter(Operation):
#     pass


# class SepFilter(Operation):
#     pass


# class AffineWarp(Operation):
#     pass


# class Warp(Operation):
#     pass


# class Rotate90(Operation):
#     pass


# class Flip(Operation):
#     pass


# class ChannelAffine(Operation):
#     pass


# class ColorSpaceConversion(Operation):
#
