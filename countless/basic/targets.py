from dataclasses import dataclass

import torch

from countless.core import Target


@dataclass
class PlainData(Target):
    data: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "plain_data"


@dataclass
class Image(Target):
    image: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "image"


@dataclass
class SpatialMap(Target):
    data: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "spatial_map"


@dataclass
class ExactSpatialMap(Target):
    data: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "exact_spatial_map"


@dataclass
class CameraMatrix(Target):
    matrix: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "camera_matrix"


@dataclass
class Matrices3D(Target):
    matrices: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "matrices_3d"


@dataclass
class Matrices2D(Target):
    matrices: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "matrices_2d"
