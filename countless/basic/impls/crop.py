import torch

from countless.basic import ops, targets
from countless.decorators import impl


def _crop_spatial(tensor: torch.Tensor, crop: ops.Crop) -> torch.Tensor:
    x, y = crop.xy
    w, h = crop.wh
    return tensor[..., y : y + h, x : x + w]


@impl()
def crop_image(operation: ops.Crop, target: targets.Image) -> targets.Image:
    return targets.Image(image=_crop_spatial(target.image, operation))


@impl()
def crop_spatial_map(
    operation: ops.Crop, target: targets.SpatialMap
) -> targets.SpatialMap:
    return targets.SpatialMap(data=_crop_spatial(target.data, operation))


@impl()
def crop_exact_spatial_map(
    operation: ops.Crop, target: targets.ExactSpatialMap
) -> targets.ExactSpatialMap:
    return targets.ExactSpatialMap(data=_crop_spatial(target.data, operation))


@impl()
def crop_camera_matrix(
    operation: ops.Crop, target: targets.CameraMatrix
) -> targets.CameraMatrix:
    newmat = target.matrix.clone()
    newmat[:2, 2] -= operation.xy
    return targets.CameraMatrix(matrix=newmat)


@impl()
def crop_matrices_3d(
    operation: ops.Crop, target: targets.Matrices3D
) -> targets.Matrices3D:
    # No transformation needed
    return target


@impl()
def crop_matrices_2d(
    operation: ops.Crop, target: targets.Matrices2D
) -> targets.Matrices2D:
    newmat = target.matrices.clone()
    newmat[..., :2, 2] -= operation.xy
    return targets.Matrices2D(matrices=newmat)
