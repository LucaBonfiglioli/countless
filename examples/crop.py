import torch


def single_op_single_tgt():
    from countless.basic.ops import Crop
    from countless.basic.targets import Image

    image = Image(torch.rand(3, 100, 100))
    crop = Crop(xy=torch.tensor([10, 20]), wh=torch.tensor([80, 80]))
    out = crop.apply_single(image)

    assert torch.allclose(out.image, image.image[..., 20 : 20 + 80, 10 : 10 + 80])


def single_op_multi_tgt():
    from countless.basic.ops import Crop
    from countless.batched.targets import BatchedImage

    image = BatchedImage(torch.rand(4, 3, 100, 100))
    crop = Crop(xy=torch.tensor([10, 20]), wh=torch.tensor([80, 80]))
    out = crop.apply_single(image)

    assert torch.allclose(out.image[0], image.image[0, ..., 20 : 20 + 80, 10 : 10 + 80])


def multi_op_single_tgt():
    from countless.basic.targets import Image
    from countless.batched.ops import BatchedCrop

    image = Image(torch.rand(3, 100, 100))
    crop = BatchedCrop(xy=torch.tensor([[10, 20]]), wh=torch.tensor([80, 80]))
    out = crop.apply_single(image)

    assert torch.allclose(out.image, image.image[..., 20 : 20 + 80, 10 : 10 + 80])


def multi_op_multi_tgt():
    from countless.batched.ops import BatchedCrop
    from countless.batched.targets import BatchedImage

    image = BatchedImage(torch.rand(4, 3, 100, 100))
    crop = BatchedCrop(
        xy=torch.tensor([[10, 20], [10, 15], [15, 10], [15, 15]]),
        wh=torch.tensor([80, 80]),
    )
    out = crop.apply_single(image)

    assert torch.allclose(out.image[0], image.image[0, ..., 20 : 20 + 80, 10 : 10 + 80])
    assert torch.allclose(out.image[1], image.image[1, ..., 15 : 15 + 80, 10 : 10 + 80])
    assert torch.allclose(out.image[2], image.image[2, ..., 10 : 10 + 80, 15 : 15 + 80])
    assert torch.allclose(out.image[3], image.image[3, ..., 15 : 15 + 80, 15 : 15 + 80])


if __name__ == "__main__":
    single_op_single_tgt()
    single_op_multi_tgt()
    multi_op_single_tgt()
    multi_op_multi_tgt()
