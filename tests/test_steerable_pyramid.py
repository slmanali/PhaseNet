import pytest
import torch

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch


@pytest.mark.parametrize("pyr_type", (0, 1))
def test_rectangular_image_builds_and_reconstructs(pyr_type):
    image = torch.rand(2, 1, 32, 48, dtype=torch.float32)
    pyramid = SCFpyr_PyTorch(height=3, nbands=4, scale_factor=2)

    coefficients = pyramid.build(image, pyr_type=pyr_type)
    reconstruction = pyramid.reconstruct(coefficients, pyr_type=pyr_type)

    assert reconstruction.shape == image[:, 0].shape
    assert torch.isfinite(reconstruction).all()
