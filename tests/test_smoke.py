from __future__ import annotations

import numpy as np
import torch

from cloud_removal.data.synthetic_clouds import apply_cloud, random_sentinel_like_patch
from cloud_removal.evaluation.metrics import evaluate_prediction
from cloud_removal.models.baseline import multi_temporal_composite
from cloud_removal.models.unet import UNetGenerator


def test_synthetic_cloud_and_metrics() -> None:
    rng = np.random.default_rng(0)
    image = random_sentinel_like_patch(32, rng)
    sample = apply_cloud(image, 0.3, rng)
    assert sample.cloudy.shape == image.shape
    assert sample.mask.shape == (32, 32)
    metrics = evaluate_prediction(image, image, sample.mask)
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] > 0.99
    assert metrics["ndvi_mae"] == 0.0


def test_models_forward() -> None:
    model = UNetGenerator(in_channels=5, out_channels=4, base_channels=8)
    x = torch.rand(2, 5, 32, 32)
    y = model(x)
    assert y.shape == (2, 4, 32, 32)


def test_temporal_composite() -> None:
    cloudy_stack = torch.zeros(2, 4, 8, 8)
    cloudy_stack[1] = 0.5
    masks = torch.ones(2, 1, 8, 8)
    masks[1] = 0
    composite = multi_temporal_composite(cloudy_stack, masks)
    assert torch.allclose(composite, torch.full((4, 8, 8), 0.5))
