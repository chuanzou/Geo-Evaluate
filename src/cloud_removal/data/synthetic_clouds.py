from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class CloudSample:
    cloudy: np.ndarray
    mask: np.ndarray
    coverage: float


def make_cloud_mask(
    height: int,
    width: int,
    coverage: float,
    rng: np.random.Generator,
    smooth_sigma: float | None = None,
) -> np.ndarray:
    """Create an organic binary cloud mask with approximately target coverage."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be between 0 and 1")

    smooth_sigma = smooth_sigma or max(height, width) / 18
    noise = rng.normal(size=(height, width)).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=smooth_sigma)
    threshold = np.quantile(smooth, 1.0 - coverage)
    return (smooth >= threshold).astype(np.float32)


def apply_cloud(
    image: np.ndarray,
    coverage: float,
    rng: np.random.Generator,
    cloud_value: float = 1.0,
) -> CloudSample:
    """Overlay synthetic white clouds on a normalized CxHxW image."""
    if image.ndim != 3:
        raise ValueError("image must have shape CxHxW")

    _, height, width = image.shape
    mask = make_cloud_mask(height, width, coverage, rng)
    cloud_layer = np.full_like(image, cloud_value, dtype=np.float32)
    cloudy = image * (1.0 - mask[None, ...]) + cloud_layer * mask[None, ...]
    return CloudSample(cloudy=np.clip(cloudy, 0.0, 1.0), mask=mask, coverage=float(mask.mean()))


def random_sentinel_like_patch(
    image_size: int,
    rng: np.random.Generator,
    channels: int = 4,
) -> np.ndarray:
    """Generate a simple Sentinel-2-like patch for smoke tests and demos."""
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    xx = xx / max(image_size - 1, 1)
    yy = yy / max(image_size - 1, 1)

    base = 0.25 + 0.35 * xx + 0.20 * yy
    fields = ((np.sin(xx * rng.uniform(12, 24)) + np.cos(yy * rng.uniform(10, 20))) > 0).astype(np.float32)
    urban = (((xx * image_size).astype(int) % rng.integers(10, 18)) < 2).astype(np.float32)
    vegetation = np.clip(base + 0.30 * fields - 0.12 * urban, 0.0, 1.0)

    red = np.clip(vegetation * 0.75 + rng.normal(0, 0.03, (image_size, image_size)), 0.0, 1.0)
    green = np.clip(vegetation * 0.88 + rng.normal(0, 0.03, (image_size, image_size)), 0.0, 1.0)
    blue = np.clip(vegetation * 0.62 + rng.normal(0, 0.03, (image_size, image_size)), 0.0, 1.0)
    nir = np.clip(vegetation * 1.15 + 0.15 * fields + rng.normal(0, 0.03, (image_size, image_size)), 0.0, 1.0)
    patch = np.stack([red, green, blue, nir], axis=0).astype(np.float32)
    return patch[:channels]
