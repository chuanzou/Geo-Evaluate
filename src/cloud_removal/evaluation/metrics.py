from __future__ import annotations

import numpy as np


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute a compact global SSIM over RGB channels."""
    pred = prediction[:3].astype(np.float64)
    true = target[:3].astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2

    channel_scores = []
    for pred_channel, true_channel in zip(pred, true, strict=True):
        mu_pred = pred_channel.mean()
        mu_true = true_channel.mean()
        var_pred = pred_channel.var()
        var_true = true_channel.var()
        covariance = ((pred_channel - mu_pred) * (true_channel - mu_true)).mean()
        numerator = (2 * mu_pred * mu_true + c1) * (2 * covariance + c2)
        denominator = (mu_pred**2 + mu_true**2 + c1) * (var_pred + var_true + c2)
        channel_scores.append(numerator / denominator)
    return float(np.mean(channel_scores))


def l1(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error across all pixels and channels."""
    return float(np.mean(np.abs(prediction - target)))


def masked_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error restricted to the cloud (mask == 1) region.

    Returns ``0.0`` when the mask is empty to avoid division-by-zero; callers
    that want to distinguish "no clouds" from "perfect reconstruction" should
    inspect ``coverage`` in parallel.
    """
    mask2d = mask.squeeze()
    denom = max(float(mask2d.sum()) * prediction.shape[0], 1.0)
    error = np.abs(prediction - target) * mask2d[None, ...]
    return float(error.sum() / denom)


def ndvi(image: np.ndarray, red_index: int = 0, nir_index: int = 3) -> np.ndarray:
    red = image[red_index]
    nir = image[nir_index]
    return (nir - red) / (nir + red + 1e-6)


def ndvi_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.abs(ndvi(prediction) - ndvi(target))
    if mask is not None:
        mask2d = mask.squeeze()
        denom = max(float(mask2d.sum()), 1.0)
        return float((error * mask2d).sum() / denom)
    return float(error.mean())


def evaluate_prediction(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    metrics = {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "l1": l1(prediction, target),
        "ndvi_mae": ndvi_mae(prediction, target, mask),
    }
    if mask is not None:
        metrics["l1_cloud_region"] = masked_l1(prediction, target, mask)
    return metrics
