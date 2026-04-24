from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_removal.data.synthetic_clouds import apply_cloud
from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.evaluation.metrics import evaluate_prediction
from cloud_removal.evaluation.plots import plot_metric_curves
from cloud_removal.models.baseline import multi_temporal_composite
from cloud_removal.models.diffusion import InpaintingDiffusion
from cloud_removal.models.unet import UNetGenerator
from cloud_removal.utils.config import ensure_dir


# The metric keys we aggregate per coverage bin.  Anything else (e.g. sample
# id, coverage) is passed through to the per-sample CSV but not averaged.
_METRIC_KEYS: tuple[str, ...] = ("psnr", "ssim", "l1", "l1_cloud_region", "ndvi_mae")


def load_gan(config: dict, device: torch.device) -> UNetGenerator:
    model = UNetGenerator(in_channels=5, out_channels=4).to(device)
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    checkpoint = checkpoint_dir / "gan_generator_best.pt"
    if not checkpoint.exists():
        checkpoint = checkpoint_dir / "gan_generator.pt"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def load_diffusion(config: dict, device: torch.device) -> InpaintingDiffusion:
    model = InpaintingDiffusion(timesteps=int(config["training"]["diffusion_timesteps"])).to(device)
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    checkpoint = checkpoint_dir / "diffusion_best.pt"
    if not checkpoint.exists():
        checkpoint = checkpoint_dir / "diffusion.pt"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def _summarise_by_coverage(
    rows: list[dict[str, float | str]],
    coverage_levels: Iterable[float],
    method: str,
) -> list[dict[str, float | str | int]]:
    """Aggregate per-sample rows into one row per coverage bin.

    ``coverage_levels`` comes from ``configs/default.yaml::data.cloud_coverages``
    so the order in the summary matches the order requested by the user
    (e.g. 5% -> 10% -> 30% -> 50% -> 70%), even for bins that contain zero
    samples (reported as NaN).
    """
    grouped: dict[float, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["coverage_bin"])].append(row)

    summary: list[dict[str, float | str | int]] = []
    for coverage in coverage_levels:
        bucket = grouped.get(float(coverage), [])
        entry: dict[str, float | str | int] = {
            "method": method,
            "coverage_bin": float(coverage),
            "num_samples": len(bucket),
        }
        for key in _METRIC_KEYS:
            if not bucket:
                entry[key] = float("nan")
                continue
            # Filter non-finite values (e.g. PSNR == inf for perfect
            # reconstructions) so the average isn't hijacked by a single
            # edge-case sample.
            values = [float(row[key]) for row in bucket if key in row and np.isfinite(float(row[key]))]
            entry[key] = float(np.mean(values)) if values else float("nan")
        summary.append(entry)
    return summary


def evaluate_method(config: dict, method: str, device: torch.device) -> Path:
    dataset = CloudRemovalDataset(config["data"]["root"], "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = None
    if method == "gan":
        model = load_gan(config, device)
    elif method == "diffusion":
        model = load_diffusion(config, device)
    elif method != "temporal":
        raise ValueError(f"Unknown method: {method}")

    coverage_levels: list[float] = [float(v) for v in config["data"]["cloud_coverages"]]

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {method}"):
            cloudy = batch["cloudy"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            if method == "gan":
                prediction = model(batch["input"].to(device))
                prediction = prediction * mask + cloudy * (1.0 - mask)
            elif method == "diffusion":
                prediction = model.inpaint(cloudy, mask, steps=25)
            else:
                target_np = target.squeeze(0).cpu().numpy()
                coverage = float(batch["coverage"].item())
                digest = hashlib.sha256(batch["id"][0].encode("utf-8")).hexdigest()
                rng_seed = int(digest[:8], 16)
                rng = np.random.default_rng(rng_seed)
                temporal_images = [cloudy.squeeze(0).cpu()]
                temporal_masks = [mask.squeeze(0).cpu()]
                for _ in range(int(config["data"]["temporal_neighbors"])):
                    observation = apply_cloud(target_np, coverage, rng)
                    temporal_images.append(torch.from_numpy(observation.cloudy))
                    temporal_masks.append(torch.from_numpy(observation.mask[None, ...]))
                stack = torch.stack(temporal_images).to(device)
                masks = torch.stack(temporal_masks).to(device)
                prediction = multi_temporal_composite(stack, masks, fallback=cloudy.squeeze(0)).unsqueeze(0)

            pred_np = prediction.squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()
            metrics = evaluate_prediction(pred_np, target_np, mask_np)
            coverage = float(batch["coverage"].item())
            coverage_bin = min(coverage_levels, key=lambda value: abs(value - coverage))
            rows.append(
                {
                    "id": batch["id"][0],
                    "method": method,
                    "coverage": coverage,
                    "coverage_bin": coverage_bin,
                    **metrics,
                }
            )

    metrics_dir = ensure_dir(config["outputs"]["metrics_dir"])

    # ---- per-sample csv (unchanged contract) ----
    output_path = metrics_dir / f"{method}_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- NEW: per-coverage summary csv ----
    summary = _summarise_by_coverage(rows, coverage_levels, method)
    summary_path = metrics_dir / f"{method}_metrics_by_coverage.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Also log a compact table to stdout so the experimenter can compare
    # performance across coverage levels at a glance.
    print(f"\n=== {method} metrics by cloud coverage ===")
    header = "coverage  n    psnr    ssim     l1    l1_cloud  ndvi_mae"
    print(header)
    print("-" * len(header))
    for entry in summary:
        print(
            f"{entry['coverage_bin']:>6.2f}  "
            f"{entry['num_samples']:>3d}  "
            f"{entry['psnr']:>6.2f}  "
            f"{entry['ssim']:>6.4f}  "
            f"{entry['l1']:>6.4f}  "
            f"{entry['l1_cloud_region']:>8.4f}  "
            f"{entry['ndvi_mae']:>8.4f}"
        )

    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    plot_metric_curves(output_path, figure_dir / f"{method}_curves.png")
    return output_path
