from __future__ import annotations

import csv
import hashlib
from pathlib import Path

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
            coverage_bin = min(config["data"]["cloud_coverages"], key=lambda value: abs(value - coverage))
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
    output_path = metrics_dir / f"{method}_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure_dir = ensure_dir(config["outputs"]["figure_dir"])
    plot_metric_curves(output_path, figure_dir / f"{method}_curves.png")
    return output_path
