from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.models.diffusion import InpaintingDiffusion
from cloud_removal.utils.config import ensure_dir


def _validation_mse(model: InpaintingDiffusion, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            predicted_noise, noise = model(cloudy, mask, target)
            total_loss += float(mse(predicted_noise, noise).item())
            total_count += noise.numel()
    model.train()
    return total_loss / max(total_count, 1)


def train_diffusion(config: dict, device: torch.device) -> Path:
    data_root = config["data"]["root"]
    train_dataset = CloudRemovalDataset(data_root, "train")
    val_dataset = CloudRemovalDataset(data_root, "val")
    loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = InpaintingDiffusion(timesteps=int(config["training"]["diffusion_timesteps"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    mse = nn.MSELoss()
    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    best_checkpoint_path = checkpoint_dir / "diffusion_best.pt"
    best_val_mse = float("inf")

    for epoch in range(int(config["training"]["epochs"])):
        progress = tqdm(loader, desc=f"Diffusion epoch {epoch + 1}")
        for batch in progress:
            cloudy = batch["cloudy"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            predicted_noise, noise = model(cloudy, mask, target)
            loss = mse(predicted_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        val_mse = _validation_mse(model, val_loader, device)
        print(f"Diffusion epoch {epoch + 1} validation noise MSE: {val_mse:.6f}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_mse": best_val_mse,
                    "metric": "noise_mse",
                },
                best_checkpoint_path,
            )
            print(f"Saved best diffusion checkpoint: {best_checkpoint_path}")

    checkpoint_path = checkpoint_dir / "diffusion.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(config["training"]["epochs"]),
            "best_val_mse": best_val_mse,
            "metric": "noise_mse",
        },
        checkpoint_path,
    )
    return best_checkpoint_path
