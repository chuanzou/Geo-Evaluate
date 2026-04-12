from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.models.unet import PatchDiscriminator, UNetGenerator
from cloud_removal.utils.config import ensure_dir


def _validation_l1(generator: UNetGenerator, loader: DataLoader, device: torch.device) -> float:
    generator.eval()
    total_loss = 0.0
    total_count = 0
    l1 = nn.L1Loss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            model_input = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            prediction = generator(model_input)
            masked_loss = l1(prediction * mask, target * mask)
            total_loss += float(masked_loss.item())
            total_count += int(mask.sum().item() * target.shape[1])
    generator.train()
    return total_loss / max(total_count, 1)


def train_gan(config: dict, device: torch.device) -> Path:
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

    generator = UNetGenerator(in_channels=5, out_channels=4).to(device)
    discriminator = PatchDiscriminator(in_channels=9).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["training"]["learning_rate"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["training"]["learning_rate"], betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    lambda_l1 = float(config["training"]["lambda_l1"])
    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    best_checkpoint_path = checkpoint_dir / "gan_generator_best.pt"
    best_val_l1 = float("inf")

    for epoch in range(int(config["training"]["epochs"])):
        progress = tqdm(loader, desc=f"GAN epoch {epoch + 1}")
        for batch in progress:
            model_input = batch["input"].to(device)
            target = batch["target"].to(device)

            fake = generator(model_input)

            opt_d.zero_grad(set_to_none=True)
            real_pair = torch.cat([model_input, target], dim=1)
            fake_pair = torch.cat([model_input, fake.detach()], dim=1)
            d_real = discriminator(real_pair)
            d_fake = discriminator(fake_pair)
            d_loss = 0.5 * (
                bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            )
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_pair = torch.cat([model_input, fake], dim=1)
            g_adv = bce(discriminator(fake_pair), torch.ones_like(d_fake))
            g_l1 = l1(fake, target) * lambda_l1
            g_loss = g_adv + g_l1
            g_loss.backward()
            opt_g.step()

            progress.set_postfix({"g": f"{g_loss.item():.3f}", "d": f"{d_loss.item():.3f}"})

        val_l1 = _validation_l1(generator, val_loader, device)
        print(f"GAN epoch {epoch + 1} validation cloud-region L1: {val_l1:.6f}")
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(
                {
                    "model_state_dict": generator.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_l1": best_val_l1,
                    "metric": "cloud_region_l1",
                },
                best_checkpoint_path,
            )
            print(f"Saved best GAN checkpoint: {best_checkpoint_path}")

    checkpoint_path = checkpoint_dir / "gan_generator.pt"
    torch.save(
        {
            "model_state_dict": generator.state_dict(),
            "epoch": int(config["training"]["epochs"]),
            "best_val_l1": best_val_l1,
            "metric": "cloud_region_l1",
        },
        checkpoint_path,
    )
    return best_checkpoint_path
