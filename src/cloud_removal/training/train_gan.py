from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.models.unet import PatchDiscriminator, UNetGenerator
from cloud_removal.utils.config import ensure_dir

# Numerical-stability constants.  ``EPS`` is used wherever we take a log on
# generator/discriminator probabilities so that ``log(0)`` can never occur.
# ``GRAD_CLIP_NORM`` caps gradient magnitude per step to avoid the exploding
# gradients that were crashing first-epoch training on MPS.
EPS = 1e-8
GRAD_CLIP_NORM = 1.0


def _stable_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """BCE-with-logits reimplemented with an explicit epsilon on the sigmoid
    probabilities.

    ``torch.nn.BCEWithLogitsLoss`` is already numerically stable via the
    log-sum-exp trick, but on MPS we have observed that extremely confident
    discriminator logits (|z| > 30) still produce NaNs when combined with the
    L1 term and BatchNorm statistics.  Clamping the probability with
    ``eps`` guarantees that ``log(p)`` and ``log(1 - p)`` both stay finite.
    """
    prob = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    return -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob)).mean()


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
    l1 = nn.L1Loss()
    lambda_l1 = float(config["training"]["lambda_l1"])
    # Mild one-sided label smoothing — "real" -> 0.9 instead of 1.0.  This is a
    # well-known GAN stabilisation trick; it prevents the discriminator from
    # driving its logits to +inf, which is exactly the pattern that caused
    # log(0) -> NaN under the previous BCEWithLogitsLoss wiring.
    real_label = 0.9
    fake_label = 0.0

    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    best_checkpoint_path = checkpoint_dir / "gan_generator_best.pt"
    best_val_l1 = float("inf")

    for epoch in range(int(config["training"]["epochs"])):
        progress = tqdm(loader, desc=f"GAN epoch {epoch + 1}")
        skipped_batches = 0
        for batch in progress:
            model_input = batch["input"].to(device)
            target = batch["target"].to(device)

            # ----- guard against NaN/Inf in the input itself -----
            if not torch.isfinite(model_input).all() or not torch.isfinite(target).all():
                skipped_batches += 1
                continue

            fake = generator(model_input)
            # The generator ends with a Sigmoid, so ``fake`` is already in
            # [0, 1] — clamp defensively in case an MPS kernel returns a
            # slightly out-of-range value from fused autograd ops.
            fake = fake.clamp(0.0, 1.0)

            # ========== Discriminator step ==========
            opt_d.zero_grad(set_to_none=True)
            real_pair = torch.cat([model_input, target], dim=1)
            fake_pair_detached = torch.cat([model_input, fake.detach()], dim=1)
            d_real_logits = discriminator(real_pair)
            d_fake_logits = discriminator(fake_pair_detached)

            d_loss_real = _stable_bce_with_logits(
                d_real_logits, torch.full_like(d_real_logits, real_label)
            )
            d_loss_fake = _stable_bce_with_logits(
                d_fake_logits, torch.full_like(d_fake_logits, fake_label)
            )
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            if not torch.isfinite(d_loss):
                skipped_batches += 1
                opt_d.zero_grad(set_to_none=True)
                continue

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP_NORM)
            opt_d.step()

            # ========== Generator step ==========
            opt_g.zero_grad(set_to_none=True)
            fake_pair = torch.cat([model_input, fake], dim=1)
            g_adv_logits = discriminator(fake_pair)
            g_adv = _stable_bce_with_logits(
                g_adv_logits, torch.full_like(g_adv_logits, real_label)
            )
            g_l1 = l1(fake, target) * lambda_l1
            g_loss = g_adv + g_l1

            if not torch.isfinite(g_loss):
                skipped_batches += 1
                opt_g.zero_grad(set_to_none=True)
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_NORM)
            opt_g.step()

            progress.set_postfix(
                {
                    "g": f"{g_loss.item():.3f}",
                    "g_adv": f"{g_adv.item():.3f}",
                    "g_l1": f"{g_l1.item():.3f}",
                    "d": f"{d_loss.item():.3f}",
                    "skip": skipped_batches,
                }
            )

        val_l1 = _validation_l1(generator, val_loader, device)
        print(
            f"GAN epoch {epoch + 1} validation cloud-region L1: {val_l1:.6f} "
            f"(skipped {skipped_batches} non-finite batches)"
        )
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
