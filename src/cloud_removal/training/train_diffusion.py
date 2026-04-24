from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.models.diffusion import InpaintingDiffusion
from cloud_removal.utils.config import ensure_dir


# --- core loss helpers ---------------------------------------------------


def _masked_mse(predicted_noise: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted noise MSE, aggregated over the whole batch.

    This is the gradient signal: only cloud-region pixels contribute, because
    outside the mask the denoiser's input is already clean `cloudy == target`
    and there's no noise to predict. Using a single denominator across the
    batch weights samples by their mask size (bigger clouds count more).
    """
    diff = (predicted_noise - noise) ** 2
    mask_b = mask.expand_as(diff)
    denom = mask_b.sum().clamp(min=1.0)
    return (diff * mask_b).sum() / denom


def _per_sample_masked_mse(
    predicted_noise: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sample version of `_masked_mse` — shape (B,).

    Used for logging only. Each sample's MSE is averaged over *its own* mask
    so samples with tiny clouds aren't drowned out in the t-bucket analysis.
    """
    diff = (predicted_noise - noise) ** 2            # (B, C, H, W)
    mask_b = mask.expand_as(diff)
    per_sample_denom = mask_b.sum(dim=(1, 2, 3)).clamp(min=1.0)  # (B,)
    return (diff * mask_b).sum(dim=(1, 2, 3)) / per_sample_denom  # (B,)


# --- validation: fixed-t grid -------------------------------------------


def _validation_mse_at_t(
    model: InpaintingDiffusion,
    loader: DataLoader,
    device: torch.device,
    t_value: int,
    generator: torch.Generator,
) -> float:
    """Average masked noise MSE over the val set with every sample frozen at t=t_value.

    Uses a fixed torch.Generator so the noise draw is reproducible — if two
    checkpoints are evaluated at the same t they see the same noise, making
    epoch-over-epoch comparisons meaningful.
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            B = target.shape[0]
            t = torch.full((B,), t_value, dtype=torch.long, device=device)
            # generator is CPU-side for portability across cuda/mps/cpu; we
            # draw on CPU and move.
            noise = torch.randn(target.shape, generator=generator).to(device)
            predicted_noise = model.forward_at_t(cloudy, mask, target, t, noise)
            mask_b = mask.expand_as(noise)
            weight = float(mask_b.sum().item())
            batch_mse = float(_masked_mse(predicted_noise, noise, mask).item())
            total_loss += batch_mse * weight
            total_weight += weight
    model.train()
    return total_loss / max(total_weight, 1.0)


def _validation_by_t_grid(
    model: InpaintingDiffusion,
    loader: DataLoader,
    device: torch.device,
    t_grid: list[int],
) -> dict[int, float]:
    """Return {t: avg_val_mse} over the full val set for each t in t_grid."""
    results: dict[int, float] = {}
    # One generator per epoch evaluation — reseeded so epochs are comparable.
    for t in t_grid:
        generator = torch.Generator()
        generator.manual_seed(1000 + int(t))  # per-t fixed seed
        results[int(t)] = _validation_mse_at_t(model, loader, device, int(t), generator)
    return results


# --- training loop ------------------------------------------------------


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

    timesteps = int(config["training"]["diffusion_timesteps"])
    model = InpaintingDiffusion(timesteps=timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    checkpoint_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    best_checkpoint_path = checkpoint_dir / "diffusion_best.pt"
    best_val_mse = float("inf")

    # Diffusion 每步只采样一个随机 timestep 来学习，数据利用率远低于 GAN，
    # 所以允许用独立的 diffusion_epochs 覆盖默认的 epochs（GAN 共用）。
    total_epochs = int(config["training"].get("diffusion_epochs", config["training"]["epochs"]))

    # --- loss logging setup ---
    # Per-sample training loss log: one row per *sample* per batch. With
    # 10k samples × 50 epochs this is ~500k rows / a few MB — fine.
    log_dir = ensure_dir(Path(config["outputs"].get("log_dir", "outputs/logs")))
    train_log_path = log_dir / "train_loss.csv"
    val_log_path = log_dir / "val_loss_by_t.csv"
    train_log_file = train_log_path.open("w", newline="", encoding="utf-8")
    train_writer = csv.writer(train_log_file)
    train_writer.writerow(["step", "epoch", "t", "sample_loss", "batch_loss"])
    val_log_file = val_log_path.open("w", newline="", encoding="utf-8")
    val_writer = csv.writer(val_log_file)
    val_writer.writerow(["epoch", "t", "val_mse"])

    # Fixed-t grid for validation: 10 evenly spaced t's across [0, T-1].
    # 10 passes over val set ≈ 10× the cost of the old single-random-t val,
    # still cheap compared to training.
    t_grid = sorted({int(round(i * (timesteps - 1) / 9)) for i in range(10)})

    global_step = 0
    try:
        for epoch in range(total_epochs):
            progress = tqdm(loader, desc=f"Diffusion epoch {epoch + 1}")
            for batch in progress:
                cloudy = batch["cloudy"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)

                predicted_noise, noise, t = model(cloudy, mask, target)
                loss = _masked_mse(predicted_noise, noise, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # --- log per-sample ---
                with torch.no_grad():
                    per_sample = _per_sample_masked_mse(predicted_noise.detach(), noise, mask)
                batch_loss_val = float(loss.item())
                t_cpu = t.detach().cpu().tolist()
                ps_cpu = per_sample.detach().cpu().tolist()
                for ti, psi in zip(t_cpu, ps_cpu, strict=True):
                    train_writer.writerow([global_step, epoch + 1, int(ti), f"{psi:.6f}", f"{batch_loss_val:.6f}"])

                progress.set_postfix({"loss": f"{batch_loss_val:.4f}"})
                global_step += 1

            # Flush training log once per epoch so partial runs are recoverable.
            train_log_file.flush()

            # --- fixed-t grid validation ---
            val_by_t = _validation_by_t_grid(model, val_loader, device, t_grid)
            for t_val, mse in val_by_t.items():
                val_writer.writerow([epoch + 1, t_val, f"{mse:.6f}"])
            val_log_file.flush()

            val_mse_mean = sum(val_by_t.values()) / len(val_by_t)
            print(f"Diffusion epoch {epoch + 1} val MSE (mean over t grid): {val_mse_mean:.6f}")
            print("  per-t:", "  ".join(f"t={t}:{v:.4f}" for t, v in val_by_t.items()))

            if val_mse_mean < best_val_mse:
                best_val_mse = val_mse_mean
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_val_mse": best_val_mse,
                        "metric": "noise_mse_mean_over_t_grid",
                        "t_grid": t_grid,
                    },
                    best_checkpoint_path,
                )
                print(f"Saved best diffusion checkpoint: {best_checkpoint_path}")
    finally:
        train_log_file.close()
        val_log_file.close()

    checkpoint_path = checkpoint_dir / "diffusion.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": total_epochs,
            "best_val_mse": best_val_mse,
            "metric": "noise_mse_mean_over_t_grid",
        },
        checkpoint_path,
    )
    return best_checkpoint_path
