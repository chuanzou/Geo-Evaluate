"""Diagnose diffusion sampler: compare different initialization strategies.

We suspect the current `inpaint()` "smart init" — `0.61*cloudy + 0.79*noise`
— biases the cloud-region output toward cloudy (white=1) values, because
inside the mask `cloudy ≈ 1` while the target is the real scene (mean ~0.3).

This script runs three sampling variants on a handful of test samples and
prints per-variant L1 and L1_cloud so we can see which init is better.

Variants:
  A) cloudy-proxy init (current): x = sqrt(alpha_T)*cloudy + sqrt(1-alpha_T)*noise
  B) pure-noise init:              x = noise                                 (standard DDPM)
  C) mean-proxy init:              x = sqrt(alpha_T)*global_mean + sqrt(1-alpha_T)*noise
  D) clip-only, x0-predict, single step (Tweedie-at-T-1): use x=noise, take one forward
     pass and return x0_pred directly (skip iterative posterior)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloud_removal.data.dataset import CloudRemovalDataset  # noqa: E402
from cloud_removal.models.diffusion import InpaintingDiffusion  # noqa: E402
from cloud_removal.evaluation.metrics import evaluate_prediction  # noqa: E402


def load_model(config: dict, device: torch.device) -> InpaintingDiffusion:
    model = InpaintingDiffusion(timesteps=int(config["training"]["diffusion_timesteps"])).to(device)
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "diffusion_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


@torch.no_grad()
def sample(
    model: InpaintingDiffusion,
    cloudy: torch.Tensor,
    mask: torch.Tensor,
    init_strategy: str,
    one_step: bool = False,
) -> torch.Tensor:
    """Modified inpaint loop with controllable init and optional single-step mode."""
    device = cloudy.device
    T = model.timesteps
    alpha_bar_last = model.alpha_bars[-1]

    if init_strategy == "cloudy_proxy":
        noise_init = torch.randn_like(cloudy)
        x = alpha_bar_last.sqrt() * cloudy + (1.0 - alpha_bar_last).sqrt() * noise_init
    elif init_strategy == "pure_noise":
        x = torch.randn_like(cloudy)
    elif init_strategy == "mean_proxy":
        # Use the known clean (outside-mask) pixels' mean as a target stand-in.
        mean_val = (cloudy * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1.0)
        noise_init = torch.randn_like(cloudy)
        x = alpha_bar_last.sqrt() * mean_val + (1.0 - alpha_bar_last).sqrt() * noise_init
    elif init_strategy == "zero_proxy":
        # target proxy = 0 (strong bias the *other* way, sanity-check)
        noise_init = torch.randn_like(cloudy)
        x = alpha_bar_last.sqrt() * torch.zeros_like(cloudy) + (1.0 - alpha_bar_last).sqrt() * noise_init
    else:
        raise ValueError(init_strategy)
    x = x * mask + cloudy * (1.0 - mask)

    if one_step:
        # Run only step = T-1 and return x0_pred directly.
        step = T - 1
        t_channel = torch.full((cloudy.shape[0], 1, *cloudy.shape[-2:]), step / max(T - 1, 1), device=device)
        eps = model.denoiser(torch.cat([x, mask, t_channel], dim=1))
        alpha_bar = model.alpha_bars[step]
        x0 = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        x0 = x0.clamp(0, 1)
        return (x0 * mask + cloudy * (1 - mask)).clamp(0, 1)

    for step in reversed(range(T)):
        t_channel = torch.full((cloudy.shape[0], 1, *cloudy.shape[-2:]), step / max(T - 1, 1), device=device)
        eps = model.denoiser(torch.cat([x, mask, t_channel], dim=1))
        alpha_bar = model.alpha_bars[step]
        x0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(0, 1)
        if step > 0:
            alpha_bar_prev = model.alpha_bars[step - 1]
            beta = model.betas[step]
            alpha = model.alphas[step]
            coef_x0 = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
            coef_xt = alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
            mean = coef_x0 * x0_pred + coef_xt * x
            var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            x = mean + var.clamp(min=0.0).sqrt() * torch.randn_like(x)
        else:
            x = x0_pred
        x = x * mask + cloudy * (1 - mask)
    return x.clamp(0, 1)


def bin_samples_by_coverage(dataset: CloudRemovalDataset, targets=(0.05, 0.10, 0.30, 0.50, 0.70), n=5):
    """Return `n` sample indices for each target coverage bin."""
    buckets = {c: [] for c in targets}
    for i, sid in enumerate(dataset.ids):
        # quick parse: id is like s2_beijing_patch_2023_02274_c50 -> coverage 0.50
        if "_c" in sid:
            try:
                cov = int(sid.rsplit("_c", 1)[1]) / 100.0
                if cov in buckets and len(buckets[cov]) < n:
                    buckets[cov].append(i)
            except ValueError:
                continue
        if all(len(v) >= n for v in buckets.values()):
            break
    return buckets


def summarize(name: str, rows: list[dict]):
    by_cov: dict = {}
    for r in rows:
        by_cov.setdefault(r["coverage_bin"], []).append(r)
    print(f"\n=== {name} ===")
    print(f"{'coverage':>8}  {'n':>3}  {'psnr':>6}  {'ssim':>6}  {'l1':>7}  {'l1_cloud':>8}  {'pred_mean_cloud':>15}")
    for cov, bucket in sorted(by_cov.items()):
        psnr = np.mean([b["psnr"] for b in bucket])
        ssim = np.mean([b["ssim"] for b in bucket])
        l1 = np.mean([b["l1"] for b in bucket])
        l1c = np.mean([b["l1_cloud_region"] for b in bucket])
        pm = np.mean([b["pred_mean_cloud"] for b in bucket])
        print(f"{cov:>8.2f}  {len(bucket):>3d}  {psnr:>6.2f}  {ssim:>6.4f}  {l1:>7.4f}  {l1c:>8.4f}  {pm:>15.4f}")


def main():
    with open(ROOT / "configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cpu")  # small number of samples, CPU is fine
    torch.manual_seed(0)

    model = load_model(config, device)
    dataset = CloudRemovalDataset(config["data"]["root"], "test")
    buckets = bin_samples_by_coverage(dataset, n=3)
    print("Samples per coverage:", {k: len(v) for k, v in buckets.items()})

    strategies = [
        ("A_cloudy_proxy (current)", "cloudy_proxy", False),
        ("B_pure_noise", "pure_noise", False),
        ("C_mean_proxy", "mean_proxy", False),
        ("D_zero_proxy", "zero_proxy", False),
        ("E_pure_noise_one_step", "pure_noise", True),
    ]

    results: dict[str, list[dict]] = {name: [] for name, _, _ in strategies}

    # Record target mean inside the cloud region, too (baseline reference).
    ref = []

    for cov, idxs in buckets.items():
        for idx in idxs:
            sample_item = dataset[idx]
            cloudy = sample_item["cloudy"].unsqueeze(0).to(device)
            mask = sample_item["mask"].unsqueeze(0).to(device)
            target = sample_item["target"].unsqueeze(0).to(device)

            target_np = target.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()
            target_mean_cloud = float((target_np * mask_np).sum() / max(mask_np.sum() * target_np.shape[0], 1))
            ref.append({"coverage_bin": cov, "target_mean_cloud": target_mean_cloud})

            for name, strat, one_step in strategies:
                torch.manual_seed(42 + idx)  # reproducibility per sample
                pred = sample(model, cloudy, mask, strat, one_step=one_step)
                pred_np = pred.squeeze(0).cpu().numpy()
                m = evaluate_prediction(pred_np, target_np, mask_np)
                pred_mean_cloud = float((pred_np * mask_np).sum() / max(mask_np.sum() * pred_np.shape[0], 1))
                results[name].append({"coverage_bin": cov, **m, "pred_mean_cloud": pred_mean_cloud})

    # print reference
    print("\n=== reference: target pixel mean inside cloud region ===")
    by_cov: dict = {}
    for r in ref:
        by_cov.setdefault(r["coverage_bin"], []).append(r["target_mean_cloud"])
    for cov, vals in sorted(by_cov.items()):
        print(f"  coverage={cov:.2f}  target_mean_cloud={np.mean(vals):.4f}")

    for name, _, _ in strategies:
        summarize(name, results[name])


if __name__ == "__main__":
    main()
