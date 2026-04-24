"""Qualitative visualization: per-coverage RGB grids showing every method.

For each cloud-coverage level in `data.cloud_coverages`, pick N samples from the
test set and render a (rows × cols) grid:

    row 0: cloudy input
    row 1: ground truth
    row 2: temporal baseline
    row 3: GAN
    row 4: diffusion

All rows share the same columns (i.e. same test samples), so visually you can
compare all three methods side-by-side against the same GT on every image.

Each grid is saved as `outputs/figures/qualitative_cov_{xx}.png`. Optionally
all coverages are stacked into a single `qualitative_all.png` for the report.

Usage:
    python scripts/qualitative_grid.py                 # defaults, 10 samples/cov
    python scripts/qualitative_grid.py --n 8           # 8 samples/cov
    python scripts/qualitative_grid.py --seed 123      # different sample pick
    python scripts/qualitative_grid.py --no-combined   # skip the combined PNG
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloud_removal.data.dataset import CloudRemovalDataset  # noqa: E402
from cloud_removal.data.synthetic_clouds import apply_cloud  # noqa: E402
from cloud_removal.models.baseline import multi_temporal_composite  # noqa: E402
from cloud_removal.models.diffusion import InpaintingDiffusion  # noqa: E402
from cloud_removal.models.unet import UNetGenerator  # noqa: E402
from cloud_removal.utils.config import resolve_device  # noqa: E402


ROW_LABELS = ["Cloudy input", "Ground truth", "Temporal", "GAN", "Diffusion"]


# --- loading --------------------------------------------------------------


def load_gan(config, device):
    model = UNetGenerator(in_channels=5, out_channels=4).to(device)
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt = ckpt_dir / "gan_generator_best.pt"
    if not ckpt.exists():
        ckpt = ckpt_dir / "gan_generator.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def load_diffusion(config, device):
    model = InpaintingDiffusion(timesteps=int(config["training"]["diffusion_timesteps"])).to(device)
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt = ckpt_dir / "diffusion_best.pt"
    if not ckpt.exists():
        ckpt = ckpt_dir / "diffusion.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


# --- per-method inference -------------------------------------------------


@torch.no_grad()
def run_gan(gan, cloudy, mask):
    inp = torch.cat([cloudy, mask], dim=1)
    raw = gan(inp)
    return (raw * mask + cloudy * (1.0 - mask)).clamp(0, 1)


@torch.no_grad()
def run_diffusion(diffusion, cloudy, mask):
    return diffusion.inpaint(cloudy, mask, steps=25)


def run_temporal(sample, config, device):
    """Mirror `evaluate.py` so the temporal baseline sees the same synthetic
    neighbours as in the quantitative results."""
    cloudy = sample["cloudy"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    target_np = sample["target"].cpu().numpy()
    coverage = float(sample["coverage"])

    # Deterministic rng from sample id so temporal matches the metrics run.
    digest = hashlib.sha256(sample["id"].encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))

    temporal_images = [cloudy.squeeze(0).cpu()]
    temporal_masks = [mask.squeeze(0).cpu()]
    for _ in range(int(config["data"]["temporal_neighbors"])):
        obs = apply_cloud(target_np, coverage, rng)
        temporal_images.append(torch.from_numpy(obs.cloudy))
        temporal_masks.append(torch.from_numpy(obs.mask[None, ...]))
    stack = torch.stack(temporal_images).to(device)
    masks = torch.stack(temporal_masks).to(device)
    pred = multi_temporal_composite(stack, masks, fallback=cloudy.squeeze(0))
    return pred.unsqueeze(0)


# --- RGB rendering --------------------------------------------------------


def to_rgb(chw: torch.Tensor | np.ndarray, gamma: float = 0.7, gain: float = 2.6) -> np.ndarray:
    """Take a CxHxW tensor/array (C ≥ 3, assumed band order R,G,B,NIR), return
    an HxWx3 uint8 RGB image.

    Satellite reflectance is dark without stretch; `gain` scales the linear
    radiance, `gamma` compresses highlights. Defaults tuned so typical
    Sentinel-2 patches land in a natural-looking range.
    """
    if isinstance(chw, torch.Tensor):
        chw = chw.detach().cpu().numpy()
    rgb = chw[:3].transpose(1, 2, 0)  # HxWx3
    rgb = np.clip(rgb * gain, 0, 1) ** gamma
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# --- sample selection -----------------------------------------------------


def pick_samples(
    dataset: CloudRemovalDataset,
    coverage_levels: list[float],
    n: int,
    seed: int,
) -> dict[float, list[int]]:
    """Pick `n` test-set indices per coverage bin. Sample IDs encode the target
    coverage as a `_cXX` suffix — we use that to bucket without having to load
    every mask. Selection is deterministic for a given (seed, n)."""
    rng = np.random.default_rng(seed)
    buckets: dict[float, list[int]] = {c: [] for c in coverage_levels}
    for i, sid in enumerate(dataset.ids):
        if "_c" not in sid:
            continue
        try:
            cov = int(sid.rsplit("_c", 1)[1]) / 100.0
        except ValueError:
            continue
        if cov in buckets:
            buckets[cov].append(i)

    picked: dict[float, list[int]] = {}
    for cov in coverage_levels:
        pool = buckets[cov]
        if len(pool) < n:
            print(f"[warn] coverage={cov} only has {len(pool)} samples, using all")
            picked[cov] = pool
        else:
            picked[cov] = list(rng.choice(pool, size=n, replace=False))
    return picked


# --- grid plotting --------------------------------------------------------


def render_grid(
    cov: float,
    samples: list[dict],
    preds: dict[str, list[np.ndarray]],
    out_path: Path,
) -> None:
    """Build a 5 × n figure for a given coverage and save as PNG."""
    n = len(samples)
    rows = [
        ("Cloudy input", [to_rgb(s["cloudy"]) for s in samples]),
        ("Ground truth", [to_rgb(s["target"]) for s in samples]),
        ("Temporal",     preds["temporal"]),
        ("GAN",          preds["gan"]),
        ("Diffusion",    preds["diffusion"]),
    ]

    fig_w = 1.4 * n + 0.9          # per-col ~1.4in + left margin for row labels
    fig_h = 1.4 * len(rows) + 0.6
    fig, axes = plt.subplots(len(rows), n, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"Qualitative comparison @ coverage = {cov:.2f}", fontsize=12, y=0.995)

    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=6)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_combined(per_cov_images: dict[float, dict], out_path: Path, n: int) -> None:
    """Stack all coverages into one tall figure for the report."""
    coverages = sorted(per_cov_images)
    total_rows = 5 * len(coverages)
    fig_w = 1.4 * n + 0.9
    fig_h = 1.15 * total_rows + 0.6
    fig, axes = plt.subplots(total_rows, n, figsize=(fig_w, fig_h), squeeze=False)
    for i, cov in enumerate(coverages):
        block = per_cov_images[cov]
        rows = [
            (f"[c={cov:.2f}] cloudy",  block["cloudy"]),
            ("ground truth", block["target"]),
            ("temporal",     block["temporal"]),
            ("gan",          block["gan"]),
            ("diffusion",    block["diffusion"]),
        ]
        for r_local, (label, imgs) in enumerate(rows):
            r = i * 5 + r_local
            for c in range(n):
                ax = axes[r][c]
                ax.imshow(imgs[c])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=4)

    fig.subplots_adjust(left=0.05, right=0.997, top=0.998, bottom=0.003, wspace=0.03, hspace=0.05)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- main -----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=10, help="samples per coverage bin")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--no-combined", action="store_true", help="skip the combined all-coverage PNG")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = resolve_device(config["device"])
    torch.manual_seed(args.seed)

    print("Loading models...")
    gan = load_gan(config, device)
    diffusion = load_diffusion(config, device)

    print("Loading test dataset...")
    dataset = CloudRemovalDataset(config["data"]["root"], "test")
    coverage_levels = [float(v) for v in config["data"]["cloud_coverages"]]
    picked = pick_samples(dataset, coverage_levels, args.n, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cov_images: dict[float, dict[str, list[np.ndarray]]] = {}
    for cov in coverage_levels:
        idxs = picked[cov]
        print(f"\nCoverage {cov:.2f}: running {len(idxs)} samples through 3 methods...")
        samples = [dataset[i] for i in idxs]
        preds = {"temporal": [], "gan": [], "diffusion": []}
        cloudy_imgs = []
        target_imgs = []
        for s in samples:
            cloudy_b = s["cloudy"].unsqueeze(0).to(device)
            mask_b = s["mask"].unsqueeze(0).to(device)

            tmp = run_temporal(s, config, device)
            g = run_gan(gan, cloudy_b, mask_b)
            d = run_diffusion(diffusion, cloudy_b, mask_b)

            cloudy_imgs.append(to_rgb(s["cloudy"]))
            target_imgs.append(to_rgb(s["target"]))
            preds["temporal"].append(to_rgb(tmp.squeeze(0)))
            preds["gan"].append(to_rgb(g.squeeze(0)))
            preds["diffusion"].append(to_rgb(d.squeeze(0)))

        render_grid(cov, samples, preds, out_dir / f"qualitative_cov_{int(cov * 100):02d}.png")
        per_cov_images[cov] = {
            "cloudy": cloudy_imgs,
            "target": target_imgs,
            **preds,
        }

    if not args.no_combined:
        render_combined(per_cov_images, out_dir / "qualitative_all.png", args.n)


if __name__ == "__main__":
    main()
