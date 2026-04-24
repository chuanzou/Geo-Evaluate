"""Per-pixel error heatmaps for each method/coverage, side-by-side.

Produces the visual counterpart to the quantitative L1_cloud table. For each
coverage bin, samples the **same** test patches as `qualitative_grid.py`
(seed-matched) and renders a 4 × N grid:

    row 1: ground truth RGB, with the cloud-mask boundary drawn as a
           thin white contour so you can see where each method had to
           inpaint
    row 2: temporal  |pred − target|  (per-pixel L1, averaged over 4 bands)
    row 3: GAN       |pred − target|
    row 4: diffusion |pred − target|

All error heatmaps share a single magma colormap with vmin=0 and a fixed
vmax (default 0.3), so brightness is directly comparable across methods
and across coverages. A shared colorbar sits at the right of each figure.

The mean L1 *inside the cloud region only* is printed in the top-right
corner of every panel — same definition as the `l1_cloud_region` column
in the metrics CSV, so the visual and quantitative stories match.

Usage:
    python scripts/qualitative_error_grid.py                 # defaults
    python scripts/qualitative_error_grid.py --n 8
    python scripts/qualitative_error_grid.py --vmax 0.2      # tighter color range
    python scripts/qualitative_error_grid.py --no-combined
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # reuse helpers from the sibling script

from cloud_removal.data.dataset import CloudRemovalDataset  # noqa: E402
from cloud_removal.utils.config import resolve_device  # noqa: E402

# Reuse the exact same inference + sample-picking + RGB conversion as the
# prediction grid, so the two figures show the same samples processed the
# same way.
from qualitative_grid import (  # noqa: E402
    load_gan,
    load_diffusion,
    pick_samples,
    run_gan,
    run_diffusion,
    run_temporal,
    to_rgb,
)


# --- error computation ---------------------------------------------------


def compute_error(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """|pred − target| averaged over channels → (H, W) numpy array.

    Channel-mean gives a single scalar field per pixel that's easy to read
    visually. Using 4 bands (including NIR) to stay consistent with the
    `l1` column in the metrics CSV.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if target.dim() == 4:
        target = target.squeeze(0)
    err = (pred - target).abs().mean(dim=0)   # (H, W)
    return err.detach().cpu().numpy()


def l1_cloud_scalar(err_map: np.ndarray, mask: np.ndarray) -> float:
    """Mean L1 inside the cloud mask (= metrics' l1_cloud_region, per sample)."""
    m = mask.squeeze()
    denom = float(m.sum())
    if denom < 1.0:
        return 0.0
    return float((err_map * m).sum() / denom)


# --- rendering -----------------------------------------------------------


def draw_gt_with_contour(ax, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Show the RGB ground truth and overlay the cloud-mask boundary."""
    ax.imshow(rgb)
    # contour at 0.5 picks out the mask boundary. Thin white line reads on
    # almost any satellite background without washing the image out.
    m = mask.squeeze()
    ax.contour(m, levels=[0.5], colors="white", linewidths=0.8, alpha=0.95)


def draw_error(ax, err: np.ndarray, vmax: float):
    """Show an error map; return the AxesImage so a shared colorbar can find it."""
    return ax.imshow(err, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")


def annotate_l1(ax, value: float) -> None:
    ax.text(
        0.97, 0.97,
        f"L1={value:.3f}",
        transform=ax.transAxes,
        ha="right", va="top",
        color="white",
        fontsize=7,
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.2),
    )


def _strip_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_coverage(
    cov: float,
    gt_rgb: list[np.ndarray],
    masks: list[np.ndarray],
    errors: dict[str, list[np.ndarray]],
    l1s: dict[str, list[float]],
    vmax: float,
    out_path: Path,
) -> None:
    n = len(gt_rgb)
    method_rows = [("Temporal", "temporal"), ("GAN", "gan"), ("Diffusion", "diffusion")]

    fig_w = 1.35 * n + 1.4   # extra width for colorbar
    fig_h = 1.35 * 4 + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    # one column per sample + one narrow column for the colorbar
    gs = fig.add_gridspec(
        nrows=4, ncols=n + 1,
        width_ratios=[1] * n + [0.06],
        wspace=0.04, hspace=0.06,
        left=0.055, right=0.985, top=0.93, bottom=0.02,
    )

    fig.suptitle(f"Per-pixel L1 error @ coverage = {cov:.2f}", fontsize=12, y=0.985)

    # Row 0: GT + contour
    for c in range(n):
        ax = fig.add_subplot(gs[0, c])
        draw_gt_with_contour(ax, gt_rgb[c], masks[c])
        _strip_axes(ax)
        if c == 0:
            ax.set_ylabel("GT + cloud\nboundary", fontsize=9, rotation=90, labelpad=6)

    # Rows 1-3: error heatmaps
    last_im = None
    for r, (label, key) in enumerate(method_rows, start=1):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c])
            last_im = draw_error(ax, errors[key][c], vmax)
            annotate_l1(ax, l1s[key][c])
            _strip_axes(ax)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=6)

    # Shared colorbar
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("per-pixel L1 error", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_combined(per_cov, vmax: float, out_path: Path, n: int) -> None:
    """All coverages stacked vertically, sharing one colorbar on the right."""
    coverages = sorted(per_cov)
    rows_per_cov = 4
    total_rows = rows_per_cov * len(coverages)

    fig_w = 1.35 * n + 1.4
    fig_h = 1.1 * total_rows + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=total_rows, ncols=n + 1,
        width_ratios=[1] * n + [0.04],
        wspace=0.04, hspace=0.06,
        left=0.055, right=0.987, top=0.995, bottom=0.005,
    )

    last_im = None
    for i, cov in enumerate(coverages):
        block = per_cov[cov]
        gt_rgb = block["gt_rgb"]
        masks = block["masks"]
        errors = block["errors"]
        l1s = block["l1s"]
        method_rows = [("Temporal", "temporal"), ("GAN", "gan"), ("Diffusion", "diffusion")]

        # Row 0 of this coverage-block: GT + contour
        for c in range(n):
            ax = fig.add_subplot(gs[i * rows_per_cov + 0, c])
            draw_gt_with_contour(ax, gt_rgb[c], masks[c])
            _strip_axes(ax)
            if c == 0:
                ax.set_ylabel(f"[c={cov:.2f}]\nGT + mask", fontsize=8, rotation=90, labelpad=4)

        # Rows 1-3: methods
        for r, (label, key) in enumerate(method_rows, start=1):
            for c in range(n):
                ax = fig.add_subplot(gs[i * rows_per_cov + r, c])
                last_im = draw_error(ax, errors[key][c], vmax)
                annotate_l1(ax, l1s[key][c])
                _strip_axes(ax)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=4)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("per-pixel L1 error", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- main ----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=10, help="samples per coverage bin")
    p.add_argument("--seed", type=int, default=42, help="match qualitative_grid.py for same samples")
    p.add_argument("--vmax", type=float, default=0.30,
                   help="upper end of shared color scale (0.3 covers most signals incl. 70%%-cov temporal)")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--no-combined", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = resolve_device(config["device"])
    torch.manual_seed(args.seed)

    # Slightly fancier matplotlib style for publication-ish look.
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.labelcolor"] = "#222"

    print("Loading models...")
    gan = load_gan(config, device)
    diffusion = load_diffusion(config, device)

    print("Loading test dataset...")
    dataset = CloudRemovalDataset(config["data"]["root"], "test")
    coverage_levels = [float(v) for v in config["data"]["cloud_coverages"]]
    picked = pick_samples(dataset, coverage_levels, args.n, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cov: dict[float, dict] = {}
    for cov in coverage_levels:
        idxs = picked[cov]
        print(f"\nCoverage {cov:.2f}: running {len(idxs)} samples through 3 methods...")
        gt_rgb_list, mask_list = [], []
        errors = {"temporal": [], "gan": [], "diffusion": []}
        l1s = {"temporal": [], "gan": [], "diffusion": []}

        for idx in idxs:
            s = dataset[idx]
            cloudy_b = s["cloudy"].unsqueeze(0).to(device)
            mask_b = s["mask"].unsqueeze(0).to(device)
            target_b = s["target"].unsqueeze(0).to(device)

            preds = {
                "temporal":  run_temporal(s, config, device),
                "gan":       run_gan(gan, cloudy_b, mask_b),
                "diffusion": run_diffusion(diffusion, cloudy_b, mask_b),
            }

            mask_np = s["mask"].cpu().numpy()
            gt_rgb_list.append(to_rgb(s["target"]))
            mask_list.append(mask_np)
            for key, pred in preds.items():
                err = compute_error(pred, target_b)
                errors[key].append(err)
                l1s[key].append(l1_cloud_scalar(err, mask_np))

        out_path = out_dir / f"qualitative_error_cov_{int(cov * 100):02d}.png"
        render_coverage(cov, gt_rgb_list, mask_list, errors, l1s, args.vmax, out_path)

        per_cov[cov] = {
            "gt_rgb": gt_rgb_list,
            "masks":  mask_list,
            "errors": errors,
            "l1s":    l1s,
        }

    if not args.no_combined:
        render_combined(per_cov, args.vmax, out_dir / "qualitative_error_all.png", args.n)


if __name__ == "__main__":
    main()
