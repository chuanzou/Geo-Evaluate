"""Plot diffusion training/validation loss curves.

Produces three PNGs under `outputs/figures/`:

  (1) diffusion_train_loss.png
      Training per-batch loss vs step, log-x axis, raw scatter + EMA line.
      Shows where the bulk of learning actually happens (hint: first 500 steps).

  (2) diffusion_train_loss_by_t.png
      Per-sample training loss bucketed by t range. Reveals *which*
      timesteps are still being learned vs. which plateaued. Essential for
      diffusion because the t=0 bucket is information-theoretically capped
      near 1.0 and would otherwise drown out the real learning signal.

  (3) diffusion_val_mse_heatmap.png
      Per-epoch, per-t validation MSE as a heatmap (epoch × t). A companion
      line plot overlay shows each t's curve across epochs.

Inputs (written by the patched `train_diffusion.py`):
  outputs/logs/train_loss.csv    columns: step, epoch, t, sample_loss, batch_loss
  outputs/logs/val_loss_by_t.csv columns: epoch, t, val_mse

Usage:
  python scripts/plot_loss.py
  python scripts/plot_loss.py --train-log outputs/logs/train_loss.csv \\
                              --val-log   outputs/logs/val_loss_by_t.csv \\
                              --out-dir   outputs/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-log", default="outputs/logs/train_loss.csv")
    p.add_argument("--val-log", default="outputs/logs/val_loss_by_t.csv")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--ema-alpha", type=float, default=0.98, help="EMA smoothing factor for plot 1")
    return p.parse_args()


# --- I/O -----------------------------------------------------------------


def read_train_log(path: Path):
    """Return step, epoch, t, sample_loss, batch_loss arrays. Tolerates old-format
    logs that lack the `t` / `sample_loss` columns (fills them with NaN)."""
    steps, epochs, ts, sample_losses, batch_losses = [], [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            epochs.append(int(row["epoch"]))
            ts.append(int(row["t"]) if row.get("t") not in (None, "") else -1)
            sample_losses.append(float(row["sample_loss"]) if row.get("sample_loss") else float("nan"))
            batch_losses.append(float(row["batch_loss"]) if row.get("batch_loss") else float("nan"))
    return (
        np.asarray(steps),
        np.asarray(epochs),
        np.asarray(ts),
        np.asarray(sample_losses),
        np.asarray(batch_losses),
    )


def read_val_log(path: Path):
    epochs, ts, mses = [], [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            ts.append(int(row["t"]))
            mses.append(float(row["val_mse"]))
    return np.asarray(epochs), np.asarray(ts), np.asarray(mses)


# --- helpers -------------------------------------------------------------


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average; higher alpha = more smoothing."""
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = acc
            continue
        acc = alpha * acc + (1.0 - alpha) * v
        out[i] = acc
    return out


# --- plot 1: train loss vs step, log-x ----------------------------------


def plot_train_loss(step, batch_loss, out_path: Path, ema_alpha: float) -> None:
    # One entry per (batch), not per sample — dedupe on step.
    _, unique_idx = np.unique(step, return_index=True)
    s = step[unique_idx]
    bl = batch_loss[unique_idx]
    order = np.argsort(s)
    s = s[order]
    bl = bl[order]

    # log scale doesn't like step=0; shift by 1 for x-axis only
    x = s + 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(x, bl, s=3, alpha=0.15, color="#4a90e2", label="per-batch loss (raw)")
    ax.plot(x, ema(bl, ema_alpha), color="#1a3d6e", linewidth=1.8, label=f"EMA α={ema_alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("masked noise MSE")
    ax.set_title("Diffusion training loss vs step")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- plot 2: train loss bucketed by t -----------------------------------


def plot_train_loss_by_t(step, t, sample_loss, out_path: Path, timesteps_hint: int = 100) -> None:
    if np.all(t < 0):
        print("  [skip] plot 2: training log has no `t` column (old format)")
        return

    # Three buckets: small / medium / large t
    T = int(t.max()) + 1 if t.max() >= 0 else timesteps_hint
    bounds = [(0, T // 3), (T // 3, 2 * T // 3), (2 * T // 3, T)]
    labels = [f"t∈[{a},{b})" for a, b in bounds]
    colors = ["#d95f02", "#7570b3", "#1b9e77"]  # orange / purple / green (colorblind-safe)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (lo, hi), lab, col in zip(bounds, labels, colors, strict=True):
        sel = (t >= lo) & (t < hi)
        if sel.sum() == 0:
            continue
        s = step[sel]
        loss = sample_loss[sel]
        # bucket the losses into equal-width step bins to get a stable line
        # despite having per-sample noise.
        nbins = 200
        bin_edges = np.linspace(s.min(), s.max(), nbins + 1)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        means = np.full(nbins, np.nan)
        for i in range(nbins):
            m = (s >= bin_edges[i]) & (s < bin_edges[i + 1])
            if m.any():
                means[i] = np.mean(loss[m])
        ax.plot(centers + 1, means, color=col, linewidth=1.8, label=lab)

    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("masked noise MSE (per-sample, bucketed)")
    ax.set_title("Training loss bucketed by diffusion timestep t")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")
    # Add an annotation explaining the t=0 floor
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(
        ax.get_xlim()[1] * 0.5,
        1.02,
        "Var(noise) = 1.0 (trivial baseline: predict 0)",
        color="red",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- plot 3: val MSE by t × epoch --------------------------------------


def plot_val_by_t(epoch, t, mse, out_path: Path) -> None:
    epochs = np.sort(np.unique(epoch))
    ts = np.sort(np.unique(t))
    grid = np.full((len(ts), len(epochs)), np.nan)
    for e, tv, m in zip(epoch, t, mse, strict=True):
        ei = np.searchsorted(epochs, e)
        ti = np.searchsorted(ts, tv)
        grid[ti, ei] = m

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1.3, 1.0]})

    # heatmap
    ax = axes[0]
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=(epochs[0] - 0.5, epochs[-1] + 0.5, ts[0] - 0.5, ts[-1] + 0.5),
        cmap="viridis",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("diffusion timestep t")
    ax.set_title("Validation noise MSE (fixed-t grid)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("masked noise MSE")

    # line overlay: one line per t
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, tv in enumerate(ts):
        ax.plot(
            epochs,
            grid[i],
            color=cmap(i / max(len(ts) - 1, 1)),
            label=f"t={tv}",
            linewidth=1.5,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val masked noise MSE")
    ax.set_title("Validation MSE per-t across epochs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- main ---------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log = Path(args.train_log)
    val_log = Path(args.val_log)

    if train_log.exists():
        step, epoch_tr, t, sample_loss, batch_loss = read_train_log(train_log)
        print(f"Loaded {len(step):,} rows from {train_log}")
        plot_train_loss(step, batch_loss, out_dir / "diffusion_train_loss.png", args.ema_alpha)
        plot_train_loss_by_t(step, t, sample_loss, out_dir / "diffusion_train_loss_by_t.png")
    else:
        print(f"[warn] missing {train_log} — run training once to generate it, then re-run this script")

    if val_log.exists():
        epoch_v, t_v, mse_v = read_val_log(val_log)
        print(f"Loaded {len(epoch_v):,} rows from {val_log}")
        plot_val_by_t(epoch_v, t_v, mse_v, out_dir / "diffusion_val_mse_heatmap.png")
    else:
        print(f"[warn] missing {val_log}")


if __name__ == "__main__":
    main()
