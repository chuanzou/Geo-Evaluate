from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

matplotlib_cache = Path("outputs") / ".matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(matplotlib_cache))

import matplotlib.pyplot as plt


def plot_metric_curves(metrics_csv: str | Path, output_path: str | Path) -> None:
    aggregates: dict[tuple[str, float], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "psnr": 0.0, "ssim": 0.0, "ndvi_mae": 0.0}
    )
    with Path(metrics_csv).open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["method"], float(row["coverage_bin"]))
            aggregates[key]["count"] += 1.0
            for metric in ["psnr", "ssim", "ndvi_mae"]:
                aggregates[key][metric] += float(row[metric])

    grouped = []
    for (method, coverage), values in aggregates.items():
        count = values["count"]
        grouped.append(
            {
                "method": method,
                "coverage_bin": coverage,
                "psnr": values["psnr"] / count,
                "ssim": values["ssim"] / count,
                "ndvi_mae": values["ndvi_mae"] / count,
            }
        )
    grouped.sort(key=lambda item: (item["method"], item["coverage_bin"]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axes, ["psnr", "ssim", "ndvi_mae"], strict=True):
        methods = sorted({row["method"] for row in grouped})
        for method in methods:
            sub = [row for row in grouped if row["method"] == method]
            ax.plot(
                [row["coverage_bin"] for row in sub],
                [row[metric] for row in sub],
                marker="o",
                label=method,
            )
        ax.set_xlabel("Cloud coverage")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
