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


# Keep plotting order aligned with the summary CSV written by evaluate.py.
# Any metric listed here will only be plotted if it actually appears in the
# per-sample CSV header, so legacy files without e.g. ``l1_cloud_region``
# still render without errors.
_CANDIDATE_METRICS: tuple[str, ...] = ("psnr", "ssim", "l1", "l1_cloud_region", "ndvi_mae")


def plot_metric_curves(metrics_csv: str | Path, output_path: str | Path) -> None:
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    active_metrics = [m for m in _CANDIDATE_METRICS if m in fieldnames]
    if not active_metrics:
        return

    aggregates: dict[tuple[str, float], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, **{m: 0.0 for m in active_metrics}}
    )
    for row in rows:
        key = (row["method"], float(row["coverage_bin"]))
        aggregates[key]["count"] += 1.0
        for metric in active_metrics:
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                continue
            if value != value or value in (float("inf"), float("-inf")):  # NaN / inf guard
                continue
            aggregates[key][metric] += value

    grouped = []
    for (method, coverage), values in aggregates.items():
        count = max(values["count"], 1.0)
        entry = {"method": method, "coverage_bin": coverage}
        for metric in active_metrics:
            entry[metric] = values[metric] / count
        grouped.append(entry)
    grouped.sort(key=lambda item: (item["method"], item["coverage_bin"]))

    fig, axes = plt.subplots(1, len(active_metrics), figsize=(4.3 * len(active_metrics), 4), constrained_layout=True)
    if len(active_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, active_metrics, strict=True):
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
