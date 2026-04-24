"""Torch-free diagnosis of the diffusion failure.

We can't run the model here (no torch in the sandbox), but we can:

1. Verify the hypothesis that the per-sample L1 is ENTIRELY driven by the
   cloud region (i.e. the composite step correctly keeps outside-mask clean).
2. Measure the actual distribution of *target* pixel values inside the cloud
   region across the test set, so we know what L1_cloud a model would get for
   various constant outputs (pred=0, pred=mean, pred=1, etc.).
3. Infer from L1_cloud what the diffusion model is roughly outputting.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/synthetic_cloud_benchmark"


def load_sample(sid: str):
    gt = np.load(DATA / "ground_truth" / f"{sid}.npy").astype(np.float32)
    cloudy = np.load(DATA / "cloudy" / f"{sid}.npy").astype(np.float32)
    mask = np.load(DATA / "masks" / f"{sid}.npy").astype(np.float32)
    if mask.ndim == 2:
        mask = mask[None, ...]
    return gt, cloudy, mask


def main():
    ids_by_cov: dict[float, list[str]] = {}
    test_ids = (DATA / "splits/test.txt").read_text().splitlines()
    for sid in test_ids:
        sid = sid.strip()
        if not sid or "_c" not in sid:
            continue
        try:
            cov = int(sid.rsplit("_c", 1)[1]) / 100.0
        except ValueError:
            continue
        ids_by_cov.setdefault(cov, []).append(sid)

    print("=== TARGET pixel stats inside the cloud region, by coverage ===")
    print(f"{'cov':>5}  {'n':>4}  {'mean':>6}  {'std':>6}  "
          f"{'L1 if pred=0':>12}  {'L1 if pred=mean':>15}  {'L1 if pred=1':>12}  {'L1 if pred=cloudy':>17}")

    # Also: what L1_cloud does the *cloudy* input itself achieve?  (Upper bound
    # you should beat: cloudy ≈ 1 in mask ==> L1_cloud ≈ 1 - mean.)
    for cov in sorted(ids_by_cov):
        sample_ids = ids_by_cov[cov][:80]  # subsample for speed
        target_vals = []
        cloudy_vals = []
        for sid in sample_ids:
            gt, cloudy, mask = load_sample(sid)
            m = mask.squeeze()
            if m.sum() == 0:
                continue
            # flatten inside-mask pixels across all 4 channels
            m3 = np.broadcast_to(m[None, ...], gt.shape)
            target_vals.append(gt[m3.astype(bool)])
            cloudy_vals.append(cloudy[m3.astype(bool)])
        target_vals = np.concatenate(target_vals)
        cloudy_vals = np.concatenate(cloudy_vals)
        mu = target_vals.mean()
        sd = target_vals.std()
        L1_zero = np.mean(np.abs(0 - target_vals))
        L1_mean = np.mean(np.abs(mu - target_vals))
        L1_one = np.mean(np.abs(1 - target_vals))
        L1_cloudy = np.mean(np.abs(cloudy_vals - target_vals))
        print(f"{cov:>5.2f}  {len(sample_ids):>4d}  "
              f"{mu:>6.4f}  {sd:>6.4f}  {L1_zero:>12.4f}  {L1_mean:>15.4f}  "
              f"{L1_one:>12.4f}  {L1_cloudy:>17.4f}")

    # Reported diffusion L1_cloud from the CSV, for easy comparison:
    print("\n=== reported diffusion L1_cloud_region from outputs/metrics/diffusion_metrics_by_coverage.csv ===")
    csv = (ROOT / "outputs/metrics/diffusion_metrics_by_coverage.csv").read_text().splitlines()
    for line in csv:
        print("  ", line)


if __name__ == "__main__":
    main()
