from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_removal.data.synthetic_clouds import apply_cloud, random_sentinel_like_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare synthetic cloud benchmark data.")
    parser.add_argument("--root", default="data/processed/synthetic_cloud_benchmark")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-ground-truth", action="store_true")
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.05, 0.10, 0.30, 0.50, 0.70])
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    gt_dir = root / "ground_truth"
    cloudy_dir = root / "cloudy"
    mask_dir = root / "masks"
    split_dir = root / "splits"
    for directory in [gt_dir, cloudy_dir, mask_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.from_ground_truth:
        gt_paths = sorted(gt_dir.glob("*.npy"))
        if not gt_paths:
            raise FileNotFoundError(f"No .npy ground-truth patches found in {gt_dir}")
        base_images = [(path.stem, np.load(path).astype(np.float32)) for path in gt_paths]
    else:
        base_images = [
            (f"patch_{index:05d}", random_sentinel_like_patch(args.image_size, rng))
            for index in range(args.num_images)
        ]

    base_indices = np.arange(len(base_images))
    rng.shuffle(base_indices)
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("--train-fraction + --val-fraction must be less than 1.0")
    train_cut = int(len(base_indices) * args.train_fraction)
    val_cut = train_cut + int(len(base_indices) * args.val_fraction)
    train_bases = {base_images[index][0] for index in base_indices[:train_cut]}
    val_bases = {base_images[index][0] for index in base_indices[train_cut:val_cut]}
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for base_id, image in base_images:
        for coverage in args.coverages:
            sample_id = f"{base_id}_c{int(round(coverage * 100)):02d}"
            cloud = apply_cloud(image, coverage, rng)
            np.save(gt_dir / f"{sample_id}.npy", image)
            np.save(cloudy_dir / f"{sample_id}.npy", cloud.cloudy)
            np.save(mask_dir / f"{sample_id}.npy", cloud.mask)
            if base_id in train_bases:
                train_ids.append(sample_id)
            elif base_id in val_bases:
                val_ids.append(sample_id)
            else:
                test_ids.append(sample_id)

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(train_ids) + len(val_ids) + len(test_ids)} samples to {root}")


if __name__ == "__main__":
    main()
