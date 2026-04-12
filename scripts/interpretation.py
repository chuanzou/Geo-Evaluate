from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

matplotlib_cache = Path("outputs") / ".matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_removal.data.dataset import CloudRemovalDataset
from cloud_removal.data.synthetic_clouds import apply_cloud
from cloud_removal.models.baseline import multi_temporal_composite
from cloud_removal.models.diffusion import InpaintingDiffusion
from cloud_removal.models.unet import UNetGenerator
from cloud_removal.utils.config import ensure_dir, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative interpretation figures.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def rgb(image: np.ndarray) -> np.ndarray:
    """Convert CxHxW red/green/blue/nir image to displayable HxWx3 RGB."""
    return np.clip(np.moveaxis(image[:3], 0, -1), 0.0, 1.0)


def mask_display(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask.squeeze(), 0.0, 1.0)


def load_state_dict(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    return state.get("model_state_dict", state)


def maybe_load_gan(config: dict, device: torch.device) -> UNetGenerator | None:
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    checkpoint = checkpoint_dir / "gan_generator_best.pt"
    if not checkpoint.exists():
        checkpoint = checkpoint_dir / "gan_generator.pt"
    if not checkpoint.exists():
        print("GAN checkpoint not found. Skipping GAN qualitative output.")
        return None

    model = UNetGenerator(in_channels=5, out_channels=4).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()
    print(f"Loaded GAN checkpoint: {checkpoint}")
    return model


def maybe_load_diffusion(config: dict, device: torch.device) -> InpaintingDiffusion | None:
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    checkpoint = checkpoint_dir / "diffusion_best.pt"
    if not checkpoint.exists():
        checkpoint = checkpoint_dir / "diffusion.pt"
    if not checkpoint.exists():
        print("Diffusion checkpoint not found. Skipping diffusion qualitative output.")
        return None

    model = InpaintingDiffusion(timesteps=int(config["training"]["diffusion_timesteps"])).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()
    print(f"Loaded diffusion checkpoint: {checkpoint}")
    return model


def sample_ids_by_coverage(dataset: CloudRemovalDataset, coverages: list[float], limit: int) -> list[int]:
    selected: list[int] = []
    used_bases: set[str] = set()
    for target_coverage in coverages:
        candidates: list[tuple[float, int, str]] = []
        for index, sample_id in enumerate(dataset.ids):
            if "_c" not in sample_id:
                continue
            base_id, coverage_text = sample_id.rsplit("_c", 1)
            try:
                coverage = int(coverage_text) / 100
            except ValueError:
                continue
            candidates.append((abs(coverage - target_coverage), index, base_id))
        candidates.sort()
        for _, index, base_id in candidates:
            if base_id not in used_bases or len(used_bases) >= limit:
                selected.append(index)
                used_bases.add(base_id)
                break
    return selected


def representative_sample_groups(dataset: CloudRemovalDataset, num_samples: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(dataset.ids):
        if "_c" not in sample_id:
            continue
        base_id, _ = sample_id.rsplit("_c", 1)
        groups.setdefault(base_id, []).append(index)

    selected: dict[str, list[int]] = {}
    for base_id in sorted(groups):
        indices = sorted(groups[base_id], key=lambda i: dataset.ids[i])
        if len(indices) >= 2:
            selected[base_id] = indices
        if len(selected) >= num_samples:
            break
    return selected


def predict_temporal(batch: dict, config: dict, device: torch.device) -> torch.Tensor:
    cloudy = batch["cloudy"].unsqueeze(0).to(device)
    mask = batch["mask"].unsqueeze(0).to(device)
    target = batch["target"].unsqueeze(0).to(device)
    target_np = target.squeeze(0).cpu().numpy()
    coverage = float(batch["coverage"])
    digest = hashlib.sha256(str(batch["id"]).encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))

    temporal_images = [cloudy.squeeze(0).cpu()]
    temporal_masks = [mask.squeeze(0).cpu()]
    for _ in range(int(config["data"]["temporal_neighbors"])):
        observation = apply_cloud(target_np, coverage, rng)
        temporal_images.append(torch.from_numpy(observation.cloudy))
        temporal_masks.append(torch.from_numpy(observation.mask[None, ...]))

    stack = torch.stack(temporal_images).to(device)
    masks = torch.stack(temporal_masks).to(device)
    return multi_temporal_composite(stack, masks, fallback=cloudy.squeeze(0)).cpu()


@torch.no_grad()
def predict_models(
    batch: dict,
    config: dict,
    device: torch.device,
    gan: UNetGenerator | None,
    diffusion: InpaintingDiffusion | None,
    diffusion_steps: int,
) -> dict[str, np.ndarray]:
    cloudy = batch["cloudy"].unsqueeze(0).to(device)
    mask = batch["mask"].unsqueeze(0).to(device)
    model_input = batch["input"].unsqueeze(0).to(device)

    predictions = {
        "Temporal": predict_temporal(batch, config, device).numpy(),
    }
    if gan is not None:
        gan_prediction = gan(model_input)
        gan_prediction = gan_prediction * mask + cloudy * (1.0 - mask)
        predictions["GAN best"] = gan_prediction.squeeze(0).cpu().numpy()
    if diffusion is not None:
        diffusion_prediction = diffusion.inpaint(cloudy, mask, steps=diffusion_steps)
        predictions["Diffusion best"] = diffusion_prediction.squeeze(0).cpu().numpy()
    return predictions


def save_cloud_coverage_figure(dataset: CloudRemovalDataset, config: dict, output_dir: Path) -> Path:
    coverages = [float(value) for value in config["data"]["cloud_coverages"]]
    indices = sample_ids_by_coverage(dataset, coverages, limit=len(coverages))
    if not indices:
        raise RuntimeError("No samples found for cloud coverage figure.")

    fig, axes = plt.subplots(3, len(indices), figsize=(3 * len(indices), 8), constrained_layout=True)
    if len(indices) == 1:
        axes = axes.reshape(3, 1)

    for column, index in enumerate(indices):
        sample = dataset[index]
        coverage = float(sample["coverage"])
        axes[0, column].imshow(rgb(sample["cloudy"].numpy()))
        axes[0, column].set_title(f"Cloudy\n{coverage:.0%}")
        axes[1, column].imshow(mask_display(sample["mask"].numpy()), cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title("Mask")
        axes[2, column].imshow(rgb(sample["target"].numpy()))
        axes[2, column].set_title("Ground truth")

    for ax in axes.flat:
        ax.axis("off")

    output_path = output_dir / "cloud_coverage_examples.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_model_completion_figures(
    dataset: CloudRemovalDataset,
    config: dict,
    output_dir: Path,
    device: torch.device,
    num_samples: int,
    diffusion_steps: int,
) -> list[Path]:
    gan = maybe_load_gan(config, device)
    diffusion = maybe_load_diffusion(config, device)
    groups = representative_sample_groups(dataset, num_samples)
    output_paths: list[Path] = []

    for figure_index, (base_id, indices) in enumerate(groups.items(), start=1):
        chosen_index = indices[-1]
        sample = dataset[chosen_index]
        predictions = predict_models(sample, config, device, gan, diffusion, diffusion_steps)
        columns = ["Cloudy", "Mask", *predictions.keys(), "Ground truth"]

        fig, axes = plt.subplots(1, len(columns), figsize=(3 * len(columns), 3.4), constrained_layout=True)
        axes[0].imshow(rgb(sample["cloudy"].numpy()))
        axes[0].set_title(f"Cloudy\n{float(sample['coverage']):.0%}")
        axes[1].imshow(mask_display(sample["mask"].numpy()), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Mask")

        for offset, (name, prediction) in enumerate(predictions.items(), start=2):
            axes[offset].imshow(rgb(prediction))
            axes[offset].set_title(name)

        axes[-1].imshow(rgb(sample["target"].numpy()))
        axes[-1].set_title("Ground truth")
        for ax in axes:
            ax.axis("off")

        output_path = output_dir / f"model_completions_{figure_index:02d}_{base_id}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])
    output_dir = ensure_dir(args.output_dir or Path(config["outputs"]["figure_dir"]) / "interpretation")

    dataset = CloudRemovalDataset(config["data"]["root"], args.split)
    cloud_path = save_cloud_coverage_figure(dataset, config, output_dir)
    completion_paths = save_model_completion_figures(
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        device=device,
        num_samples=args.num_samples,
        diffusion_steps=args.diffusion_steps,
    )

    print(f"Saved cloud coverage figure: {cloud_path}")
    for path in completion_paths:
        print(f"Saved model completion figure: {path}")


if __name__ == "__main__":
    main()
