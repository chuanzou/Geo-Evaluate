from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_removal.training.evaluate import evaluate_method
from cloud_removal.training.train_diffusion import train_diffusion
from cloud_removal.training.train_gan import train_gan
from cloud_removal.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cloud removal benchmark experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["temporal", "gan", "diffusion"], required=True)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])

    if args.train and args.method == "gan":
        checkpoint = train_gan(config, device)
        print(f"Saved GAN checkpoint: {checkpoint}")
    elif args.train and args.method == "diffusion":
        checkpoint = train_diffusion(config, device)
        print(f"Saved diffusion checkpoint: {checkpoint}")
    elif args.train and args.method == "temporal":
        print("Temporal baseline has no trainable parameters.")

    metrics_path = evaluate_method(config, args.method, device)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
