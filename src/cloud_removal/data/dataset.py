from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CloudRemovalDataset(Dataset):
    """Dataset for paired cloudy images, masks, and cloud-free targets."""

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        sample_id = self.ids[index]
        gt = np.load(self.root / "ground_truth" / f"{sample_id}.npy").astype(np.float32)
        cloudy = np.load(self.root / "cloudy" / f"{sample_id}.npy").astype(np.float32)
        mask = np.load(self.root / "masks" / f"{sample_id}.npy").astype(np.float32)

        if mask.ndim == 2:
            mask = mask[None, ...]

        model_input = np.concatenate([cloudy, mask], axis=0)
        return {
            "id": sample_id,
            "input": torch.from_numpy(model_input),
            "cloudy": torch.from_numpy(cloudy),
            "mask": torch.from_numpy(mask),
            "target": torch.from_numpy(gt),
            "coverage": float(mask.mean()),
        }
