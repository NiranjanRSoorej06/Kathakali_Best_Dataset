from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BASE = Path(__file__).resolve().parents[2]
DATA_FILE = BASE / "datasets/processed_exp9/exp9_xyc_38f.npz"


class KathakaliDataset(Dataset):
    def __init__(self, split):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        if not DATA_FILE.exists():
            raise FileNotFoundError(DATA_FILE)

        with np.load(DATA_FILE, allow_pickle=False) as data:
            selected = data["split"] == split

            self.X = data["X"][selected].copy()
            self.frame_mask = data["frame_mask"][selected].copy()
            self.labels = data["labels"][selected].copy()
            self.class_names = data["class_names"].copy()
            self.video_names = data["video_names"][selected].copy()

        # Input: (N, T, V, C) -> ST-GCN: (N, C, T, V)
        self.X = self.X.transpose(0, 3, 1, 2).astype(np.float32)

        if self.X.shape[1:] != (3, 38, 50):
            raise ValueError(f"Unexpected input shape: {self.X.shape}")

        if not np.isfinite(self.X).all():
            raise ValueError("Dataset contains non-finite input values")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.from_numpy(self.frame_mask[index]),
        )


if __name__ == "__main__":
    for split in ("train", "validation", "test"):
        ds = KathakaliDataset(split)
        x, y, mask = ds[0]

        print(
            f"{split:10s} | samples={len(ds):3d} "
            f"| input={tuple(x.shape)} "
            f"| label={y.item()} "
            f"| valid_frames={int(mask.sum())}"
        )

    print("\nExperiment 9 dataset loader validation passed.")
