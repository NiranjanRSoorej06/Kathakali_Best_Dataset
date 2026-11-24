import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = Path(r"/home/niranjanr/CLC/22-11-26_S5_break/Video_classification_Kathakali/DGNN_Audit/Canonical_Preprocessing/v1/graph_sequences_xy.npz")


class KathakaliDataset(Dataset):
    def __init__(self, split):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        with np.load(DATA_FILE) as data:
            selected = data["split"] == split

            self.X = data["X"][selected].copy()
            self.frame_mask = data["frame_mask"][selected].copy()
            self.joint_mask = data["joint_coordinate_mask"][selected].copy()
            self.labels = data["class_id"][selected].copy()
            self.class_names = data["class_name"][selected].copy()
            self.lengths = data["lengths"][selected].copy()

        # Replace unavailable joint coordinates with zero.
        self.X[~self.joint_mask] = 0.0

        # ST-GCN format: (N, C, T, V)
        self.X = self.X.transpose(0, 3, 1, 2).astype(np.float32)

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

    print("\nDataset loader validation passed.")
