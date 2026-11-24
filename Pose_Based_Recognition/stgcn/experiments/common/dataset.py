from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "datasets/processed/metadata_split.csv"

class KathakaliDataset(Dataset):
    def __init__(self, split, in_channels):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        if in_channels not in (2, 3):
            raise ValueError("in_channels must be 2 or 3")

        self.df = pd.read_csv(METADATA)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.in_channels = in_channels

        if self.df.empty:
            raise ValueError(f"No samples for split {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        rel = str(row["feature_path"])
        path = ROOT / rel

        if not path.exists():
            rel = rel.replace("stgcn/processed/", "datasets/processed/", 1)
            path = ROOT / rel

        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            frame_mask = data["frame_valid"].astype(bool)

            if int(data["label_index"].item()) != int(row["label_index"]):
                raise ValueError(f"Label mismatch: {path}")

        if features.shape != (100, 50, 3):
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features: {path}")
        if frame_mask.shape != (100,) or not frame_mask.any():
            raise ValueError(f"Invalid frame mask: {path}")

        x = torch.from_numpy(
            features[:, :, :self.in_channels]
        ).permute(2, 0, 1).contiguous()

        return (
            x,
            torch.tensor(int(row["label_index"]), dtype=torch.long),
            torch.from_numpy(frame_mask.copy())
        )
