from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "datasets/processed/metadata_split.csv"

class KathakaliDataset(Dataset):
    def __init__(self, split):
        self.df = pd.read_csv(METADATA)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = str(row["feature_path"])
        feature_path = feature_path.replace(
            "stgcn/processed/", "datasets/processed/", 1
        )
        path = ROOT / feature_path

        if not path.is_file():
            raise FileNotFoundError(f"Feature file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)

        if features.shape != (100, 50, 3):
            raise ValueError(f"Unexpected feature shape {features.shape}: {path}")

        # Keep X, Y, and confidence (3 input channels).
        features = features[:, :, :3]

        # (T,V,C) -> (C,T,V,M)
        x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(-1)
        y = int(row["label_index"])
        return x, y
