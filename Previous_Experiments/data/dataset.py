from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KathakaliDataset(Dataset):
    def __init__(self, root, split, manifest_path=None):
        self.root = Path(root)

        if manifest_path is None:
            manifest_path = self.root / "dataset_splits" / "corrected_split_manifest.csv"

        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.classes = sorted(pd.read_csv(manifest_path)["label"].unique())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        assert len(self.df) > 0, f"No samples found for split: {split}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self.root / row["file"]

        with np.load(file_path) as data:
            keypoints = data["keypoints"].astype(np.float32)
            scores = np.clip(data["scores"], 0.0, 1.0).astype(np.float32)

        features = np.concatenate(
            [keypoints, scores[..., None]], axis=-1
        )

        # (T, V, C) -> (C, T, V)
        features = torch.from_numpy(features).permute(2, 0, 1).contiguous()

        label = self.class_to_idx[row["label"]]
        return features, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    for split in ["train", "validation", "test"]:
        dataset = KathakaliDataset(root, split)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        x, y = next(iter(loader))

        print(f"\n{split.upper()}")
        print("Samples:", len(dataset))
        print("Batch shape:", tuple(x.shape))
        print("Labels shape:", tuple(y.shape))
        print("Feature range:", x.min().item(), x.max().item())

    print("\nClasses:", KathakaliDataset(root, "train").classes)
