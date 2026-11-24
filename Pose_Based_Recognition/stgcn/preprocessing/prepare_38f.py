from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_METADATA = ROOT / "datasets/processed/metadata_split.csv"
OUTPUT_DIR = ROOT / "datasets/processed_38f"
FEATURE_DIR = OUTPUT_DIR / "features"

FRAMES = 38
EXPECTED_SOURCE_SHAPE = (100, 50, 3)

def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"{OUTPUT_DIR} is not empty. Refusing to overwrite."
        )

    df = pd.read_csv(SOURCE_METADATA)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Select 38 unique, evenly spaced positions from 100 frames.
    indices = np.rint(
        np.linspace(0, 99, FRAMES)
    ).astype(int)

    assert len(indices) == FRAMES
    assert len(np.unique(indices)) == FRAMES

    print("Selected frame indices:", indices.tolist())
    print("Input samples:", len(df))

    new_paths = []

    for i, row in df.iterrows():
        source = ROOT / row["feature_path"]

        with np.load(source, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)

        if features.shape != EXPECTED_SOURCE_SHAPE:
            raise ValueError(
                f"Unexpected shape {features.shape}: {source}"
            )

        sampled = features[indices]

        if sampled.shape != (FRAMES, 50, 3):
            raise ValueError(f"Incorrect output shape: {sampled.shape}")

        relative_path = (
            Path("datasets/processed_38f/features")
            / str(row["class_name"])
            / f"{Path(row['video_name']).stem}.npz"
        )

        destination = ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            destination,
            features=sampled,
            sampled_indices=indices
        )

        new_paths.append(str(relative_path))

        if (i + 1) % 50 == 0 or i + 1 == len(df):
            print(f"Processed {i + 1}/{len(df)}")

    # Preserve the original split and labels.
    df["feature_path"] = new_paths
    df["num_frames"] = FRAMES

    df.to_csv(OUTPUT_DIR / "metadata_split.csv", index=False)

    print("\n38-frame preprocessing complete.")
    print("Samples:", len(df))
    print("Split counts:")
    print(df["split"].value_counts())
    print("Output:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
