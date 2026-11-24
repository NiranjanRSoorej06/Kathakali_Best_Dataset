from pathlib import Path
import re
import json
import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "dwpose/extracted_430/DWPose_50J_100F_41Classes"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "datasets/processed_exp9"
OUTPUT_FILE = OUTPUT_DIR / "exp9_xyc_38f.npz"
MANIFEST_FILE = OUTPUT_DIR / "split_manifest.json"

EXPECTED_VIDEOS = 430
EXPECTED_CLASSES = 41
SOURCE_FRAMES = 100
TARGET_FRAMES = 38
JOINTS = 50

if OUTPUT_FILE.exists() or MANIFEST_FILE.exists():
    raise FileExistsError(
        "Experiment 9 processed output already exists. Refusing to overwrite."
    )

if not SOURCE.is_dir():
    raise FileNotFoundError(SOURCE)

files = sorted(SOURCE.rglob("*.npz"))
if len(files) != EXPECTED_VIDEOS:
    raise RuntimeError(f"Expected 430 NPZ files, found {len(files)}")

# Use one consistent set of 38 evenly spaced source frames.
sample_indices = np.rint(
    np.linspace(0, SOURCE_FRAMES - 1, TARGET_FRAMES)
).astype(np.int64)

if len(np.unique(sample_indices)) != TARGET_FRAMES:
    raise RuntimeError("Frame sampling produced duplicate indices")

records = []
class_names = set()
errors = []

for path in files:
    try:
        match = re.search(r"_(\d+)\.npz$", path.name)
        if not match:
            raise ValueError("Filename has no video index")

        video_index = int(match.group(1))

        with np.load(path, allow_pickle=False) as data:
            xy = data["keypoints"]
            confidence = data["scores"]
            frame_valid = data["frame_valid"]
            label = str(data["label"].item())
            video_name = str(data["video_name"].item())

        if xy.shape != (100, 50, 2):
            raise ValueError(f"Unexpected XY shape: {xy.shape}")
        if confidence.shape != (100, 50):
            raise ValueError(f"Unexpected confidence shape: {confidence.shape}")
        if frame_valid.shape != (100,):
            raise ValueError(f"Unexpected frame mask shape: {frame_valid.shape}")

        if not np.isfinite(xy).all():
            raise ValueError("Non-finite XY coordinates")
        if not np.isfinite(confidence).all():
            raise ValueError("Non-finite confidence values")

        # Preserve the original confidence values without clipping/rescaling.
        sampled_xy = xy[sample_indices]
        sampled_conf = confidence[sample_indices]
        sampled_mask = frame_valid[sample_indices].astype(bool)

        # Input layout before transpose: (T, V, C)
        x = np.concatenate(
            [sampled_xy, sampled_conf[..., None]], axis=-1
        ).astype(np.float32)

        # Invalid frames are zeroed and masked.
        x[~sampled_mask] = 0.0

        if video_index in range(1, 8) or video_index == 11:
            split = "train"
        elif video_index == 8:
            split = "validation"
        elif video_index in (9, 10):
            split = "test"
        else:
            raise ValueError(f"Unexpected video index: {video_index}")

        class_names.add(label)
        records.append({
            "x": x,
            "mask": sampled_mask,
            "label": label,
            "video_name": video_name,
            "source_file": str(path),
            "video_index": video_index,
            "split": split,
        })

    except Exception as exc:
        errors.append((str(path), str(exc)))

if errors:
    for item in errors[:20]:
        print("ERROR:", item)
    raise RuntimeError(f"{len(errors)} source files failed validation")

if len(class_names) != EXPECTED_CLASSES:
    raise RuntimeError(f"Expected 41 classes, found {len(class_names)}")

records.sort(key=lambda r: (r["label"], r["video_index"], r["video_name"]))
classes = sorted(class_names)
class_to_id = {name: i for i, name in enumerate(classes)}

X = np.stack([r["x"] for r in records]).astype(np.float32)
frame_mask = np.stack([r["mask"] for r in records]).astype(bool)
labels = np.asarray([class_to_id[r["label"]] for r in records], dtype=np.int64)
names = np.asarray([r["label"] for r in records])
video_names = np.asarray([r["video_name"] for r in records])
source_files = np.asarray([r["source_file"] for r in records])
video_indices = np.asarray([r["video_index"] for r in records], dtype=np.int64)
splits = np.asarray([r["split"] for r in records])

if X.shape != (430, 38, 50, 3):
    raise RuntimeError(f"Unexpected final X shape: {X.shape}")

split_counts = {
    split: int(np.sum(splits == split))
    for split in ("train", "validation", "test")
}
expected_counts = {"train": 307, "validation": 41, "test": 82}
if split_counts != expected_counts:
    raise RuntimeError(f"Unexpected split counts: {split_counts}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    frame_mask=frame_mask,
    labels=labels,
    class_names=np.asarray(classes),
    video_names=video_names,
    source_files=source_files,
    video_indices=video_indices,
    split=splits,
    sampled_frame_indices=sample_indices,
)

confidence = X[..., 2]
manifest = {
    "experiment": 9,
    "source": str(SOURCE),
    "output": str(OUTPUT_FILE),
    "videos": len(records),
    "classes": classes,
    "class_to_id": class_to_id,
    "input_shape": list(X.shape),
    "channels": ["x", "y", "confidence"],
    "confidence_handling": "Raw values preserved; no clipping or rescaling",
    "source_frames": SOURCE_FRAMES,
    "sampled_frames": TARGET_FRAMES,
    "sampled_frame_indices": sample_indices.tolist(),
    "split_rule": {
        "train": "video indices 1-7 and 11",
        "validation": "video index 8",
        "test": "video indices 9-10",
    },
    "split_counts": split_counts,
    "confidence_min": float(confidence.min()),
    "confidence_max": float(confidence.max()),
    "confidence_mean": float(confidence.mean()),
}

with open(MANIFEST_FILE, "w") as f:
    json.dump(manifest, f, indent=2)

print("Experiment 9 preprocessing completed.")
print("Dataset:", OUTPUT_FILE)
print("Manifest:", MANIFEST_FILE)
print("X shape:", X.shape)
print("Frame mask:", frame_mask.shape)
print("Classes:", len(classes))
print("Split counts:", split_counts)
print("Confidence min/max/mean:",
      confidence.min(), confidence.max(), confidence.mean())
print("Sampled frame indices:", sample_indices.tolist())
