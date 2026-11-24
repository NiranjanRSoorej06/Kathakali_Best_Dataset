import json
import csv
import argparse
import numpy as np

from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE.parent / "dwpose/extracted_430/"
    "DWPose_50J_100F_41Classes"
)

OUTPUT_DIR = (
    BASE / "datasets/processed/"
    "normalized_50J_100F"
)

EXPECTED_JOINTS = 50
EXPECTED_FRAMES = 100

# Positions in the selected 50-joint representation
NECK = 1
RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5

MIN_REFERENCE_FRAMES = 50


def process_file(source, destination):
    with np.load(source) as data:
        keypoints = data["keypoints"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        frame_valid = data["frame_valid"].astype(bool)
        joint_indices = data["joint_indices"]

        label = str(data["label"].item())
        video_name = str(data["video_name"].item())

        width = int(data["original_width"])
        height = int(data["original_height"])

        fps = float(data["fps"])
        total_frames = int(data["total_frames"])

    if keypoints.shape != (EXPECTED_FRAMES, EXPECTED_JOINTS, 2):
        raise ValueError(f"Unexpected keypoint shape: {source}")

    if scores.shape != (EXPECTED_FRAMES, EXPECTED_JOINTS):
        raise ValueError(f"Unexpected score shape: {source}")

    if len(joint_indices) != EXPECTED_JOINTS:
        raise ValueError(f"Unexpected joint mapping: {source}")

    if not np.isfinite(keypoints).all():
        raise ValueError(f"Non-finite coordinates: {source}")

    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite scores: {source}")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {source}")

    # Convert normalized image coordinates back to pixels.
    pixel_points = keypoints.copy()
    pixel_points[:, :, 0] *= width
    pixel_points[:, :, 1] *= height

    # Identify frames with reliable neck and shoulder detections.
    reference_points = pixel_points[
        :, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER], :
    ]

    reference_scores = scores[
        :, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER]
    ]

    reference_valid = (
        frame_valid
        & np.all(reference_scores > 0, axis=1)
        & np.all(np.isfinite(reference_points), axis=(1, 2))
    )

    shoulder_widths = np.linalg.norm(
        reference_points[:, 1, :] - reference_points[:, 2, :],
        axis=1
    )

    reference_valid &= (
        np.isfinite(shoulder_widths)
        & (shoulder_widths > 1.0)
    )

    usable_widths = shoulder_widths[reference_valid]

    if len(usable_widths) < MIN_REFERENCE_FRAMES:
        raise ValueError(
            f"Insufficient reliable shoulder frames: "
            f"{source} ({len(usable_widths)})"
        )

    # One robust scale value per video.
    reference_shoulder_width = float(np.median(usable_widths))

    if not np.isfinite(reference_shoulder_width):
        raise ValueError(f"Invalid reference width: {source}")

    # A frame can be normalized if its neck is detected.
    normalization_valid = (
        frame_valid
        & (scores[:, NECK] > 0)
        & np.all(np.isfinite(pixel_points[:, NECK, :]), axis=1)
    )

    # Neck-centred and shoulder-scaled coordinates.
    normalized = np.zeros_like(pixel_points, dtype=np.float32)

    for t in np.where(normalization_valid)[0]:
        normalized[t] = (
            pixel_points[t] - pixel_points[t, NECK]
        ) / reference_shoulder_width

    # Preserve raw SimCC scores as the third feature channel.
    features = np.concatenate(
        [normalized, scores[:, :, None]],
        axis=2
    ).astype(np.float32)

    # Invalid normalization frames have zero coordinates.
    # Original scores and original frame-validity are preserved.
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite output features: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination,
        features=features,
        keypoints_normalized=normalized,
        scores=scores,
        frame_valid=frame_valid,
        normalization_valid=normalization_valid,
        frame_indices=data_frame_indices(source),
        label=np.asarray(label),
        label_index=np.asarray(-1, dtype=np.int64),
        video_name=np.asarray(video_name),
        joint_indices=joint_indices,
        reference_shoulder_width=np.asarray(
            reference_shoulder_width, dtype=np.float32
        ),
        reference_frame_count=np.asarray(
            len(usable_widths), dtype=np.int32
        ),
        original_width=np.asarray(width, dtype=np.int32),
        original_height=np.asarray(height, dtype=np.int32),
        fps=np.asarray(fps, dtype=np.float32),
        total_frames=np.asarray(total_frames, dtype=np.int64)
    )

    return {
        "label": label,
        "video_name": video_name,
        "reference_shoulder_width": reference_shoulder_width,
        "reference_frames": int(len(usable_widths)),
        "valid_frames": int(frame_valid.sum()),
        "normalization_valid_frames": int(normalization_valid.sum()),
        "output": str(destination.relative_to(OUTPUT_DIR))
    }


def data_frame_indices(source):
    with np.load(source) as data:
        return data["frame_indices"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing processed dataset"
    )
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    files = sorted(INPUT_DIR.rglob("*.npz"))

    if not files:
        raise RuntimeError("No input NPZ files found")

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {OUTPUT_DIR}\n"
                "Use --overwrite only if you intend to replace it."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a deterministic class mapping.
    labels = set()

    for source in files:
        with np.load(source) as data:
            labels.add(str(data["label"].item()))

    classes = sorted(labels)
    class_to_index = {
        label: index for index, label in enumerate(classes)
    }

    print("=" * 65)
    print("ST-GCN DATA PREPROCESSING")
    print("=" * 65)
    print("Input videos:", len(files))
    print("Classes:", len(classes))
    print("Output:", OUTPUT_DIR)
    print()

    results = []
    errors = []

    for i, source in enumerate(files, 1):
        destination = OUTPUT_DIR / source.relative_to(INPUT_DIR)

        try:
            result = process_file(source, destination)

            # Store the deterministic class index.
            label_index = class_to_index[result["label"]]

            with np.load(destination) as data:
                saved = {
                    key: data[key]
                    for key in data.files
                }

            saved["label_index"] = np.asarray(
                label_index, dtype=np.int64
            )

            np.savez_compressed(destination, **saved)

            results.append(result)

            print(
                f"[{i}/{len(files)}] OK "
                f"{result['video_name']} "
                f"| {result['label']}"
            )

        except Exception as exc:
            errors.append((str(source), str(exc)))
            print(f"[{i}/{len(files)}] ERROR {source.name}: {exc}")

    # Class mapping.
    with open(OUTPUT_DIR / "class_to_index.json", "w") as f:
        json.dump(class_to_index, f, indent=2)

    # Class distribution.
    distribution = Counter(
        result["label"] for result in results
    )

    with open(OUTPUT_DIR / "class_distribution.json", "w") as f:
        json.dump(dict(sorted(distribution.items())), f, indent=2)

    # Video manifest.
    manifest_path = OUTPUT_DIR / "manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "label",
                "label_index",
                "reference_shoulder_width",
                "reference_frames",
                "valid_frames",
                "normalization_valid_frames",
                "output"
            ]
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                **result,
                "label_index": class_to_index[result["label"]]
            })

    # Processing summary.
    summary = {
        "input_videos": len(files),
        "processed_videos": len(results),
        "failed_videos": len(errors),
        "number_of_classes": len(classes),
        "classes": classes,
        "feature_shape_per_video": [100, 50, 3],
        "stgcn_layout_per_video": [3, 100, 50, 1],
        "feature_channels": [
            "neck_centered_shoulder_scaled_x",
            "neck_centered_shoulder_scaled_y",
            "raw_dwpose_simcc_score"
        ],
        "normalization": {
            "origin": "neck",
            "scale": "median shoulder width per video",
            "shoulder_positions": [
                RIGHT_SHOULDER,
                LEFT_SHOULDER
            ],
            "minimum_reference_frames": MIN_REFERENCE_FRAMES
        },
        "score_handling": "Original raw SimCC scores preserved",
        "original_data_modified": False,
        "class_distribution": dict(sorted(distribution.items())),
        "errors": errors
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 65)
    print("PREPROCESSING COMPLETE")
    print("=" * 65)
    print("Input videos:", len(files))
    print("Successfully processed:", len(results))
    print("Failed:", len(errors))
    print("Classes:", len(classes))
    print("Output:", OUTPUT_DIR)

    if errors:
        print("\nErrors:")
        for filename, error in errors[:20]:
            print(filename, error)

    print("\nClass distribution:")
    for label, count in sorted(distribution.items()):
        print(f"{label}: {count}")

    print("=" * 65)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
