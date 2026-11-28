import json
import csv
import numpy as np
from pathlib import Path

BASE = Path.home() / "Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40"

INPUT_DIR = BASE / "raw_pose"
OUTPUT_DIR = BASE / "processed"

EXPECTED_JOINTS = 50
EXPECTED_FRAMES = 100

NECK = 1
RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5

MIN_REFERENCE_FRAMES = 50

CLASSES = [
    'Aakasham', 'Aana', 'Aayudham', 'Acchan', 'Albutham',
    'Amma', 'Anugraham', 'Ariyikkuka', 'Baalan', 'Bhoomi',
    'Bhrahmaavu', 'Chandran', 'Chundu', 'Gunam', 'Irikkuka',
    'Kaadu', 'Kallan', 'Kannu', 'Kashtam', 'Kuyil',
    'Manassu', 'Nadakkuka', 'Nee_Roudram', 'Nee_Shringaaram',
    'Paadam', 'Pazham', 'Pushpam', 'Raajyam', 'Samudram',
    'Satyam', 'Shivan', 'Simham', 'Sooryan', 'Svargam',
    'Thaamara', 'Thalamudi', 'Thaliru', 'Urakkam', 'Veedu',
    'Vishnu', 'Vriksham'
]

CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}


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
        frame_indices = data["frame_indices"]

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

    if label not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown class label: {label}")

    # Convert normalized image coordinates back to pixels.
    pixel_points = keypoints.copy()
    pixel_points[:, :, 0] *= width
    pixel_points[:, :, 1] *= height

    # Identify frames with reliable neck and shoulder detections.
    reference_points = pixel_points[:, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER], :]
    reference_scores = scores[:, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER]]

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
        frame_indices=frame_indices,
        label=np.asarray(label),
        label_index=np.asarray(CLASS_TO_INDEX[label], dtype=np.int64),
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
        "video_name": video_name,
        "label": label,
        "label_index": CLASS_TO_INDEX[label],
        "reference_frames": int(len(usable_widths)),
        "valid_frames": int(frame_valid.sum()),
        "normalization_valid_frames": int(normalization_valid.sum()),
        "reference_shoulder_width": reference_shoulder_width,
        "output": str(destination)
    }


def main():
    files = sorted(INPUT_DIR.rglob("*.npz"))

    if len(files) != 40:
        raise RuntimeError(f"Expected 40 NPZ files, found {len(files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    print("=" * 60)
    print("NEW40 ST-GCN PREPROCESSING")
    print("=" * 60)
    print("Input videos:", len(files))
    print("Classes:", len(CLASSES))
    print("Output:", OUTPUT_DIR)

    for i, source in enumerate(files, 1):
        destination = OUTPUT_DIR / source.relative_to(INPUT_DIR)

        try:
            result = process_file(source, destination)
            results.append(result)

            print(
                f"[{i}/40] OK {result['video_name']} "
                f"| {result['label']} "
                f"| reference frames: {result['reference_frames']}"
            )

        except Exception as exc:
            errors.append({
                "file": str(source),
                "error": repr(exc)
            })
            print(f"[{i}/40] ERROR {source.name}: {exc}")

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
        writer.writerows(results)

    summary = {
        "input_videos": len(files),
        "processed_videos": len(results),
        "failed_videos": len(errors),
        "number_of_model_classes": len(CLASSES),
        "class_mapping": CLASS_TO_INDEX,
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
            "minimum_reference_frames": MIN_REFERENCE_FRAMES
        },
        "errors": errors
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("Processed:", len(results))
    print("Failed:", len(errors))
    print("Output:", OUTPUT_DIR)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
