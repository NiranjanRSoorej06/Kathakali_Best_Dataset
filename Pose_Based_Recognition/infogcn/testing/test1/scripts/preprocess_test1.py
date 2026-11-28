"""Canonical EXP10/ST-GCN skeleton preprocessing for testing/test1 raw DWPose files."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_test1 import (  # noqa: E402
    EXPECTED_FRAMES,
    EXPECTED_JOINTS,
    EXPECTED_VIDEOS,
    JOINT_INDICES,
    LEFT_SHOULDER,
    MIN_REFERENCE_FRAMES,
    NECK,
    PROCESSED_DIR,
    RAW_POSE_DIR,
    RIGHT_SHOULDER,
    load_canonical_class_names,
)


def process_raw_pose_file(
    source: Path,
    destination: Path,
    class_to_index: dict[str, int],
    overwrite: bool = False,
) -> dict:
    """Apply exact canonical neck-centering and median-shoulder scaling to one raw pose NPZ."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing processed file: {destination}")

    with np.load(source, allow_pickle=False) as data:
        keypoints = data["keypoints"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        frame_valid = data["frame_valid"].astype(bool)
        joint_indices = data["joint_indices"]
        frame_indices = data["frame_indices"]
        label = str(data["label"].item())
        video_name = str(data["video_name"].item())
        width = int(data["original_width"])
        height = int(data["original_height"])
        fps = float(data["fps"])
        total_frames = int(data["total_frames"])

    if keypoints.shape != (EXPECTED_FRAMES, EXPECTED_JOINTS, 2):
        raise ValueError(f"Unexpected keypoint shape {keypoints.shape}: {source}")
    if scores.shape != (EXPECTED_FRAMES, EXPECTED_JOINTS):
        raise ValueError(f"Unexpected score shape {scores.shape}: {source}")
    if frame_valid.shape != (EXPECTED_FRAMES,):
        raise ValueError(f"Unexpected frame_valid shape {frame_valid.shape}: {source}")
    if not np.array_equal(joint_indices, JOINT_INDICES):
        raise ValueError(f"Unexpected joint_indices mapping: {source}")
    if not np.isfinite(keypoints).all():
        raise ValueError(f"Non-finite coordinates: {source}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite scores: {source}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions ({width}x{height}): {source}")
    if label not in class_to_index:
        raise ValueError(f"Unknown canonical class label '{label}': {source}")

    # Convert normalized image coordinates back to pixels (matching canonical preprocess_stgcn.py).
    pixel_points = keypoints.copy()
    pixel_points[:, :, 0] *= width
    pixel_points[:, :, 1] *= height

    reference_points = pixel_points[:, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER], :]
    reference_scores = scores[:, [NECK, RIGHT_SHOULDER, LEFT_SHOULDER]]

    reference_valid = (
        frame_valid
        & np.all(reference_scores > 0, axis=1)
        & np.all(np.isfinite(reference_points), axis=(1, 2))
    )
    shoulder_widths = np.linalg.norm(
        reference_points[:, 1, :] - reference_points[:, 2, :],
        axis=1,
    )
    reference_valid &= np.isfinite(shoulder_widths) & (shoulder_widths > 1.0)
    usable_widths = shoulder_widths[reference_valid]

    if len(usable_widths) < MIN_REFERENCE_FRAMES:
        raise ValueError(
            f"Insufficient reliable shoulder frames ({len(usable_widths)} < {MIN_REFERENCE_FRAMES}): {source}"
        )

    reference_shoulder_width = float(np.median(usable_widths))
    if not np.isfinite(reference_shoulder_width) or reference_shoulder_width <= 0.0:
        raise ValueError(f"Invalid reference shoulder width ({reference_shoulder_width}): {source}")

    normalization_valid = (
        frame_valid
        & (scores[:, NECK] > 0)
        & np.all(np.isfinite(pixel_points[:, NECK, :]), axis=1)
    )

    normalized = np.zeros_like(pixel_points, dtype=np.float32)
    for t in np.where(normalization_valid)[0]:
        normalized[t] = (pixel_points[t] - pixel_points[t, NECK]) / reference_shoulder_width

    # Preserve raw DWPose SimCC scores as channel 2.
    features = np.concatenate([normalized, scores[:, :, None]], axis=2).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite output features: {source}")

    label_index = class_to_index[label]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    with open(temp_path, "wb") as f:
        np.savez_compressed(
            f,
            features=features,
            keypoints_normalized=normalized,
            scores=scores,
            frame_valid=frame_valid,
            normalization_valid=normalization_valid,
            frame_indices=frame_indices,
            label=np.asarray(label),
            label_index=np.asarray(label_index, dtype=np.int64),
            video_name=np.asarray(video_name),
            joint_indices=joint_indices,
            reference_shoulder_width=np.asarray(reference_shoulder_width, dtype=np.float32),
            reference_frame_count=np.asarray(len(usable_widths), dtype=np.int32),
            original_width=np.asarray(width, dtype=np.int32),
            original_height=np.asarray(height, dtype=np.int32),
            fps=np.asarray(fps, dtype=np.float32),
            total_frames=np.asarray(total_frames, dtype=np.int64),
        )
    os.replace(temp_path, destination)

    return {
        "video_name": video_name,
        "label": label,
        "label_index": label_index,
        "reference_shoulder_width": reference_shoulder_width,
        "reference_frames": int(len(usable_widths)),
        "valid_frames": int(frame_valid.sum()),
        "normalization_valid_frames": int(normalization_valid.sum()),
        "output": str(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical preprocessing for testing/test1")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_canonical_class_names()
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    if not RAW_POSE_DIR.exists():
        raise FileNotFoundError(f"Raw pose directory not found: {RAW_POSE_DIR}")

    raw_files = sorted(RAW_POSE_DIR.rglob("*.npz"))
    if len(raw_files) != EXPECTED_VIDEOS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEOS} raw_pose NPZ files, found {len(raw_files)}")

    manifest_path = PROCESSED_DIR / "manifest.csv"
    summary_path = PROCESSED_DIR / "preprocessing_summary.json"
    if not args.overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite existing processed manifest/summary in testing/test1. "
            "Pass --overwrite to regenerate."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []

    print("=" * 60)
    print("TEST1 CANONICAL PREPROCESSING (50J, 100F, 3C)")
    print("=" * 60)

    for idx, source in enumerate(raw_files, 1):
        destination = PROCESSED_DIR / source.relative_to(RAW_POSE_DIR)
        try:
            row = process_raw_pose_file(source, destination, class_to_index, overwrite=args.overwrite)
            results.append(row)
            print(
                f"[{idx}/{len(raw_files)}] OK {row['video_name']} | {row['label']} "
                f"| ref_frames={row['reference_frames']} | valid={row['valid_frames']}"
            )
        except Exception as exc:
            errors.append({"file": str(source), "error": repr(exc)})
            print(f"[{idx}/{len(raw_files)}] ERROR {source.name}: {exc}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
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
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "input_raw_pose_files": len(raw_files),
        "processed_videos": len(results),
        "failed_videos": len(errors),
        "number_of_canonical_classes": len(class_names),
        "class_mapping": class_to_index,
        "feature_shape_per_video": [EXPECTED_FRAMES, EXPECTED_JOINTS, 3],
        "infogcn_input_tensor_shape_per_video": [3, EXPECTED_FRAMES, EXPECTED_JOINTS],
        "feature_channels": [
            "neck_centered_shoulder_scaled_x",
            "neck_centered_shoulder_scaled_y",
            "raw_dwpose_simcc_score",
        ],
        "normalization": {
            "origin": "neck (joint index 1 in 50J)",
            "scale": "median shoulder width per video (joints 2 and 5)",
            "minimum_reference_frames": MIN_REFERENCE_FRAMES,
        },
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nProcessed: {len(results)} | Failed: {len(errors)}")
    print(f"Manifest : {manifest_path}")
    if errors or len(results) != EXPECTED_VIDEOS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
