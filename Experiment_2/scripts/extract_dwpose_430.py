import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[2]
EXP_DIR = BASE / "Experiment_2"

DWPOSE_DIR = (
    EXP_DIR / "dwpose" / "implementation"
    / "DWPose" / "ControlNet-v1-1-nightly"
)

OUTPUT = (
    EXP_DIR / "dwpose" / "extracted_430"
    / "DWPose_50J_100F_41Classes"
)

VIDEO_ROOTS = [
    BASE / "GitHub_Fill_In",
    BASE / "Videos_Cropped",
]

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
JOINT_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7]
    + list(range(92, 113))
    + list(range(113, 134)),
    dtype=np.int64,
)

def discover_videos():
    videos = []
    seen = set()

    for root in VIDEO_ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    videos.append((path, resolved))

    return sorted(videos, key=lambda item: str(item[0]))

def extract_video(video_path, pose):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    frame_indices = np.linspace(
        0, total_frames - 1, 100, dtype=np.int64
    )

    keypoints = np.zeros((100, 50, 2), dtype=np.float32)
    scores = np.zeros((100, 50), dtype=np.float32)
    frame_valid = np.zeros(100, dtype=bool)

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok:
            continue

        raw_points, raw_scores = pose(frame)

        raw_points = np.asarray(raw_points).reshape(-1, 134, 2)
        raw_scores = np.asarray(raw_scores).reshape(-1, 134)

        if len(raw_points) == 0 or len(raw_scores) == 0:
            continue

        # Select the detected person with the highest mean confidence.
        person = int(np.argmax(raw_scores.mean(axis=1)))

        points = raw_points[person, JOINT_INDICES].copy()
        confidence = raw_scores[person, JOINT_INDICES].copy()

        frame_h, frame_w = frame.shape[:2]

        # Normalize pixel coordinates to image coordinates.
        points[:, 0] /= frame_w
        points[:, 1] /= frame_h

        keypoints[i] = points.astype(np.float32)
        scores[i] = confidence.astype(np.float32)
        frame_valid[i] = bool(np.any(confidence > 0))

    cap.release()

    return {
        "keypoints": keypoints,
        "scores": scores,
        "frame_indices": frame_indices,
        "frame_valid": frame_valid,
        "label": video_path.parent.name,
        "video_name": video_path.name,
        "original_width": np.int32(width),
        "original_height": np.int32(height),
        "fps": np.float32(fps),
        "total_frames": np.int64(total_frames),
        "joint_indices": JOINT_INDICES,
    }

def valid_existing(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["keypoints"].shape == (100, 50, 2)
                and data["scores"].shape == (100, 50)
                and data["frame_indices"].shape == (100,)
            )
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N videos (0 = all)"
    )
    args = parser.parse_args()

    videos = discover_videos()
    print(f"Discovered videos: {len(videos)}")

    classes = {path.parent.name for path, _ in videos}
    print(f"Classes: {len(classes)}")

    if len(videos) != 430 or len(classes) != 41:
        raise RuntimeError(
            f"Expected 430 videos and 41 classes; "
            f"found {len(videos)} videos and {len(classes)} classes"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # DWPose loads its ONNX weights using paths relative to this directory.
    os.chdir(DWPOSE_DIR)
    sys.path.insert(0, str(DWPOSE_DIR))

    from annotator.dwpose.wholebody import Wholebody

    print("Loading DWPose...")
    pose = Wholebody()
    print("DWPose loaded.")

    if args.limit:
        videos = videos[:args.limit]

    manifest = EXP_DIR / "metadata" / "dwpose_extraction_manifest.jsonl"
    completed = 0
    skipped = 0
    failed = 0

    for number, (original_path, video_path) in enumerate(videos, 1):
        label = original_path.parent.name
        output_path = OUTPUT / label / f"{video_path.stem}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and valid_existing(output_path):
            skipped += 1
            print(f"[{number}/{len(videos)}] SKIP {label}/{video_path.name}")
            continue

        try:
            result = extract_video(video_path, pose)

            # Atomic write: incomplete files are never mistaken for finished ones.
            temp_path = output_path.with_name(output_path.name + ".tmp")
            with open(temp_path, "wb") as file:
                np.savez_compressed(file, **result)
            os.replace(temp_path, output_path)

            valid_count = int(result["frame_valid"].sum())
            completed += 1

            record = {
                "video": str(original_path),
                "output": str(output_path.relative_to(BASE)),
                "status": "success",
                "valid_frames": valid_count,
            }

            with open(manifest, "a") as file:
                file.write(json.dumps(record) + "\n")

            print(
                f"[{number}/{len(videos)}] OK "
                f"{label}/{video_path.name} "
                f"| valid frames: {valid_count}/100"
            )

        except Exception as error:
            failed += 1
            record = {
                "video": str(original_path),
                "status": "failed",
                "error": repr(error),
            }

            with open(manifest, "a") as file:
                file.write(json.dumps(record) + "\n")

            print(
                f"[{number}/{len(videos)}] FAILED "
                f"{label}/{video_path.name}: {error}"
            )

    print("\nExtraction run finished.")
    print("Newly completed:", completed)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output:", OUTPUT)

if __name__ == "__main__":
    main()
