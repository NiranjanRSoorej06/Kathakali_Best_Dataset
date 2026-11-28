import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path.home() / "Kathakali_Best_Dataset"

VIDEO_DIR = PROJECT / "new_test_videos" / "New folder"

STGCN_DIR = (
    PROJECT / "Pose_Based_Recognition"
    / "stgcn"
)

TEST_DIR = STGCN_DIR / "testing" / "new40"
OUTPUT = TEST_DIR / "raw_pose"
LOG_DIR = TEST_DIR / "logs"

DWPOSE_DIR = (
    PROJECT / "Pose_Based_Recognition"
    / "dwpose" / "implementation"
    / "DWPose" / "ControlNet-v1-1-nightly"
)

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

JOINT_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7]
    + list(range(92, 113))
    + list(range(113, 134)),
    dtype=np.int64,
)

CLASSES = [
    "Aakasham", "Aana", "Aayudham", "Acchan", "Albutham",
    "Amma", "Anugraham", "Ariyikkuka", "Baalan", "Bhoomi",
    "Bhrahmaavu", "Chandran", "Chundu", "Gunam", "Irikkuka",
    "Kaadu", "Kallan", "Kannu", "Kashtam", "Kuyil",
    "Manassu", "Nadakkuka", "Nee_Roudram", "Nee_Shringaaram",
    "Paadam", "Pazham", "Pushpam", "Raajyam", "Samudram",
    "Satyam", "Shivan", "Simham", "Sooryan", "Svargam",
    "Thaamara", "Thalamudi", "Thaliru", "Urakkam", "Veedu",
    "Vishnu", "Vriksham"
]

def get_label(filename):
    matches = [
        cls for cls in CLASSES
        if filename.startswith(cls + "_")
    ]

    if not matches:
        raise ValueError(f"Cannot determine class from filename: {filename}")

    return max(matches, key=len)


def extract_video(video_path, pose):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

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

        person = int(np.argmax(raw_scores.mean(axis=1)))

        points = raw_points[person, JOINT_INDICES].copy()
        confidence = raw_scores[person, JOINT_INDICES].copy()

        frame_h, frame_w = frame.shape[:2]

        points[:, 0] /= frame_w
        points[:, 1] /= frame_h

        keypoints[i] = points.astype(np.float32)
        scores[i] = confidence.astype(np.float32)
        frame_valid[i] = bool(np.any(confidence > 0))

    cap.release()

    label = get_label(video_path.name)

    return {
        "keypoints": keypoints,
        "scores": scores,
        "frame_indices": frame_indices,
        "frame_valid": frame_valid,
        "label": label,
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
    videos = sorted(
        [
            p for p in VIDEO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
    )

    print("Videos discovered:", len(videos))

    if len(videos) != 40:
        raise RuntimeError(f"Expected 40 videos, found {len(videos)}")

    for video in videos:
        get_label(video.name)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    os.chdir(DWPOSE_DIR)
    sys.path.insert(0, str(DWPOSE_DIR))

    from annotator.dwpose.wholebody import Wholebody

    print("Loading DWPose Wholebody...")
    pose = Wholebody()
    print("DWPose initialized.")

    manifest_path = LOG_DIR / "dwpose_40_manifest.jsonl"
    errors = []
    completed = 0
    skipped = 0

    for i, video in enumerate(videos, 1):
        label = get_label(video.name)

        output_path = OUTPUT / label / f"{video.stem}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and valid_existing(output_path):
            skipped += 1
            print(f"[{i}/40] SKIP {video.name}")
            continue

        try:
            result = extract_video(video, pose)

            temp_path = output_path.with_suffix(".npz.tmp")

            with open(temp_path, "wb") as f:
                np.savez_compressed(f, **result)

            os.replace(temp_path, output_path)

            valid_frames = int(result["frame_valid"].sum())

            record = {
                "video": str(video),
                "label": label,
                "output": str(output_path),
                "status": "success",
                "valid_frames": valid_frames,
            }

            completed += 1

            print(
                f"[{i}/40] OK {video.name} "
                f"| {label} | valid: {valid_frames}/100"
            )

        except Exception as exc:
            record = {
                "video": str(video),
                "label": label,
                "status": "failed",
                "error": repr(exc),
            }

            errors.append(record)
            print(f"[{i}/40] FAILED {video.name}: {exc}")

        with open(manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    summary = {
        "total_videos": len(videos),
        "completed_this_run": completed,
        "skipped": skipped,
        "failed": len(errors),
        "errors": errors,
        "output": str(OUTPUT),
    }

    summary_path = LOG_DIR / "dwpose_40_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DWPose extraction finished")
    print("=" * 60)
    print("Videos:", len(videos))
    print("Completed:", completed)
    print("Skipped:", skipped)
    print("Failed:", len(errors))
    print("Output:", OUTPUT)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
