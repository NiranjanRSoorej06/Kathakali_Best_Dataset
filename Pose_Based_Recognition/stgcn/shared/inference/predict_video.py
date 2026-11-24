import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STGCN_DIR = Path(__file__).resolve().parent

DWPOSE_DIR = (
    ROOT / "dwpose/implementation/DWPose/"
    "ControlNet-v1-1-nightly"
)

CHECKPOINT = (
    STGCN_DIR / "runs/three_partition_xy_conf_seed21/best_model.pt"
)

sys.path.insert(0, str(STGCN_DIR))

from model import STGCN

JOINT_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7]
    + list(range(92, 113))
    + list(range(113, 134)),
    dtype=np.int64
)

NECK = 1
RIGHT_SHOULDER = 2
LEFT_SHOULDER = 5
MIN_REFERENCE_FRAMES = 50


def extract_pose(video_path, pose):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    return keypoints, scores, frame_valid, width, height


def normalize_features(keypoints, scores, frame_valid, width, height):
    if keypoints.shape != (100, 50, 2):
        raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")

    pixel_points = keypoints.copy()
    pixel_points[:, :, 0] *= width
    pixel_points[:, :, 1] *= height

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
            f"Only {len(usable_widths)} reliable shoulder frames found. "
            f"At least {MIN_REFERENCE_FRAMES} are required."
        )

    reference_shoulder_width = float(np.median(usable_widths))

    normalization_valid = (
        frame_valid
        & (scores[:, NECK] > 0)
        & np.all(np.isfinite(pixel_points[:, NECK, :]), axis=1)
    )

    normalized = np.zeros_like(pixel_points, dtype=np.float32)

    for t in np.where(normalization_valid)[0]:
        normalized[t] = (
            pixel_points[t] - pixel_points[t, NECK]
        ) / reference_shoulder_width

    features = np.concatenate(
        [normalized, scores[:, :, None]],
        axis=2
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values detected")

    print(f"Valid pose frames: {int(frame_valid.sum())}/100")
    print(f"Normalization-valid frames: {int(normalization_valid.sum())}/100")
    print(f"Reference shoulder frames: {len(usable_widths)}/100")

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Predict Kathakali mudra from a video using ST-GCN"
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Video:", video_path)

    # Initialize DWPose exactly as in the extraction pipeline.
    import os
    os.chdir(DWPOSE_DIR)
    sys.path.insert(0, str(DWPOSE_DIR))

    from annotator.dwpose.wholebody import Wholebody

    print("\nLoading DWPose...")
    pose = Wholebody()
    print("DWPose loaded.")

    print("\nExtracting pose features...")
    keypoints, scores, frame_valid, width, height = extract_pose(
        video_path, pose
    )

    features = normalize_features(
        keypoints, scores, frame_valid, width, height
    )

    # (T,V,C) -> (N,C,T,V,M)
    x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
    x = x.to(device)

    print("ST-GCN input shape:", tuple(x.shape))

    # Load checkpoint and saved class names.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    if isinstance(class_names, dict):
        class_names = [
            class_names[i] if i in class_names else class_names[str(i)]
            for i in range(len(class_names))
        ]

    if len(class_names) != 41:
        raise ValueError(
            f"Expected 41 class names, found {len(class_names)}"
        )

    model = STGCN(num_classes=41, in_channels=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nRunning prediction...")

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(args.top_k, len(class_names))
    values, indices = torch.topk(probabilities, top_k)

    predicted_index = int(indices[0].item())
    predicted_class = class_names[predicted_index]

    print("\n" + "=" * 45)
    print("       KATHAKALI MUDRA PREDICTION")
    print("=" * 45)
    print("Video:", video_path.name)
    print("Predicted mudra:", predicted_class)
    print(f"Model score: {values[0].item() * 100:.2f}%")

    print(f"\nTop {top_k} predictions:")
    for rank, (value, index) in enumerate(zip(values, indices), 1):
        name = class_names[int(index.item())]
        print(f"{rank}. {name:<22} {value.item() * 100:.2f}%")

    print("=" * 45)
    print("Note: Model scores are softmax probabilities,")
    print("not necessarily calibrated confidence estimates.")


if __name__ == "__main__":
    main()
