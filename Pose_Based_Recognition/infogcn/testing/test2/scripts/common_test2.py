"""Shared constants and read-only canonical helpers for testing/test2 (EXP10RERUN Seed 21)."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

TEST2_ROOT = Path(__file__).resolve().parents[1]
INFOGCN_ROOT = TEST2_ROOT.parents[1]
POSE_ROOT = INFOGCN_ROOT.parent
STGCN_ROOT = POSE_ROOT / "stgcn"

CANONICAL_METADATA_PATH = STGCN_ROOT / "datasets" / "processed" / "metadata_split.csv"
NEW40_PROCESSED_DIR = STGCN_ROOT / "testing" / "new40" / "processed"
NEW40_MANIFEST_PATH = NEW40_PROCESSED_DIR / "manifest.csv"

EXP10RERUN_RUN_NAME = "exp10rerun_seed21"
EXP10RERUN_CHECKPOINT_PATH = INFOGCN_ROOT / "runs" / "exp10rerun" / EXP10RERUN_RUN_NAME / "best_model.pt"
EXP10RERUN_LOG_PATH = INFOGCN_ROOT / "logs" / "exp10rerun" / EXP10RERUN_RUN_NAME / "training_log.csv"

RESULTS_DIR = TEST2_ROOT / "results"

EXPECTED_VIDEOS = 40
EXPECTED_FRAMES = 100
EXPECTED_JOINTS = 50
EXPECTED_CHANNELS = 3

# Canonical 50-joint subset: 8 upper-body joints [0..7] + 21 left hand [92..112] + 21 right hand [113..133]
JOINT_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7] + list(range(92, 113)) + list(range(113, 134)),
    dtype=np.int64,
)


def load_canonical_class_names() -> list[str]:
    """Load the ordered 41 canonical class names from read-only metadata_split.csv."""
    if not CANONICAL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Canonical metadata not found: {CANONICAL_METADATA_PATH}")
    metadata = pd.read_csv(CANONICAL_METADATA_PATH)
    classes = (
        metadata[["label_index", "class_name"]]
        .drop_duplicates()
        .sort_values("label_index")["class_name"]
        .tolist()
    )
    if len(classes) != 41:
        raise ValueError(f"Expected 41 canonical classes, got {len(classes)}")
    return classes


def require_cuda() -> tuple[torch.device, str]:
    """Verify CUDA GPU availability and forbid silent CPU fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is NOT available (torch.cuda.is_available() == False). "
            "CPU fallback is disabled by strict requirement."
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    return device, gpu_name
