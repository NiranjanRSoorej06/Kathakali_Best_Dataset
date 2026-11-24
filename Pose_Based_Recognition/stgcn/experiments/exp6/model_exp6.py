import numpy as np
from pathlib import Path
from experiments.exp7.model_exp7 import STGCN

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "shared/graph/adjacency_partitions_exp7.npy"

def build_model():
    adjacency = np.load(GRAPH)
    return STGCN(adjacency, num_classes=41, in_channels=2)
