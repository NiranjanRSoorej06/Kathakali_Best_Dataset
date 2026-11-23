"""50-joint Kathakali graph with 3 ST-GCN spatial partitions."""

import numpy as np
from collections import deque

NUM_JOINTS = 50

JOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
]

FINGERS = ["thumb", "forefinger", "middle", "ring", "pinky"]

JOINT_NAMES += ["left_hand_root"]
for finger in FINGERS:
    JOINT_NAMES += [f"left_{finger}{i}" for i in range(1, 5)]

JOINT_NAMES += ["right_hand_root"]
for finger in FINGERS:
    JOINT_NAMES += [f"right_{finger}{i}" for i in range(1, 5)]

EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (7, 8),
    (4, 29),
]

for root, start in [(8, 9), (29, 30)]:
    for finger_idx in range(5):
        finger_start = start + finger_idx * 4
        EDGES.append((root, finger_start))
        for j in range(3):
            EDGES.append((finger_start + j, finger_start + j + 1))

assert len(JOINT_NAMES) == NUM_JOINTS
assert len(EDGES) == 49

CENTER = 1  # Neck


def build_adjacency():
    adj = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)

    for u, v in EDGES:
        adj[u, v] = 1
        adj[v, u] = 1

    distance = np.full(NUM_JOINTS, -1, dtype=np.int32)
    distance[CENTER] = 0
    queue = deque([CENTER])

    while queue:
        u = queue.popleft()
        for v in np.where(adj[u] > 0)[0]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                queue.append(v)

    assert np.all(distance >= 0), "Disconnected joints detected."

    A = np.zeros((3, NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    A[0] = np.eye(NUM_JOINTS, dtype=np.float32)

    for source in range(NUM_JOINTS):
        for target in range(NUM_JOINTS):
            if adj[source, target] == 0:
                continue

            if distance[target] < distance[source]:
                A[1, source, target] = 1
            elif distance[target] > distance[source]:
                A[2, source, target] = 1
            else:
                A[0, source, target] = 1

    for k in range(3):
        degree = A[k].sum(axis=0)
        degree[degree == 0] = 1
        A[k] /= degree[None, :]

    return A


if __name__ == "__main__":
    A = build_adjacency()
    print("Graph validation: PASSED")
    print("Partitions shape:", A.shape)
    print("Partition connections:", [int(np.count_nonzero(x)) for x in A])