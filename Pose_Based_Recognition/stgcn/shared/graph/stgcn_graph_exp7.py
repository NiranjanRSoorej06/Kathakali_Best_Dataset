import json
import numpy as np
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "DGNN_Audit" / "Pre_Model_Artifacts"
OUTPUT = ROOT / "STGCN_Experiment" / "graph"

with open(SOURCE / "skeleton_graph_definition.json") as f:
    definition = json.load(f)

V = definition["node_count"]
CENTER = 1  # neck

# Build undirected graph from the existing anatomical edges.
adj = np.zeros((V, V), dtype=np.float32)

for edge in definition["anatomical_edges"]:
    u, v = edge["source"], edge["target"]
    adj[u, v] = 1
    adj[v, u] = 1

# Calculate shortest-path distance from the neck.
distance = np.full(V, -1, dtype=np.int32)
distance[CENTER] = 0
queue = deque([CENTER])

while queue:
    u = queue.popleft()
    for v in np.where(adj[u] > 0)[0]:
        if distance[v] == -1:
            distance[v] = distance[u] + 1
            queue.append(v)

assert np.all(distance >= 0), "Graph contains disconnected joints."

# ST-GCN spatial partitions.
A = np.zeros((3, V, V), dtype=np.float32)
A[0] = np.eye(V, dtype=np.float32)

for source in range(V):
    for target in range(V):
        if adj[source, target] == 0:
            continue

        if distance[target] < distance[source]:
            A[1, source, target] = 1  # Centripetal
        elif distance[target] > distance[source]:
            A[2, source, target] = 1  # Centrifugal
        else:
            A[0, source, target] = 1

# Normalize each partition by target-node degree.
for k in range(3):
    degree = A[k].sum(axis=0)
    degree[degree == 0] = 1
    A[k] /= degree[None, :]

OUTPUT.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT / "adjacency_partitions.npy", A)

metadata = {
    "nodes": V,
    "center_node": CENTER,
    "center_name": "neck",
    "partitions": ["self", "centripetal", "centrifugal"],
    "shape": list(A.shape),
    "normalization": "column degree normalization"
}

with open(OUTPUT / "graph_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Graph partitions created.")
print("Shape:", A.shape)
print("Partition connections:", [int(np.count_nonzero(x)) for x in A])
print("Joint distances from neck:", distance.tolist())
