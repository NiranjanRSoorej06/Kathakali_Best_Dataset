from pathlib import Path
from collections import deque
import numpy as np

V = 50
CENTER = 1

# Anatomical connections
edges = [
    (0,1), (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (7,8), (4,29)
]

# Five fingers, four joints per finger, for each hand
for start in [8, 29]:
    for finger in range(5):
        first = start + 1 + finger * 4
        edges.append((start, first))
        for j in range(3):
            edges.append((first+j, first+j+1))

# Undirected graph for shortest-path distances
neighbors = [[] for _ in range(V)]
for i, j in edges:
    neighbors[i].append(j)
    neighbors[j].append(i)

# Calculate distance from neck
distance = [-1] * V
distance[CENTER] = 0
queue = deque([CENTER])

while queue:
    node = queue.popleft()
    for nxt in neighbors[node]:
        if distance[nxt] == -1:
            distance[nxt] = distance[node] + 1
            queue.append(nxt)

assert all(d >= 0 for d in distance), "Graph contains disconnected joints"

# Three spatial partitions
A = np.zeros((3, V, V), dtype=np.float32)

for i in range(V):
    A[0, i, i] = 1.0  # Self connections

for i, j in edges:
    if distance[i] == distance[j]:
        A[0, i, j] = 1.0
        A[0, j, i] = 1.0
    elif distance[i] < distance[j]:
        A[1, i, j] = 1.0  # Centripetal
        A[2, j, i] = 1.0  # Centrifugal
    else:
        A[1, j, i] = 1.0
        A[2, i, j] = 1.0

# Normalize each partition by column degree
for k in range(3):
    degree = A[k].sum(axis=0)
    inv_degree = np.zeros_like(degree)
    inv_degree[degree > 0] = 1.0 / degree[degree > 0]
    A[k] = A[k] @ np.diag(inv_degree)

Path("graph").mkdir(exist_ok=True)
np.save("graph/stgcn_adjacency.npy", A)

print("ST-GCN graph created")
print("Adjacency shape:", A.shape)
print("Partitions: root, centripetal, centrifugal")
print("Joint distances:", distance)
print("Saved: graph/stgcn_adjacency.npy")
