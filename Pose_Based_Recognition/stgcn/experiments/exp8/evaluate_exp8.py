import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


from dataset_exp7 import KathakaliDataset
from model_exp8 import STGCN

BASE = Path(__file__).resolve().parents[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
print("Evaluation device:", device, f"({gpu_name})" if gpu_name else "")

data = np.load(r"/home/niranjanr/CLC/22-11-26_S5_break/Video_classification_Kathakali/DGNN_Audit/Canonical_Preprocessing/v1/graph_sequences_xy.npz")
class_names = []
for class_id in range(41):
    names = data["class_name"][data["class_id"] == class_id]
    if len(names) == 0 or len(set(names.tolist())) != 1:
        raise ValueError(f"Invalid class mapping for class {class_id}")
    class_names.append(str(names[0]))

dataset = KathakaliDataset("test")
loader = DataLoader(dataset, batch_size=16, shuffle=False)

A = np.load(BASE / "shared/graph/adjacency_single_exp8.npy")
model = STGCN(A, num_classes=41, in_channels=2).to(device)

checkpoint = torch.load(
    BASE / "runs/exp8/exp8_xy_single_adjacency/best_model.pt",
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, labels, mask in loader:
        x = x.to(device)
        mask = mask.to(device)

        outputs = model(x, mask)
        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n========== TEST RESULTS ==========")
print("Test samples:", len(y_true))
print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(41),
    target_names=class_names,
    zero_division=0
)
print(report)

(BASE / "logs/exp8/exp8_xy_single_adjacency").mkdir(parents=True, exist_ok=True)
(BASE / "logs/exp8/exp8_xy_single_adjacency/test_classification_report.txt").write_text(report)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(41))
np.savetxt(BASE / "logs/exp8/exp8_xy_single_adjacency/test_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

with open(BASE / "logs/exp8/exp8_xy_single_adjacency/test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true_id", "true_class", "predicted_id", "predicted_class"])
    for true_id, pred_id in zip(y_true, y_pred):
        writer.writerow([
            true_id, class_names[true_id],
            pred_id, class_names[pred_id]
        ])

metrics = {
    "test_samples": len(y_true),
    "test_accuracy": float(accuracy),
    "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted_f1),
    "device": str(device),
    "gpu_name": gpu_name,
    "checkpoint": str(BASE / "runs/exp8/exp8_xy_single_adjacency/best_model.pt"),
}
(BASE / "logs/exp8/exp8_xy_single_adjacency/test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

print("\nTest reports saved in logs/exp8/exp8_xy_single_adjacency/")
