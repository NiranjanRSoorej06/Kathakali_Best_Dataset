import csv
import json
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset_exp9 import KathakaliDataset
from model_exp9 import STGCN

BASE = Path(__file__).resolve().parents[2]
RUN = "exp9_xy_confidence_single_adjacency_38f"
CHECKPOINT = BASE / "runs" / "exp9" / RUN / "best_model.pt"
OUT = BASE / "logs" / "exp9" / RUN

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    outputs = [
        OUT / "test_classification_report.txt",
        OUT / "test_confusion_matrix.csv",
        OUT / "test_predictions.csv",
        OUT / "test_metrics.json",
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite:\n" + "\n".join(existing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = KathakaliDataset("test")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    class_names = [str(n) for n in dataset.class_names]

    adjacency = np.load(BASE / "shared" / "graph" / "adjacency_single_exp9.npy")
    model = STGCN(adjacency, num_classes=41, in_channels=3).to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels, mask in loader:
            outputs_batch = model(x.to(device), mask.to(device))
            y_true.extend(labels.numpy())
            y_pred.extend(outputs_batch.argmax(dim=1).cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n========== TEST RESULTS ==========")
    print("Test samples:", len(y_true))
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report = classification_report(
        y_true, y_pred, labels=np.arange(41),
        target_names=class_names, zero_division=0
    )
    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    OUT.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(41))
    np.savetxt(outputs[1], cm, delimiter=",", fmt="%d")

    with open(outputs[2], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_id", "true_class", "predicted_id", "predicted_class"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([t, class_names[t], p, class_names[p]])

    metrics = {
        "experiment": RUN,
        "test_samples": len(y_true),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "best_training_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "checkpoint": str(CHECKPOINT),
    }
    outputs[3].write_text(json.dumps(metrics, indent=2) + "\n")
    print("Reports saved in:", OUT)

if __name__ == "__main__":
    main()
