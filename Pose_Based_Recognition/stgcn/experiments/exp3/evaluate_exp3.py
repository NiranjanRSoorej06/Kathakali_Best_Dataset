import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_exp3 import KathakaliDataset
from model_exp3 import STGCN

RUN_DIR = ROOT / "runs/exp3"
LOG_DIR = ROOT / "logs/exp3"
CHECKPOINT = RUN_DIR / "best_model.pt"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print("Device:", DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation accuracy:", checkpoint["val_accuracy"])
    print("Classes:", num_classes)

    test_ds = KathakaliDataset("test")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = STGCN(
        num_classes=num_classes,
        in_channels=3
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    predictions = []
    targets = []
    total_loss = 0.0

    print("\nEvaluating existing checkpoint...")

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            predictions.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    test_loss = total_loss / len(test_ds)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets, predictions,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        targets, predictions,
        average="weighted", zero_division=0
    )

    print("\n========== EXP3 TEST RESULTS ==========")
    print("Test samples:", len(test_ds))
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")

    report = classification_report(
        targets,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        LOG_DIR / "classification_report.csv"
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(range(num_classes))
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(LOG_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Exp3 - Test Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
    plt.yticks(range(num_classes), class_names, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(LOG_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    pd.DataFrame({
        "actual_index": targets,
        "predicted_index": predictions,
        "actual_class": [class_names[i] for i in targets],
        "predicted_class": [class_names[i] for i in predictions],
        "correct": targets == predictions
    }).to_csv(LOG_DIR / "predictions.csv", index=False)

    metrics = {
        "experiment": "Exp3",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_accuracy": float(checkpoint["val_accuracy"]),
        "test_samples": len(test_ds),
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1)
    }

    with open(LOG_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nResults saved to:", LOG_DIR)


if __name__ == "__main__":
    main()
