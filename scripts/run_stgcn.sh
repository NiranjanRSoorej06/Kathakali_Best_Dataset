#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "===================================="
echo "       KATHAKALI ST-GCN TRAINING"
echo "===================================="

# Check required files
for f in \
    data/dataset.py \
    models/st_gcn.py \
    graph/stgcn_adjacency.npy \
    dataset_splits/corrected_split_manifest.csv
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

mkdir -p training runs/stgcn

# Create training program
cat > training/train_stgcn.py <<'PY'
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import KathakaliDataset
from models import STGCN


# ---------------- CONFIGURATION ----------------

SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 15
NUM_WORKERS = 0

OUTPUT = ROOT / "runs" / "stgcn"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- DATA ----------------

train_set = KathakaliDataset(ROOT, "train")
val_set = KathakaliDataset(ROOT, "validation")
test_set = KathakaliDataset(ROOT, "test")

assert train_set.classes == val_set.classes == test_set.classes

classes = train_set.classes
num_classes = len(classes)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

print("\nDataset:")
print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Test:", len(test_set))
print("Classes:", classes)

# ---------------- MODEL ----------------

model = STGCN(
    num_classes=num_classes,
    in_channels=3,
    graph_path=str(ROOT / "graph" / "stgcn_adjacency.npy"),
    dropout=0.3
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

print("\nModel parameters:", sum(
    p.numel() for p in model.parameters()
))

# ---------------- EVALUATION FUNCTION ----------------

def evaluate(loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += batch_size

            all_targets.extend(y.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        all_targets,
        all_predictions
    )


# ---------------- TRAINING ----------------

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
history = []

checkpoint_path = OUTPUT / "best_model.pt"

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(1) == y).sum().item()
        total += batch_size

    train_loss = running_loss / total
    train_acc = running_correct / total

    val_loss, val_acc, _, _ = evaluate(val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    })

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f} "
        f"LR: {current_lr:.6f}",
        flush=True
    )

    # Save best checkpoint based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "classes": classes,
            "num_classes": num_classes,
            "input_shape": [3, 100, 50],
            "seed": SEED
        }, checkpoint_path)

        print("  Best checkpoint saved.", flush=True)

    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(
        OUTPUT / "training_history.csv",
        index=False
    )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ---------------- TEST BEST CHECKPOINT ----------------

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc, targets, predictions = evaluate(test_loader)

print("\n========== FINAL TEST RESULTS ==========")
print("Best epoch:", best_epoch)
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_acc, 4))
print("Test accuracy (%):", round(test_acc * 100, 2))

# ---------------- CONFUSION MATRIX ----------------

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

for actual, predicted in zip(targets, predictions):
    confusion[actual, predicted] += 1

pd.DataFrame(
    confusion,
    index=classes,
    columns=classes
).to_csv(OUTPUT / "confusion_matrix.csv")

# ---------------- PER-CLASS METRICS ----------------

class_metrics = []

for i, name in enumerate(classes):
    tp = confusion[i, i]
    fp = confusion[:, i].sum() - tp
    fn = confusion[i, :].sum() - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    class_metrics.append({
        "class": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correct": int(tp),
        "total": int(confusion[i, :].sum())
    })

pd.DataFrame(class_metrics).to_csv(
    OUTPUT / "per_class_metrics.csv",
    index=False
)

results = {
    "model": "ST-GCN",
    "best_epoch": best_epoch,
    "best_validation_loss": float(best_val_loss),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_accuracy_percent": float(test_acc * 100),
    "classes": classes,
    "train_samples": len(train_set),
    "validation_samples": len(val_set),
    "test_samples": len(test_set),
    "parameters": sum(p.numel() for p in model.parameters())
}

with open(OUTPUT / "test_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved outputs in:", OUTPUT)
print(" - best_model.pt")
print(" - training_history.csv")
print(" - confusion_matrix.csv")
print(" - per_class_metrics.csv")
print(" - test_results.json")
print("\nST-GCN TRAINING COMPLETED.")


PY

echo "Training program created."
echo "Starting ST-GCN training..."

python -u training/train_stgcn.py
