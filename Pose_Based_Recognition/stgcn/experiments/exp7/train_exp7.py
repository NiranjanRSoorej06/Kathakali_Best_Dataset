import sys
import csv
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))


from dataset_exp7 import KathakaliDataset
from model_exp7 import STGCN


# Configuration
SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 20

BASE = Path(__file__).resolve().parents[2]
GRAPH_PATH = BASE / "shared/graph/adjacency_partitions_exp7.npy"
DATA_PATH = Path(r"/home/niranjanr/CLC/22-11-26_S5_break/Video_classification_Kathakali/DGNN_Audit/Canonical_Preprocessing/v1/graph_sequences_xy.npz")
MODEL_PATH = BASE / "runs/exp7/exp7_xy_3partition/best_model.pt"
LOG_PATH = BASE / "logs/exp7/exp7_xy_3partition/training_log.csv"
METRICS_PATH = BASE / "logs/exp7/exp7_xy_3partition/best_validation_metrics.json"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
print("Device:", device, f"({gpu_name})" if gpu_name else "")

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)
if MODEL_PATH.exists() or LOG_PATH.exists() or METRICS_PATH.exists():
    raise FileExistsError("exp7_xy_3partition outputs already exist; refusing to overwrite this run")

# Datasets and loaders
train_dataset = KathakaliDataset("train")
val_dataset = KathakaliDataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Model
adjacency = np.load(GRAPH_PATH)
model = STGCN(adjacency, num_classes=41, in_channels=2).to(device)

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

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_val_f1 = 0.0
best_epoch = 0
epochs_without_improvement = 0

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_macro_f1", "learning_rate"
    ])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_losses = []
    train_true, train_pred = [], []

    for x, labels, mask in train_loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(x, mask)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_losses.append(loss.item())
        train_true.extend(labels.cpu().numpy())
        train_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    train_loss = float(np.mean(train_losses))
    train_acc = accuracy_score(train_true, train_pred)

    # Validation
    model.eval()
    val_losses = []
    val_true, val_pred = [], []

    with torch.no_grad():
        for x, labels, mask in val_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            outputs = model(x, mask)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    val_loss = float(np.mean(val_losses))
    val_acc = accuracy_score(val_true, val_pred)
    val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro-F1: {val_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, train_loss, train_acc,
            val_loss, val_acc, val_f1, current_lr
        ])

    # Save best checkpoint based on validation loss
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_val_accuracy = float(val_acc)
        best_val_f1 = float(val_f1)
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "num_classes": 41,
            "seed": SEED
        }, MODEL_PATH)

        print("  -> Best model checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

print("\nTraining finished.")
print("Best validation loss:", best_val_loss)
print("Best epoch:", best_epoch)
print("Checkpoint:", MODEL_PATH)
print("Training log:", LOG_PATH)
METRICS_PATH.write_text(json.dumps({
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "best_validation_accuracy": best_val_accuracy,
    "best_validation_macro_f1": best_val_f1,
    "seed": SEED,
    "epochs_configured": EPOCHS,
    "device": str(device),
    "gpu_name": gpu_name,
    "checkpoint": str(MODEL_PATH),
    "training_log": str(LOG_PATH),
}, indent=2) + "\n", encoding="utf-8")
print("Validation metrics:", METRICS_PATH)
