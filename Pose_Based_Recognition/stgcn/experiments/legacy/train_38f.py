import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_38f import KathakaliDataset, METADATA
from model import STGCN

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/three_partition_xy_conf_38f"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, predictions, targets = 0.0, [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            predictions.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(targets, predictions),
        np.array(targets),
        np.array(predictions)
    )

def main():
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = KathakaliDataset("train")
    val_ds = KathakaliDataset("val")
    test_ds = KathakaliDataset("test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    metadata = pd.read_csv(METADATA)
    class_names = (
        metadata[["label_index", "class_name"]]
        .drop_duplicates()
        .sort_values("label_index")["class_name"].tolist()
    )
    num_classes = len(class_names)

    model = STGCN(num_classes=num_classes, in_channels=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Classes: {num_classes}")
    print("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr_now
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {lr_now:.6g}",
            flush=True
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "class_names": class_names
            }, OUT / "best_model.pt")
        else:
            patience_counter += 1

        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch} | Best validation accuracy: {best_val_acc:.4f}")

    checkpoint = torch.load(
        OUT / "best_model.pt", map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )
    test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\nTEST RESULTS")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=class_names, zero_division=0
    ))

    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(OUT / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUT / "confusion_matrix.csv"
    )

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ST-GCN 38 Frames — Test Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
    plt.yticks(range(num_classes), class_names, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    print("\nSaved outputs to:", OUT)

if __name__ == "__main__":
    main()
