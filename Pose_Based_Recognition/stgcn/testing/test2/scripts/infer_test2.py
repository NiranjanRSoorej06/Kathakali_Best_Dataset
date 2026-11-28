from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "testing/test2"
PROCESSED = TEST_ROOT / "processed"
MANIFEST = PROCESSED / "manifest.csv"
CHECKPOINT = ROOT / "runs/additional/three_partition_xy_conf_38f/best_model.pt"
RESULTS = TEST_ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

# Import the exact experiment model
sys.path.insert(0, str(ROOT / "experiments/exp3"))
sys.path.insert(0, str(ROOT / "shared/graph"))
from model_exp3 import STGCN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]

# IMPORTANT: 3 input channels = normalized XY + confidence
model = STGCN(num_classes=len(class_names), in_channels=3)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device)
model.eval()

print("Checkpoint loaded successfully.")
print("Epoch:", checkpoint["epoch"])
print("Validation accuracy:", checkpoint["val_accuracy"])
print("Number of classes:", len(class_names))

df = pd.read_csv(MANIFEST)

predictions = []
y_true = []
y_pred = []
all_probs = []

with torch.inference_mode():
    for _, row in df.iterrows():
        path = Path(row["output"])

        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)

        if features.shape != (100, 50, 3):
            raise ValueError(f"Unexpected shape {features.shape}: {path}")

        # (T,V,C) -> (C,T,V,M)
        x = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(-1)
        x = x.unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

        true_label = str(row["label"])
        true_idx = int(row["label_index"])
        pred_label = class_names[pred_idx]

        confidence = float(probs[pred_idx].item())

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        all_probs.append(probs.cpu().numpy())

        predictions.append({
            "video_name": row["video_name"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": true_label == pred_label,
            "confidence": confidence,
            "true_index": true_idx,
            "predicted_index": pred_idx,
        })

        print(
            f"{row['video_name']} | "
            f"True: {true_label} | "
            f"Pred: {pred_label} | "
            f"Confidence: {confidence:.4f} | "
            f"{'CORRECT' if true_label == pred_label else 'WRONG'}"
        )

# Metrics
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

# Save predictions
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(RESULTS / "predictions.csv", index=False)

# Save confusion matrix
pd.DataFrame(
    cm, index=class_names, columns=class_names
).to_csv(RESULTS / "confusion_matrix.csv")

# Save per-class report
pd.DataFrame(report).transpose().to_csv(RESULTS / "classification_report.csv")

summary = {
    "checkpoint": str(CHECKPOINT),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "checkpoint_validation_accuracy": float(checkpoint["val_accuracy"]),
    "device": str(device),
    "total_videos": len(df),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "correct": int((y_true == y_pred).sum()),
    "incorrect": int((y_true != y_pred).sum()),
    "class_names": class_names,
}

with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 60)
print("NEW 40 VIDEO EVALUATION")
print("=" * 60)
print(f"Total videos    : {len(df)}")
print(f"Correct         : {summary['correct']}")
print(f"Incorrect       : {summary['incorrect']}")
print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")
print("=" * 60)

print("\nResults saved to:", RESULTS)
