# Experiment 7 — 410 Videos, XY, 3-Partition ST-GCN

## Objective
Evaluate 3-partition ST-GCN using XY coordinates from the canonical 410-video dataset.

## Dataset
- Videos: 410
- Classes: 41
- Train: 287
- Validation: 41
- Test: 82
- Input: XY
- Frames: 38
- Joints: 50
- Channels: 2

## Architecture
- Model: ST-GCN
- Graph: 3 partitions
- Partitions: Self, Centripetal, Centrifugal
- Center node: Neck
- Seed: 42

## Training Configuration
- Epochs: 100
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau

## Results

| Metric | Result |
|---|---:|
| Best epoch | 100 |
| Training accuracy | 100.00% |
| Validation accuracy | 68.29% |
| Validation Macro F1 | 0.6138 |
| Test accuracy | 68.29% |
| Test Macro F1 | 0.6533 |
| Test Weighted F1 | 0.6533 |

Correct predictions: 56 / 82
Incorrect predictions: 26 / 82

## Observations
- Significant training-validation accuracy gap.
- Evidence of overfitting.
- Seven classes recorded zero test F1.
- Two test samples per class; interpret per-class metrics cautiously.

## Artifacts
- runs/exp7/exp7_xy_3partition/best_model.pt
- logs/exp7/exp7_xy_3partition/training_log.csv
- logs/exp7/exp7_xy_3partition/best_validation_metrics.json
- logs/exp7/exp7_xy_3partition/test_metrics.json
- logs/exp7/exp7_xy_3partition/test_predictions.csv
- logs/exp7/exp7_xy_3partition/test_classification_report.txt
- logs/exp7/exp7_xy_3partition/test_confusion_matrix.csv

## Status
Completed.
