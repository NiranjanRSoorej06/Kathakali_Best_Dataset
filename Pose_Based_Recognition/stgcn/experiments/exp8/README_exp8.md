# Experiment 8 — XY Single-Adjacency ST-GCN

## Dataset
- 410 videos
- 41 classes
- 50 joints
- 38 frames
- XY coordinates (2 channels)
- Split: 287 train / 41 validation / 82 test
- Canonical graph_sequences_xy.npz

## Model
- Single normalized adjacency matrix (50 x 50)
- Self-loops included
- Five ST-GCN blocks
- Channels: 2 → 32 → 32 → 64 → 64 → 128
- Temporal kernel: 9
- Mask-aware temporal processing and masked pooling
- Classifier: 128 → 128 → 41

## Training
- Seed: 42
- Maximum epochs: 100
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Early stopping patience: 20
- Best checkpoint selected by validation loss

## Results
- Best epoch: 48
- Test samples: 82
- Test accuracy: 42.68%
- Macro F1: 0.3985
- Weighted F1: 0.3985

## Artifacts
- Checkpoint: runs/exp8/exp8_xy_single_adjacency/best_model.pt
- Training log: logs/exp8/exp8_xy_single_adjacency/training_log.csv
- Console log: logs/exp8/exp8_xy_single_adjacency/training_console.log
- Validation metrics: logs/exp8/exp8_xy_single_adjacency/best_validation_metrics.json
- Test metrics: logs/exp8/exp8_xy_single_adjacency/test_metrics.json
- Predictions: logs/exp8/exp8_xy_single_adjacency/test_predictions.csv
- Confusion matrix: logs/exp8/exp8_xy_single_adjacency/test_confusion_matrix.csv
- Classification report: logs/exp8/exp8_xy_single_adjacency/test_classification_report.txt

## Comparison
Experiment 7 — 3-partition ST-GCN: 68.29%
Experiment 8 — Single-adjacency ST-GCN: 42.68%

The architectures differ in depth and channel widths; this is not an isolated graph-partition ablation.
