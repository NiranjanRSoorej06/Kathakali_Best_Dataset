# Experiment 9 — XY + Confidence, Single-Adjacency ST-GCN (38 Frames)

## 1. Objective

Evaluate a single-adjacency ST-GCN using 430 Kathakali videos, with XY coordinates and DWPose confidence scores as input features and 38-frame temporal sampling.

## 2. Dataset

- Total videos: 430
- Classes: 41
- Joints: 50
- Original sequence length: 100 frames
- Input features: X, Y, Confidence
- Input sequence length: 38 frames

### Dataset splits

| Split | Videos |
|---|---:|
| Training | 307 |
| Validation | 41 |
| Testing | 82 |
| **Total** | **430** |

Split assignment:
- Training: video indices 1–7 and 11
- Validation: video index 8
- Testing: video indices 9–10

All three splits contain all 41 classes.

## 3. Preprocessing

Source:
`dwpose/extracted_430/DWPose_50J_100F_41Classes`

Processing performed by `prepare_exp9.py`:

1. Load XY coordinates and confidence scores from the DWPose NPZ files.
2. Combine the three features into a single input representation.
3. Sample 38 evenly spaced frames from the original 100 frames.
4. Preserve raw confidence values without clipping or rescaling.
5. Apply frame masks to the sequences.
6. Map the 41 classes to sorted canonical class IDs.
7. Create training, validation, and test splits.

Prepared dataset:
`processed_exp9/exp9_xyc_38f.npz`

Input tensor shape:
`(N, 3, 38, 50)` — channels, frames, joints.

## 4. Model Architecture

- Architecture: ST-GCN
- Graph configuration: Single adjacency matrix
- Adjacency matrix: 50 × 50
- Input channels: 3 (XY + Confidence)
- Number of classes: 41
- ST-GCN blocks: 5
- Channel progression: 3 → 32 → 32 → 64 → 64 → 128
- Temporal kernel size: 9
- Temporal strides: 1, 1, 2, 1, 2
- Mask-aware temporal processing
- Masked global pooling
- Classifier: 128 → 128 → 41
- Trainable parameters: 289,353

The single-adjacency graph was copied from the Experiment 8 graph configuration.

## 5. Training Configuration

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Maximum epochs | 100 |
| Early stopping patience | 20 |
| Scheduler | ReduceLROnPlateau |
| Best checkpoint epoch | 97 |
| Device | cuda |
| GPU | NVIDIA H100 PCIe MIG 2g.20gb |

Best validation loss: 1.3465

Best validation accuracy: 0.6098 (60.98%)

Best validation macro-F1: 0.5589

Final epoch validation accuracy: 58.54%

The best checkpoint was selected using validation loss. The test split was not used during training.

## 6. Test Results

Evaluation was performed using the best checkpoint on the 82-video test split.

| Metric | Result |
|---|---:|
| Test samples | 82 |
| Test accuracy | 69.51% |
| Macro F1 | 0.6656 |
| Weighted F1 | 0.6656 |

## 7. Saved Artifacts

### Scripts

- `prepare_exp9.py`
- `dataset_exp9.py`
- `model_exp9.py`
- `train_exp9.py`
- `evaluate_exp9.py`

### Prepared data

- `processed_exp9/exp9_xyc_38f.npz`
- `processed_exp9/split_manifest.json`

### Model checkpoint

`runs/exp9/exp9_xy_confidence_single_adjacency_38f/best_model.pt`

### Training and evaluation outputs

`logs/exp9/exp9_xy_confidence_single_adjacency_38f/`

- `training_log.csv`
- `training_metrics.json`
- `test_metrics.json`
- `test_classification_report.txt`
- `test_confusion_matrix.csv`
- `test_predictions.csv`

## 8. Conclusion

Experiment 9 achieved **69.51% test accuracy** and a **0.6656 macro F1-score** using XY coordinates and confidence scores with a single-adjacency ST-GCN and 38-frame sampling.

This is the strongest test accuracy among Experiments 7–9:

- Experiment 7: 68.29%
- Experiment 8: 42.68%
- Experiment 9: 69.51%

These experiments differ in dataset size, input features, and temporal sampling. Their results should therefore not be interpreted as a controlled comparison of graph architectures alone.
