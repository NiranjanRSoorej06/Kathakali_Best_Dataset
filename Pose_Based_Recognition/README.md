# Kathakali Pose-Based Recognition

A pose-based Kathakali gesture-recognition pipeline using **DWPose** for 2D skeleton extraction and **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** for classification.

The project is organized into two main stages:

1. **Pose extraction:** extract 2D body-joint coordinates and confidence scores from Kathakali videos using DWPose.
2. **Pose-based recognition:** prepare skeleton features, construct the joint graph, train ST-GCN models, and evaluate gesture classification.

---

## 1. Dataset

The current standard dataset contains:

| Property | Value |
|---|---:|
| Gesture classes | 41 |
| Videos | 430 |
| Sampled frames per video | 100 |
| Selected joints per frame | 50 |
| Coordinate channels | X, Y |
| Additional channel | DWPose confidence |

The standard processed representation is **100 frames × 50 joints × 3 features** (X, Y, confidence).

The selected DWPose joint indices are:

```text
[0–7] + [92–133]
```

The extracted coordinates and confidence scores are retained as separate fields in the raw pose files.

## 2. Overall Pipeline

```text
Kathakali source videos
        |
        v
      DWPose
        |
        v
2D keypoints + confidence scores
        |
        v
Pose dataset preparation
        |
        v
ST-GCN graph-based recognition
        |
        v
Evaluation
(accuracy, macro-F1, loss, confusion matrix)
```

## 3. Repository Organization

The project is located at:

```text
Kathakali_Best_Dataset/
├── GitHub_Fill_In/
├── Videos_Cropped/
└── Pose_Based_Recognition/
```

The source videos remain in the parent directory. They are not duplicated inside the recognition project.

Main project structure:

```text
Pose_Based_Recognition/
├── README.md
├── dwpose/
│   ├── implementation/
│   ├── old_archive/
│   └── extracted_430/
│       └── DWPose_50J_100F_41Classes/
├── scripts/
├── logs/
├── metadata/
└── stgcn/
    ├── README.md
    ├── experiments/
    │   ├── common/
    │   ├── legacy/
    │   ├── exp1/
    │   ├── exp2/
    │   ├── exp3/
    │   ├── exp4/
    │   ├── exp5/
    │   ├── exp6/
    │   ├── exp7/
    │   ├── exp8/
    │   └── exp9/
    ├── preprocessing/
    ├── shared/
    │   ├── graph/
    │   └── inference/
    ├── datasets/
    │   ├── processed/
    │   ├── processed_38f/
    │   └── processed_exp9/
    ├── runs/
    └── logs/
```

### Main directories

| Directory | Purpose |
|---|---|
| `dwpose/implementation/` | DWPose implementation and model weights |
| `dwpose/old_archive/` | Earlier pose-extraction data retained for reference |
| `dwpose/extracted_430/` | Pose extraction output for the 430-video dataset |
| `scripts/` | Extraction and supporting scripts |
| `metadata/` | Extraction manifests and metadata |
| `stgcn/experiments/` | Experiment-specific model, data, training and evaluation code |
| `stgcn/experiments/common/` | Shared dataset and training/evaluation utilities |
| `stgcn/experiments/legacy/` | Earlier implementations retained for reference |
| `stgcn/preprocessing/` | Processed-dataset preparation scripts |
| `stgcn/shared/graph/` | Graph construction code, adjacency matrices and graph metadata |
| `stgcn/shared/inference/` | Video prediction/inference code |
| `stgcn/datasets/` | Processed skeleton datasets |
| `stgcn/runs/` | Model checkpoints |
| `stgcn/logs/` | Training and evaluation outputs |

## 4. DWPose Extraction

The latest extraction output is stored in:

```text
dwpose/extracted_430/DWPose_50J_100F_41Classes/
```

The extracted pose representation for each video contains:

| Field | Shape | Description |
|---|---|---|
| `keypoints` | `(100, 50, 2)` | X and Y coordinates |
| `scores` | `(100, 50)` | Joint confidence scores |
| `frame_indices` | `(100,)` | Sampled source-frame indices |
| `frame_valid` | `(100,)` | Frame-validity indicators |

The raw coordinates and confidence scores are stored separately. The normalized ST-GCN dataset is prepared from these extracted pose files.

## 5. ST-GCN Dataset Preparation

Processed datasets are stored under:

```text
stgcn/datasets/
```

### 5.1 Standard 100-frame dataset

Location:

```text
stgcn/datasets/processed/
```

The normalized feature files use the following fields:

| Key | Shape / type | Description |
|---|---|---|
| `features` | `(100, 50, 3)` | Normalized X, normalized Y and confidence |
| `keypoints_normalized` | `(100, 50, 2)` | Normalized coordinates |
| `scores` | `(100, 50)` | Confidence scores |
| `frame_valid` | `(100,)` | Frame-validity mask |
| `normalization_valid` | `(100,)` | Normalization-validity mask |
| `frame_indices` | `(100,)` | Sampled source-frame indices |

The dataset metadata is stored in:

```text
stgcn/datasets/processed/metadata.csv
stgcn/datasets/processed/metadata_split.csv
```

The recorded split for the 430-video standard dataset is:

| Split | Videos |
|---|---:|
| Train | 300 |
| Validation | 65 |
| Test | 65 |
| **Total** | **430** |

The metadata contains 41 gesture classes. The processed dataset was checked for consistent feature shapes, finite values, matching labels/class indices, and valid frame and normalization masks.

### 5.2 38-frame dataset

Experiment 9 uses a 38-frame representation. Its combined dataset is stored at:

```text
stgcn/datasets/processed_exp9/exp9_xyc_38f.npz
```

The combined file contains:

| Key | Shape |
|---|---|
| `X` | `(430, 38, 50, 3)` |
| `frame_mask` | `(430, 38)` |
| `labels` | `(430,)` |
| `class_names` | `(41,)` |
| `video_names` | `(430,)` |
| `source_files` | `(430,)` |
| `video_indices` | `(430,)` |
| `split` | `(430,)` |
| `sampled_frame_indices` | `(38,)` |

## 6. ST-GCN Graph and Model Components

Graph-related files are stored in:

```text
stgcn/shared/graph/
```

The graph represents the skeletal relationships between the 50 selected joints.

The project includes two graph configurations:

- **Single adjacency:** one skeletal adjacency matrix.
- **Three-partition adjacency:** the skeletal graph divided into three spatial partitions.

Relevant shared files include:

```text
graph.py
model_single_adjacency.py
adjacency_partitions_exp7.npy
adjacency_single_exp8.npy
adjacency_single_exp9.npy
graph_metadata_exp7.json
stgcn_graph_exp7.py
```

The exact graph file used by an experiment should be taken from that experiment's implementation/configuration rather than inferred from the filename alone.

Shared video inference code is located at:

```text
stgcn/shared/inference/predict_video.py
```

## 7. Experiments

The experiment series investigates pose-feature representation, graph configuration, dataset version/size, and temporal sampling.

### 7.1 Experiment configurations

| Exp | Dataset | Input features | Graph configuration | Frames | Purpose |
|---:|---:|---|---|---:|---|
| 1 | Earlier 410-video dataset | XY | Original ST-GCN | Existing dataset | Earlier baseline configuration |
| 2 | Earlier 410-video dataset | Normalized XY | 3-partition | Existing dataset | Earlier partitioned-graph configuration |
| 3 | Current 430-video dataset | XY + confidence | Single adjacency | 100 | Current-repository baseline |
| 4 | Current 430-video dataset | XY | Single adjacency | 100 | Evaluate XY-only input with single adjacency |
| 5 | Current 430-video dataset | XY + confidence | 3-partition | 100 | Evaluate confidence features with 3-partition graph |
| 6 | Current 430-video dataset | XY | 3-partition | 100 | Evaluate XY-only input with 3-partition graph |
| 7 | Earlier 410-video dataset | XY | 3-partition | 100 | Evaluate the 410-video configuration with 3-partition graph |
| 8 | Earlier 410-video dataset | XY | Single adjacency | 100 | Evaluate the 410-video configuration with single adjacency |
| 9 | Current 430-video dataset | XY + confidence | Single adjacency | 38 | Evaluate reduced temporal sampling |

Experiments 1 and 2 refer to earlier 410-video work. Their previously quoted results came from a separate repository and are not included in the current-repository result comparison below. Experiment 3 is the current repository's 430-video baseline. Results from different repositories must not be combined as if they were produced by the same implementation and evaluation setup.

### 7.2 Recorded test results

The following table reports the results recorded for the current project experiments. Results from the separate repository for Experiments 1 and 2 are deliberately excluded.

| Exp | Test samples | Test accuracy | Test macro-F1 |
|---:|---:|---:|---:|
| 3 | 65 | 95.38%* | Not recorded in the consolidated result |
| 4 | 65 | 93.85% | 88.94% |
| 5 | 65 | 93.85% | 90.89% |
| 6 | 65 | 95.38% | 92.20% |
| 7 | 82 | 68.29% | 65.33% |
| 8 | 82 | 42.68% | 39.85% |
| 9 | 82 | 69.51% | 66.56% |

`*` Experiment 3's official recorded baseline is 95.38%. A later re-evaluation of its existing checkpoint produced 96.92% accuracy (63/65 correct), macro-F1 95.12%, and test loss 0.14050. The two values are retained as separate records because the discrepancy has not been reconciled. Do not silently replace the official baseline with the re-evaluation result.

### 7.3 Detailed recorded results

#### Experiment 3 — 430 videos, XY + confidence, single adjacency

- Official recorded baseline test accuracy: **95.38%**.
- A later evaluation of `runs/exp3/best_model.pt` recorded 65 test samples, 96.92% test accuracy (63/65 correct), 95.12% macro-F1 and 0.14050 test loss.
- The official baseline and later re-evaluation are both retained pending reconciliation.

#### Experiment 4 — 430 videos, XY, single adjacency

- Best checkpoint: epoch 82.
- Test samples: 65.
- Test loss: 0.10663.
- Test accuracy: 93.85%.
- Test macro-F1: 88.94%.

#### Experiment 5 — 430 videos, XY + confidence, 3-partition

- Best checkpoint: epoch 57.
- Test samples: 65.
- Test loss: 0.10880.
- Test accuracy: 93.85%.
- Test macro-F1: 90.89%.

#### Experiment 6 — 430 videos, XY, 3-partition

- Best checkpoint: epoch 56.
- Test samples: 65.
- Test loss: 0.11643.
- Test accuracy: 95.38%.
- Test macro-F1: 92.20%.

#### Experiment 7 — 410 videos, XY, 3-partition

- Test samples: 82.
- Test accuracy: 68.29%.
- Test macro-F1: 65.33%.

#### Experiment 8 — 410 videos, XY, single adjacency

- Test samples: 82.
- Test accuracy: 42.68%.
- Test macro-F1: 39.85%.

#### Experiment 9 — 430 videos, XY + confidence, single adjacency, 38 frames

- Temporal sampling: 38 frames.
- Best checkpoint: epoch 97.
- Test samples: 82.
- Test accuracy: 69.51%.
- Test macro-F1: 66.56%.

## 8. Interpretation of the Results

### 8.1 Best recorded result

Among the results listed above, Experiment 6 has the strongest recorded combination of test accuracy and macro-F1:

- Test accuracy: **95.38%**
- Test macro-F1: **92.20%**
- Test samples: 65
- Best checkpoint: epoch 56

Experiment 3 has an official baseline accuracy of 95.38%; its later checkpoint re-evaluation is recorded separately because it produced a different result.

### 8.2 Single adjacency versus three partitions

Experiments 4 and 6 use the same 430-video dataset, XY input and 100-frame representation, while changing the graph configuration.

| Experiment | Graph | Accuracy | Macro-F1 |
|---|---|---:|---:|
| 4 | Single adjacency | 93.85% | 88.94% |
| 6 | 3-partition | 95.38% | 92.20% |

In this comparison, the 3-partition configuration is higher by 1.53 percentage points in accuracy and 3.26 percentage points in macro-F1.

### 8.3 Confidence features

Experiments 5 and 6 both use the 430-video dataset, 100 frames and a 3-partition graph. They differ in input features:

| Experiment | Input | Accuracy | Macro-F1 |
|---|---|---:|---:|
| 5 | XY + confidence | 93.85% | 90.89% |
| 6 | XY | 95.38% | 92.20% |

For these recorded runs, XY-only input is higher on both metrics. This is an observed result for these runs, not proof that confidence features are generally harmful.

### 8.4 410-video experiments

Experiments 7 and 8 both use the 410-video dataset and XY features:

| Experiment | Graph | Accuracy | Macro-F1 |
|---|---|---:|---:|
| 7 | 3-partition | 68.29% | 65.33% |
| 8 | Single adjacency | 42.68% | 39.85% |

The 3-partition run is higher in this pair. The 410-video and 430-video experiment groups have different dataset versions and test-set sizes (82 versus 65). Their results should not be interpreted as a controlled measurement of the effect of adding 20 videos.

### 8.5 Temporal sampling

Experiment 9 uses 38 frames and records 69.51% test accuracy and 66.56% macro-F1. The standard 100-frame experiments have different configurations and, in some cases, different test-set sizes. Therefore, these results alone do not isolate the effect of reducing the frame count.

## 9. Checkpoints, Logs and Evaluation Outputs

Model checkpoints are stored under:

```text
stgcn/runs/
```

Training and evaluation outputs are stored under:

```text
stgcn/logs/
```

Experiment-specific directories preserve the corresponding checkpoints and reports. Depending on the experiment, available outputs include:

```text
best_model.pt
training_history.csv
training_log.csv
test_metrics.json
predictions.csv
classification_report.csv
confusion_matrix.csv
confusion_matrix.png
```

Not every experiment directory necessarily contains every listed file. Use the files present in the relevant experiment's output directory as the record for that run.

The best checkpoint is selected using validation loss in the shared training engine.

## 10. Accessing the Project

From the HPC account:

```bash
cd ~/Kathakali_Best_Dataset/Pose_Based_Recognition
```

Enter the ST-GCN project:

```bash
cd ~/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn
```

Open an experiment directory, for example:

```bash
cd experiments/exp6
```

View checkpoints:

```bash
cd ~/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/runs
```

View evaluation logs:

```bash
cd ~/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/logs
```

List directories if `tree` is not installed:

```bash
find . -maxdepth 3 -type d | sort
```

## 11. Reproducibility Notes

- Keep the original source videos in the parent dataset directory.
- Keep raw DWPose extraction outputs separate from processed ST-GCN datasets.
- Preserve each experiment's model, dataset wrapper, training script, evaluation script, checkpoint and logs.
- Do not overwrite historical metrics when re-evaluating a checkpoint; save the new evaluation separately and record the checkpoint and evaluation settings.
- Record the dataset version, split, input channels, frame count, graph configuration, model architecture, random seed and training settings for every new run.
- Compare experiments directly only when the relevant data split and other experimental conditions are held constant.
- Keep results from separate repositories clearly distinguished.

## 12. Current Project Summary

| Component | Recorded state |
|---|---|
| DWPose extraction for 430 videos | Completed |
| Standard 100-frame pose dataset | Prepared |
| 38-frame dataset for Experiment 9 | Prepared |
| ST-GCN experiment implementations | Organized |
| Experiments 3–9 | Training/evaluation results recorded |
| Experiments 1–2 | Historical configurations; prior quoted results belong to a separate repository |
| Checkpoints and logs | Retained in experiment-specific directories |
| Best recorded current-project result in this summary | Experiment 6: 95.38% accuracy, 92.20% macro-F1 |

**Project outcome:** a DWPose-to-ST-GCN pipeline for 41-class Kathakali gesture recognition, with a best recorded current-project test accuracy of **95.38%** and macro-F1 of **92.20%** (Experiment 6).
