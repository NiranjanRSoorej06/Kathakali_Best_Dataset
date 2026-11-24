# Experiment 5

- Dataset: 430 videos, 41 classes
- Split: 300 train / 65 validation / 65 test
- Frames: 100
- Joints: 50
- Features: XY + Confidence
- Input channels: 3
- Graph: Three partitions
- Seed: 42
- Maximum epochs: 100
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 0.0001
- Early stopping patience: 20
- Training status: Not started

Run from the ST-GCN project root:

```bash
python -m experiments.exp5.train_exp5
python -m experiments.exp5.evaluate_exp5
```
