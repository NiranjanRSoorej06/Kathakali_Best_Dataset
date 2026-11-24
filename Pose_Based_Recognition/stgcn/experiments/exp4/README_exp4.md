# Experiment 4

- Dataset: 430 videos, 41 classes
- Split: 300 train / 65 validation / 65 test
- Frames: 100
- Joints: 50
- Features: XY
- Input channels: 2
- Graph: Single adjacency
- Seed: 42
- Maximum epochs: 100
- Batch size: 16
- Learning rate: 0.001
- Weight decay: 0.0001
- Early stopping patience: 20
- Training status: Not started

Run from the ST-GCN project root:

```bash
python -m experiments.exp4.train_exp4
python -m experiments.exp4.evaluate_exp4
```
