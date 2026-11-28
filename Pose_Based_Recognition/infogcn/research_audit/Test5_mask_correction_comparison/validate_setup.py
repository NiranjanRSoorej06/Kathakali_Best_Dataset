"""Preflight configuration, dataset availability, baseline runs and initialization parity."""
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STGCN = ROOT.parent / "stgcn"
sys.path.insert(0, str(ROOT))
from experiments.exp10rerun.model_exp10rerun import InfoGCN
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset_class():
    p = STGCN / "experiments/common/dataset.py"
    spec = importlib.util.spec_from_file_location("test5_dataset_readonly", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KathakaliDataset


def aliases(key):
    pairs = [
        (r"^stem\.0\.", "stem_conv."), (r"^stem\.1\.", "stem_bn."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.0\.", r"blocks.\1.temporal.pre_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.1\.", r"blocks.\1.temporal.bn_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.3\.", r"blocks.\1.temporal.conv_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.0\.", r"blocks.\1.temporal.pre_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.1\.", r"blocks.\1.temporal.bn_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.3\.", r"blocks.\1.temporal.conv_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_pool\.1\.", r"blocks.\1.temporal.pool_conv."),
        (r"^blocks\.(\d+)\.temporal\.branch_1x1\.", r"blocks.\1.temporal.point."),
        (r"^blocks\.(\d+)\.temporal\.bn\.", r"blocks.\1.temporal.out_bn."),
        (r"^blocks\.(\d+)\.residual\.0\.", r"blocks.\1.residual_conv."),
        (r"^blocks\.(\d+)\.residual\.1\.", r"blocks.\1.residual_bn."),
    ]
    for pattern, repl in pairs:
        mapped, n = re.subn(pattern, repl, key)
        if n:
            return mapped
    return key


def main():
    config = json.loads((OUT / "comparison_config.json").read_text())
    required_training = {
        "epochs": 100, "batch_size": 32, "lr": 0.001, "weight_decay": 0.0001,
        "patience": 20, "num_workers": 0, "base_channels": 48, "dropout": 0.15,
        "head_dropout": 0.3, "label_smoothing": 0.0, "selection_metric": "val_loss",
        "scheduler": "plateau", "scheduler_factor": 0.5, "scheduler_patience": 6,
        "gradient_clip_norm": 5.0, "optimizer": "AdamW", "use_ib": False,
    }
    assert config["training"] | {} == required_training | {"augmentation": "none"}
    assert not (OUT / "baseline").exists() and not (OUT / "mask_correct").exists(), "Refusing to overwrite existing run outputs"

    metadata_path = STGCN / "datasets/processed/metadata_split.csv"
    df = pd.read_csv(metadata_path)
    counts = df.split.value_counts().to_dict()
    assert counts == config["split_counts"], f"Split mismatch: {counts}"
    classes = df[["label_index", "class_name"]].drop_duplicates().sort_values("label_index")
    assert len(classes) == 41 and classes.label_index.tolist() == list(range(41))
    feature_count = 0
    missing = []
    for rel in df.feature_path.astype(str):
        p = STGCN / rel
        if not p.exists():
            p = STGCN / rel.replace("stgcn/processed/", "datasets/processed/", 1)
        if not p.exists():
            missing.append(str(rel))
        else:
            feature_count += 1
    assert not missing, f"Missing feature files: {missing[:5]}"

    # Inspect only training and validation examples before the planned final evaluation.
    Dataset = load_dataset_class()
    observed = {}
    for split in ("train", "val"):
        ds = Dataset(split, in_channels=3)
        valid_counts = []
        for i in range(len(ds)):
            x, _y, mask = ds[i]
            assert tuple(x.shape) == (3, 100, 50) and torch.isfinite(x).all()
            valid_counts.append(int(mask.sum()))
        observed[split] = {"samples": len(ds), "min_valid_frames": min(valid_counts),
                           "max_valid_frames": max(valid_counts)}

    base_cfg = dict(num_classes=41, in_channels=3, num_joints=50, base_channels=48,
                    dropout=0.15, head_dropout=0.3, use_ib=False, latent_dim=128)
    torch.manual_seed(21)
    baseline = InfoGCN(**base_cfg)
    torch.manual_seed(21)
    corrected = MaskedInfoGCN(**base_cfg)
    a, b = baseline.state_dict(), corrected.state_dict()
    paired, mismatched, missing_mapped = 0, [], []
    for key, value in a.items():
        target = aliases(key)
        if target not in b or b[target].shape != value.shape:
            missing_mapped.append((key, target))
        else:
            paired += 1
            if not torch.equal(value, b[target]):
                mismatched.append((key, target))
    assert paired >= 200 and not mismatched and not missing_mapped, (
        f"Seeded initialization mismatch: paired={paired}, mismatched={mismatched[:5]}, missing={missing_mapped[:5]}"
    )

    source_runs = {
        21: "exp10rerun_seed21", 42: "exp10_seed42", 7: "exp10_seed7", 13: "exp10_seed13"
    }
    baseline_runs = {}
    expected_model_config = {k: base_cfg[k] for k in ("num_classes", "in_channels", "num_joints", "base_channels", "dropout", "head_dropout", "use_ib", "latent_dim")}
    for seed, name in source_runs.items():
        summary_p = ROOT / "logs/exp10rerun" / name / "best_validation_metrics.json"
        history_p = ROOT / "logs/exp10rerun" / name / "training_log.csv"
        ckpt_p = ROOT / "runs/exp10rerun" / name / "best_model.pt"
        assert summary_p.exists() and history_p.exists() and ckpt_p.exists(), f"Missing baseline artifacts for seed {seed}"
        summary = json.loads(summary_p.read_text())
        assert summary["seed"] == seed and summary["selection_metric"] == "val_loss"
        assert summary["epochs_configured"] == 100 and summary["batch_size"] == 32
        assert summary["learning_rate"] == 0.001 and summary["weight_decay"] == 0.0001
        assert summary["scheduler"] == "plateau" and summary["model_config"] == expected_model_config
        baseline_runs[str(seed)] = {"source_run": name, "best_epoch": summary["best_epoch"],
                                    "last_epoch": summary["last_epoch"], "early_stopped": summary["stopped_early"],
                                    "checkpoint_sha256": sha(ckpt_p), "history_sha256": sha(history_p)}

    result = {
        "status": "PASS — training may begin",
        "config_sha256": sha(OUT / "comparison_config.json"),
        "metadata_sha256": sha(metadata_path),
        "split_counts": counts,
        "class_count": len(classes),
        "feature_paths_present": feature_count,
        "train_val_feature_check": observed,
        "test_split_policy": "Metadata/path existence checked; test feature contents not opened during preflight.",
        "baseline_runs": baseline_runs,
        "seeded_initialization": {"paired_state_tensors": paired, "mismatches": 0},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "isolated_output_paths": {"baseline": str(OUT / "baseline"), "mask_correct": str(OUT / "mask_correct")},
        "protected_source_paths_unchanged_by_design": [str(ROOT / "experiments/exp10rerun"), str(ROOT / "runs/exp10rerun"), str(ROOT / "logs/exp10rerun")],
    }
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
