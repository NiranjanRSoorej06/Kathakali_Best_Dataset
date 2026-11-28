"""Stage 1: New40 integrity and checkpoint compatibility checks only; no inference."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research_audit"
TEST5 = AUDIT / "Test5_mask_correction_comparison"
TEST2 = ROOT / "testing" / "test2"
BASELINE_RESULTS = ROOT / "testing" / "test2_multiseed" / "results"
MANIFEST = ROOT.parent / "stgcn" / "testing" / "new40" / "processed" / "manifest.csv"
OUT = Path(__file__).resolve().parent
SEEDS = [21, 42, 7, 13]

sys.path.insert(0, str(TEST2 / "scripts"))
sys.path.insert(0, str(ROOT))
from common_test2 import EXPECTED_VIDEOS, load_canonical_class_names  # noqa: E402
from evaluate_exp10rerun_test2 import validate_and_load_samples  # noqa: E402
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_bundle_sha256(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for sample in samples:
        path = Path(sample["npz_path"])
        h.update(path.name.encode())
        h.update(sha256(path).encode())
    return h.hexdigest()


def main() -> None:
    checks: dict = {"stage": 1, "inference_performed": False, "seeds": {}, "status": "FAIL"}
    names = load_canonical_class_names()
    samples, invalid = validate_and_load_samples(MANIFEST, names)
    checks["data"] = {
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "feature_bundle_sha256": feature_bundle_sha256(samples),
        "expected_videos": EXPECTED_VIDEOS, "loaded_videos": len(samples),
        "invalid_or_unindexed_count": len(invalid),
        "video_names_sha256": hashlib.sha256("\n".join(x["video_name"] for x in samples).encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(np.asarray([x["label_index"] for x in samples], dtype=np.int64).tobytes()).hexdigest(),
        "feature_shapes": sorted({list(x["features"].shape).__repr__() for x in samples}),
        "frame_valid_lengths": sorted({int(x["frame_valid"].sum()) for x in samples}),
        "class_count": len(names), "canonical_class_names": names,
        "invalid_examples": invalid[:3],
    }
    if len(samples) != EXPECTED_VIDEOS or invalid:
        raise RuntimeError(f"New40 data preflight failed: {len(samples)} samples, {len(invalid)} invalid")

    baseline_names = []
    for seed in SEEDS:
        ckpt_path = TEST5 / "mask_correct" / f"seed_{seed}" / "best_model.pt"
        baseline_dir = BASELINE_RESULTS / f"exp10_seed{seed}"
        if seed == 21:
            baseline_dir = ROOT / "testing" / "test2" / "results"
        metrics_path = baseline_dir / "external_eval_metrics.json"
        predictions_path = baseline_dir / "predictions.csv"
        if not all(p.exists() for p in (ckpt_path, metrics_path, predictions_path)):
            raise FileNotFoundError(f"Missing checkpoint or established baseline outputs for seed {seed}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if int(ckpt.get("seed", -1)) != seed:
            raise ValueError(f"Checkpoint seed mismatch for {seed}: {ckpt.get('seed')}")
        if list(ckpt.get("class_names", [])) != names:
            raise ValueError(f"Checkpoint class order mismatch for seed {seed}")
        model = MaskedInfoGCN(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        model_config = dict(ckpt["model_config"])
        checkpoint_epoch = int(ckpt["epoch"])
        del model, ckpt

        base_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        base_pred = pd.read_csv(predictions_path)
        if len(base_pred) != EXPECTED_VIDEOS or int(base_metrics["evaluated_videos"]) != EXPECTED_VIDEOS:
            raise ValueError(f"Baseline seed {seed} does not cover exactly 40 New40 rows")
        if not {"video_name", "true_id", "predicted_id"}.issubset(base_pred.columns):
            raise ValueError(f"Baseline seed {seed} prediction schema is missing required columns")
        base_pred = base_pred.set_index("video_name").reindex([x["video_name"] for x in samples])
        if base_pred[["true_id", "predicted_id"]].isna().any().any():
            raise ValueError(f"Baseline seed {seed} video identities do not match the New40 manifest")
        if base_pred["true_id"].astype(int).tolist() != [x["label_index"] for x in samples]:
            raise ValueError(f"Baseline seed {seed} labels do not match New40 manifest")
        reported_seed = base_metrics.get("checkpoint_selection", {}).get("seed")
        if reported_seed is not None and int(reported_seed) != seed:
            raise ValueError(f"Baseline metric seed mismatch for {seed}")
        baseline_names.append(base_pred.index.tolist())
        checks["seeds"][str(seed)] = {
            "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256(ckpt_path),
            "checkpoint_epoch": checkpoint_epoch,
            "strict_state_dict_load": True, "parameter_count": n_params,
            "model_config": model_config,
            "baseline_metrics": str(metrics_path), "baseline_predictions": str(predictions_path),
            "baseline_checkpoint": base_metrics.get("checkpoint"),
            "baseline_accuracy": float(base_metrics["accuracy"]),
            "baseline_prediction_rows": len(base_pred),
        }
    if any(items != baseline_names[0] for items in baseline_names[1:]):
        raise ValueError("Baseline seeds do not share the exact same video ordering")

    checks["protocol"] = {
        "comparison": "same-seed EXP10RERUN baseline vs Test 5 mask-correct checkpoint",
        "evaluation_inputs": "existing ST-GCN testing/new40/processed/manifest.csv and referenced NPZs",
        "loader": "existing testing/test2 validate_and_load_samples; existing 100x50x3 feature layout and frame_valid masks",
        "label_mapping": "canonical metadata_split.csv ordered 41-class mapping",
        "loss": "comparable; both variants use InfoGCNLoss with use_ib=False (cross entropy)",
        "individual_video_predictions_written": False,
        "test1_single_exp10_results_preserved": True,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    checks["status"] = "PASS"
    (OUT / "preflight.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": checks["status"], "inference_performed": False,
                      "videos": len(samples), "seeds": SEEDS, "preflight": str(OUT / 'preflight.json')}, indent=2))


if __name__ == "__main__":
    main()
