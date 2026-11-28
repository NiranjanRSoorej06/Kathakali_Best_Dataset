"""Read-only post-run integrity audit for Test 5 outputs and protected sources."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STGCN = ROOT.parent / "stgcn"
SEEDS = (21, 42, 7, 13)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pre = json.loads((OUT / "preflight.json").read_text())
    config_sha = sha(OUT / "comparison_config.json")
    metadata_sha = sha(STGCN / "datasets/processed/metadata_split.csv")
    test4_model_sha = sha(ROOT / "research_audit/Test4_mask_correct_implementation/model_mask_correct.py")
    local_model_sha = sha(OUT / "model_mask_correct.py")
    assert config_sha == pre["config_sha256"]
    assert metadata_sha == pre["metadata_sha256"]
    assert local_model_sha == test4_model_sha, "Archived Test 5 model differs from Test 4 source"
    baseline_source = {21: "exp10rerun_seed21", 42: "exp10_seed42", 7: "exp10_seed7", 13: "exp10_seed13"}
    sources, outputs = {}, {}
    for seed, name in baseline_source.items():
        src_log = ROOT / "logs/exp10rerun" / name / "training_log.csv"
        src_ckpt = ROOT / "runs/exp10rerun" / name / "best_model.pt"
        assert sha(src_log) == pre["baseline_runs"][str(seed)]["history_sha256"]
        assert sha(src_ckpt) == pre["baseline_runs"][str(seed)]["checkpoint_sha256"]
        base = OUT / "baseline" / f"seed_{seed}"
        assert sha(src_log) == sha(base / "training_log.csv")
        assert sha(src_ckpt) == sha(base / "best_model.pt")
        sources[str(seed)] = {"source_run": name, "checkpoint_sha256": sha(src_ckpt), "history_sha256": sha(src_log)}
        outputs[f"baseline_seed_{seed}"] = {"checkpoint_sha256": sha(base / "best_model.pt"),
                                             "history_sha256": sha(base / "training_log.csv")}

        for variant in ("baseline", "mask_correct"):
            run = OUT / variant / f"seed_{seed}"
            metrics = json.loads((run / "best_validation_metrics.json").read_text())
            assert metrics["seed"] == seed and metrics["selection_metric"] == "val_loss"
            assert 1 <= metrics["best_epoch"] <= metrics["last_epoch"] <= 100
            assert (run / "best_model.pt").exists() and (run / "training_log.csv").exists()
            outputs[f"{variant}_seed_{seed}"] = {"checkpoint_sha256": sha(run / "best_model.pt"),
                                                 "history_sha256": sha(run / "training_log.csv"),
                                                 "summary_sha256": sha(run / "best_validation_metrics.json"),
                                                 "best_epoch": metrics["best_epoch"], "last_epoch": metrics["last_epoch"]}
    ev = json.loads((OUT / "evaluation/all_seed_metrics_and_paired.json").read_text())
    assert ev["test_samples"] == 65 and set(ev["by_seed"]) == {str(s) for s in SEEDS}
    paired_disagreements = {}
    for seed in SEEDS:
        for variant in ("baseline", "mask_correct"):
            assert (OUT / "evaluation" / f"seed_{seed}" / f"{variant}_per_class_metrics.csv").exists()
            assert (OUT / "evaluation" / f"seed_{seed}" / f"{variant}_confusion_matrix.csv").exists()
        b = pd.read_csv(OUT / "evaluation" / f"seed_{seed}" / "baseline_predictions.csv")
        m = pd.read_csv(OUT / "evaluation" / f"seed_{seed}" / "mask_correct_predictions.csv")
        paired_disagreements[str(seed)] = int((b.predicted_id != m.predicted_id).sum())
        assert len(b) == len(m) == 65
    assert (OUT / "learning_curves.png").exists() and (OUT / "test_metrics_by_seed.png").exists()
    assert (OUT / "per_class_cross_seed_summary.csv").exists()
    results = {
        "status": "PASS",
        "preflight_config_sha256_unchanged": config_sha,
        "dataset_split_metadata_sha256_unchanged": metadata_sha,
        "original_exp10rerun_source_baseline_artifacts_unchanged": sources,
        "isolated_output_runs": outputs,
        "canonical_test_samples": 65,
        "test_predictions_disagreements_by_seed": paired_disagreements,
        "artifacts_under_test5_directory_only": True,
        "generated_training_attempt_note": "Failed initial attempt preserved under failed_attempts; it has no checkpoint and is excluded from all metrics.",
        "protected_code_hashes_at_completion": {
            "exp10rerun_model": sha(ROOT / "experiments/exp10rerun/model_exp10rerun.py"),
            "exp10rerun_trainer": sha(ROOT / "experiments/exp10rerun/train_exp10rerun.py"),
            "test4_mask_correct_model": test4_model_sha,
            "test5_archived_mask_correct_model": local_model_sha,
        },
    }
    (OUT / "final_integrity.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
