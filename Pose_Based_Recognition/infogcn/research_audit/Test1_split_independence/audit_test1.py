#!/usr/bin/env python3
"""Read-only Test 1 clip-level split independence audit.

Reads canonical metadata, source videos, and processed pose features. Writes only
to the directory containing this script. No model code is imported or run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # infogcn
SPLIT_CSV = ROOT.parent / "stgcn/datasets/processed/metadata_split.csv"
INVENTORY_CSV = ROOT.parents[1] / "metadata/video_validation_inventory.csv"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    vals = dct.flatten()
    med = np.median(vals[1:])
    bits = vals > med
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def decode_samples(path: Path, fractions=(.05, .16, .27, .38, .50, .62, .73, .84, .95)):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open source")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    targets = sorted(set(max(0, min(count - 1, int(frac * max(0, count - 1)))) for frac in fractions))
    frames = {}
    # Seek to sparse points instead of decoding every frame in long source clips.
    for target in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok:
            frames[target] = frame
    cap.release()
    fractions_actual = [i / max(1, count - 1) for i in targets if i in frames]
    body_hashes, full_hashes, background_hashes = [], [], []
    thumbs = []
    for i in targets:
        if i not in frames:
            continue
        frame = frames[i]
        h, w = frame.shape[:2]
        # Fixed center crop captures the performer; corner strips provide a
        # coarse background/setup fingerprint. Neither is an identity signal.
        body = frame[int(.08*h):int(.96*h), int(.22*w):int(.78*w)]
        corners = [
            frame[:int(.24*h), :int(.22*w)], frame[:int(.24*h), int(.78*w):],
            frame[int(.78*h):, :int(.22*w)], frame[int(.78*h):, int(.78*w):],
        ]
        corner_tiles = [cv2.resize(x, (64, 32), interpolation=cv2.INTER_AREA) for x in corners]
        corner_img = np.vstack([np.hstack(corner_tiles[:2]), np.hstack(corner_tiles[2:])])
        body_hashes.append(phash64(body))
        full_hashes.append(phash64(frame))
        background_hashes.append(phash64(corner_img))
        thumb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(thumb)
        thumb.thumbnail((150, 140))
        thumbs.append((i / max(1, count - 1), thumb))
    return {
        "count": count, "fps": fps, "width": width, "height": height,
        "fractions": fractions_actual, "body_hashes": body_hashes,
        "full_hashes": full_hashes, "background_hashes": background_hashes,
        "thumbs": thumbs,
    }


def pose_summary(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data:
            return None
        x = data["features"].astype(np.float32)
        mask = data["frame_valid"].astype(bool) if "frame_valid" in data else np.ones(len(x), bool)
    return x, mask


def pose_pair(a, b):
    if a is None or b is None:
        return math.nan, math.nan
    xa, ma = a
    xb, mb = b
    n = min(len(xa), len(xb))
    m = ma[:n] & mb[:n]
    if not m.any():
        return math.nan, math.nan
    aa, bb = xa[:n][m, :, :2].reshape(-1), xb[:n][m, :, :2].reshape(-1)
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    corr = float(np.corrcoef(aa, bb)[0, 1]) if np.std(aa) and np.std(bb) else math.nan
    return rmse, corr


def source_bundle(path: str):
    p = Path(path)
    for part in p.parts:
        if part.startswith("Videos_Cropped") and part != "Videos_Cropped":
            return part
    if "Video_Collection_first_person" in p.parts:
        return "Video_Collection_first_person"
    return "other"


def main():
    split_rows = read_csv(SPLIT_CSV)
    inventory_rows = read_csv(INVENTORY_CSV)
    inventory = {(Path(r["path"]).parent.name, Path(r["path"]).name): r for r in inventory_rows}
    rows = []
    decoded = {}
    poses = {}
    errors = []
    for r in split_rows:
        video_name, cls = r["video_name"], r["class_name"]
        inv = inventory.get((cls, video_name))
        if inv is None:
            errors.append({"video_name": video_name, "error": "not found in inventory"})
            continue
        path = Path(inv["path"])
        bundle = source_bundle(inv["path"])
        feature_path = ROOT.parent / "stgcn/datasets/processed" / Path(r["feature_path"]).relative_to("stgcn/processed")
        rec = {
            "video_name": video_name, "class_name": cls, "label_index": r["label_index"],
            "split": r["split"], "source_path": str(path), "source_bundle": bundle,
            "source_directory": str(path.parent), "file_exists": path.is_file(),
            "feature_path": str(feature_path), "feature_exists": feature_path.is_file(),
            "metadata_width": r["original_width"], "metadata_height": r["original_height"],
            "metadata_fps": r["fps"], "metadata_frames": r["total_frames"],
            "valid_frames": r["valid_frames"], "normalization_valid_frames": r["normalization_valid_frames"],
            "reference_shoulder_width": r["reference_shoulder_width"],
            "inventory_duration": inv["duration_seconds"], "inventory_readable": inv["readable"],
        }
        if not path.is_file():
            errors.append({"video_name": video_name, "error": "source file missing"})
            rows.append(rec)
            continue
        try:
            rec["sha256"] = sha256(path)
            d = decode_samples(path)
            decoded[(cls, video_name)] = d
            rec.update({
                "decoded_width": d["width"], "decoded_height": d["height"],
                "decoded_fps": d["fps"], "decoded_frames": d["count"],
                "decoded_duration": d["count"] / d["fps"] if d["fps"] else math.nan,
                "sampled_frames": len(d["thumbs"]),
            })
            poses[(cls, video_name)] = pose_summary(feature_path)
            rows.append(rec)
        except Exception as e:
            errors.append({"video_name": video_name, "error": f"{type(e).__name__}: {e}"})
            rows.append(rec)

    # All same-class pairs are tested. This covers likely repeated-take and
    # duplicate candidates while avoiding treating common backgrounds across
    # different gestures as source-overlap evidence.
    classes = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            classes[row["class_name"]].append(row)
    pairs = []
    for cls, group in classes.items():
        for i, a in enumerate(group):
            for b in group[i+1:]:
                da, db = decoded[(cls, a["video_name"])], decoded[(cls, b["video_name"])]
                na, nb = min(len(da["body_hashes"]), len(db["body_hashes"])), min(len(da["full_hashes"]), len(db["full_hashes"]))
                body_direct = [hamming(da["body_hashes"][k], db["body_hashes"][k]) for k in range(na)]
                full_direct = [hamming(da["full_hashes"][k], db["full_hashes"][k]) for k in range(nb)]
                body_dist = np.array([[hamming(x, y) for y in db["body_hashes"]] for x in da["body_hashes"]])
                bg_dist = [hamming(x, y) for x, y in zip(da["background_hashes"], db["background_hashes"])]
                # Longest monotonic match chain with pHash distance <= 4.
                match = body_dist <= 4
                dp = np.zeros(match.shape, dtype=np.int16)
                for x in range(match.shape[0]):
                    for y in range(match.shape[1]):
                        if match[x, y]:
                            dp[x, y] = 1 + (int(dp[:x, :y].max()) if x and y else 0)
                chain = int(dp.max()) if dp.size else 0
                rmse, corr = pose_pair(poses.get((cls, a["video_name"])), poses.get((cls, b["video_name"])))
                same_sha = a["sha256"] == b["sha256"]
                split_cross = a["split"] != b["split"]
                same_bundle = a["source_bundle"] == b["source_bundle"]
                same_dir = a["source_directory"] == b["source_directory"]
                min_mean_body = float(body_dist.min(axis=1).mean()) if body_dist.size else math.nan
                close_body = int((body_dist.min(axis=1) <= 4).sum()) if body_dist.size else 0
                category = "unknown"
                basis = "No strong pair-specific relation signal"
                if same_sha:
                    category, basis = "confirmed", "Byte-identical source file (SHA-256 match)"
                elif chain >= 7 and close_body >= 7 and (not math.isnan(rmse) and rmse < .03):
                    category, basis = "high-confidence", "Long monotonic near-frame match chain plus very close pose sequence; requires human source review"
                elif chain >= 4 or (not math.isnan(rmse) and rmse < .06 and corr > .97):
                    category, basis = "possible", "Some aligned visual/pose similarity; choreography and common setup can explain it"
                elif same_dir and abs(float(a["metadata_frames"]) - float(b["metadata_frames"])) <= 2 and abs(float(a["metadata_fps"]) - float(b["metadata_fps"])) < .02:
                    category, basis = "possible", "Same storage directory and near-identical duration/FPS; not proof of shared source"
                elif same_bundle:
                    category, basis = "unknown", "Same broad source bundle only; bundle is not a verified session label"
                pairs.append({
                    "class_name": cls, "video_a": a["video_name"], "split_a": a["split"],
                    "path_a": a["source_path"], "video_b": b["video_name"], "split_b": b["split"],
                    "path_b": b["source_path"], "cross_split": split_cross,
                    "source_bundle_a": a["source_bundle"], "source_bundle_b": b["source_bundle"],
                    "same_bundle": same_bundle, "same_source_directory": same_dir,
                    "same_sha256": same_sha, "frames_a": a["decoded_frames"], "frames_b": b["decoded_frames"],
                    "fps_a": a["decoded_fps"], "fps_b": b["decoded_fps"],
                    "pHash_body_aligned_mean": float(np.mean(body_direct)) if body_direct else math.nan,
                    "pHash_full_aligned_mean": float(np.mean(full_direct)) if full_direct else math.nan,
                    "pHash_body_cross_min_mean": min_mean_body,
                    "pHash_body_samples_with_match_le4": close_body,
                    "pHash_body_monotonic_chain_le4": chain,
                    "pHash_background_aligned_mean": float(np.mean(bg_dist)) if bg_dist else math.nan,
                    "pose_xy_rmse": rmse, "pose_xy_correlation": corr,
                    "evidence_category": category, "evidence_basis": basis,
                })

    # Candidate related-video groups are connected components of high-confidence
    # and possible same-class edges. Confirmed file identity is tracked separately.
    by_split = defaultdict(Counter)
    for row in rows:
        by_split[row["source_bundle"]][row["split"]] += 1
    bundle_rows = []
    for bundle, c in sorted(by_split.items()):
        bundle_rows.append({"source_bundle": bundle, "train": c["train"], "val": c["val"], "test": c["test"],
                            "total": sum(c.values()), "spans_multiple_splits": sum(v > 0 for v in c.values()) > 1,
                            "interpretation": "container only; not a verified recording/session group"})
    class_dim = defaultdict(Counter)
    class_split = defaultdict(Counter)
    for r in rows:
        class_dim[r["class_name"]][f'{r["metadata_width"]}x{r["metadata_height"]}'] += 1
        class_split[r["class_name"]][r["split"]] += 1
    class_rows = []
    for cls in sorted(class_dim):
        class_rows.append({"class_name": cls, "total": sum(class_dim[cls].values()),
                           "dimension_groups": ";".join(f"{k}:{v}" for k,v in sorted(class_dim[cls].items())),
                           "num_dimension_groups": len(class_dim[cls]),
                           "train": class_split[cls]["train"], "val": class_split[cls]["val"], "test": class_split[cls]["test"]})

    # Use category-qualified edges and compute per-class connected components.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for p in pairs:
        if p["evidence_category"] in ("high-confidence", "possible"):
            union((p["class_name"], p["video_a"]), (p["class_name"], p["video_b"]))
    comps = defaultdict(list)
    for key in list(parent): comps[find(key)].append(key)
    group_rows = []
    for gidx, members in enumerate(comps.values(), 1):
        if len(members) < 2: continue
        member_rows = [next(r for r in rows if r["class_name"] == c and r["video_name"] == n) for c,n in members]
        splits = sorted({r["split"] for r in member_rows})
        edgecats = [p["evidence_category"] for p in pairs if (p["class_name"],p["video_a"]) in members and (p["class_name"],p["video_b"]) in members]
        group_rows.append({"candidate_group_id": f"CG{gidx:03d}", "class_name": members[0][0],
                           "members": ";".join(sorted(n for _,n in members)),
                           "splits": ";".join(splits), "num_clips": len(members),
                           "spans_multiple_splits": len(splits)>1,
                           "strongest_category": "high-confidence" if "high-confidence" in edgecats else "possible",
                           "note": "Candidate component; transitive links can over-group similar choreography; review edges individually"})

    pair_fields = list(pairs[0].keys()) if pairs else []
    write_csv(HERE / "clip_manifest_audit.csv", rows, list(rows[0].keys()))
    write_csv(HERE / "same_class_pairwise_evidence.csv", pairs, pair_fields)
    write_csv(HERE / "source_bundle_split_counts.csv", bundle_rows, list(bundle_rows[0].keys()))
    write_csv(HERE / "class_split_dimension_counts.csv", class_rows, list(class_rows[0].keys()))
    write_csv(HERE / "candidate_related_groups.csv", group_rows, list(group_rows[0].keys()) if group_rows else ["candidate_group_id","class_name","members","splits","num_clips","spans_multiple_splits","strongest_category","note"])
    write_csv(HERE / "audit_errors.csv", errors, ["video_name", "error"])

    cross_pairs = [p for p in pairs if p["cross_split"]]
    counts = Counter(p["evidence_category"] for p in cross_pairs)
    crossing_groups = [g for g in group_rows if g["spans_multiple_splits"]]
    summary = {
        "split_manifest_rows": len(split_rows), "inventory_rows": len(inventory_rows),
        "audited_source_files": len([r for r in rows if r.get("sha256")]),
        "source_and_feature_errors": errors,
        "split_counts": dict(Counter(r["split"] for r in split_rows)),
        "class_count": len(set(r["class_name"] for r in split_rows)),
        "source_bundle_count": len(bundle_rows),
        "source_bundles_spanning_splits": sum(b["spans_multiple_splits"] for b in bundle_rows),
        "clips_in_source_bundles_spanning_splits": sum(b["total"] for b in bundle_rows if b["spans_multiple_splits"]),
        "same_class_pair_count": len(pairs), "cross_split_same_class_pair_count": len(cross_pairs),
        "cross_split_pair_evidence_counts": dict(counts),
        "candidate_groups_total": len(group_rows),
        "candidate_groups_spanning_splits": len(crossing_groups),
        "clips_in_candidate_groups_spanning_splits": len({n for g in crossing_groups for n in g["members"].split(";")}),
        "dimension_groups": dict(Counter(f'{r["metadata_width"]}x{r["metadata_height"]}' for r in rows)),
        "classes_multi_dimension": sum(r["num_dimension_groups"] > 1 for r in class_rows),
        "classes_single_dimension": sum(r["num_dimension_groups"] == 1 for r in class_rows),
        "visual_samples_per_video": 9,
        "pairwise_method": "same-class pairwise pHash comparison on 9 body/full/background samples; normalized-time pose XY RMSE/correlation; exact SHA-256; all thresholds are screening heuristics, not proof",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save a review sheet for the most suspicious cross-split candidates.
    ranked = sorted(cross_pairs, key=lambda p: (p["evidence_category"] == "high-confidence", p["evidence_category"] == "possible", -p["pHash_body_monotonic_chain_le4"], -p["pHash_body_samples_with_match_le4"], -p["pose_xy_correlation"] if not math.isnan(p["pose_xy_correlation"]) else 1))
    chosen = [p for p in ranked if p["evidence_category"] in ("high-confidence", "possible")][:40]
    if chosen:
        cell_w, cell_h = 150, 140
        sheet = Image.new("RGB", (cell_w*8, (cell_h+54)*len(chosen)), "white")
        draw = ImageDraw.Draw(sheet)
        for ix, p in enumerate(chosen):
            y = ix*(cell_h+54)
            draw.text((4,y+2), f'{p["class_name"]}: {p["video_a"]} [{p["split_a"]}] vs {p["video_b"]} [{p["split_b"]}] | {p["evidence_category"]} | chain={p["pHash_body_monotonic_chain_le4"]} poseRMSE={p["pose_xy_rmse"]:.4f}', fill="black")
            for col, name in enumerate((p["video_a"], p["video_b"])):
                d = decoded[(p["class_name"], name)]
                for j, (frac, thumb) in enumerate(d["thumbs"][:4]):
                    sheet.paste(thumb, (col*3*cell_w+j*cell_w, y+22))
                    draw.text((col*3*cell_w+j*cell_w+3, y+22), f'{frac:.0%}', fill="yellow", stroke_width=1, stroke_fill="black")
        sheet.save(HERE / "candidate_pair_review.png")
    summary["review_sheet_pair_count"] = len(chosen)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
