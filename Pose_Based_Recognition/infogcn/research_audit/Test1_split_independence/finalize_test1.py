#!/usr/bin/env python3
"""Summarize already-generated Test 1 CSVs; does not read source videos."""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(name):
    with (HERE/name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write(name, rows, fields):
    with (HERE/name).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

clips=read("clip_manifest_audit.csv")
pairs=read("same_class_pairwise_evidence.csv")
groups=read("candidate_related_groups.csv")
bundles=read("source_bundle_split_counts.csv")
classes=read("class_split_dimension_counts.csv")
manual=read("manual_visual_review.csv")
cross=[p for p in pairs if p["cross_split"]=="True"]
possible=[p for p in cross if p["evidence_category"]=="possible"]
# Add manually reviewed visual candidates as a transparent overlay. The raw
# pairwise classifier remains unchanged in same_class_pairwise_evidence.csv.
machine_edges={(p["class_name"], *sorted((p["video_a"],p["video_b"]))) for p in possible}
manual_edges={(p["class_name"], *sorted((p["video_a"],p["video_b"]))) for p in manual}
manual_extra=manual_edges-machine_edges
for g in groups:
    if g["spans_multiple_splits"]!="True": continue
    members=set(g["members"].split(";"))
    for p in manual:
        if p["class_name"]!=g["class_name"]: continue
        a,b=p["video_a"],p["video_b"]
        if a in members and b not in members: members.add(b)
        elif b in members and a not in members: members.add(a)
    g["members"]=";".join(sorted(members)); g["num_clips"]=str(len(members))
write("candidate_related_groups.csv", groups, list(groups[0].keys()))
cross_groups=[g for g in groups if g["spans_multiple_splits"]=="True"]
group_clip_names=sorted({n for g in cross_groups for n in g["members"].split(";")})
possible_clip_names=sorted({n for p in possible for n in (p["video_a"],p["video_b"])})
combined_possible_clip_names=sorted(set(possible_clip_names)|{n for p in manual for n in (p["video_a"],p["video_b"])})

split_counts=Counter(r["split"] for r in clips)
class_counts=defaultdict(Counter)
for r in clips: class_counts[r["class_name"]][r["split"]]+=1
dimension_counts=Counter(f'{r["metadata_width"]}x{r["metadata_height"]}' for r in clips)
hash_counts=Counter(r.get("sha256","") for r in clips)
exact_hash_groups={h:sum(1 for r in clips if r.get("sha256")==h) for h,n in hash_counts.items() if h and n>1}
suffix_split=defaultdict(Counter)
for r in clips:
    m=re.search(r"_(\d+)\.mp4$",r["video_name"],re.I)
    suffix_split[m.group(1) if m else "none"][r["split"]]+=1
suffix_rows=[{"suffix_number":k,"train":v["train"],"val":v["val"],"test":v["test"],"total":sum(v.values())} for k,v in sorted(suffix_split.items(), key=lambda kv:(kv[0]=="none",int(kv[0]) if kv[0].isdigit() else 999))]
write("filename_suffix_split_counts.csv", suffix_rows, ["suffix_number","train","val","test","total"])

# Pair evidence mechanism counts for transparent screening interpretation.
mechanisms=Counter()
for p in possible:
    if int(p["pHash_body_monotonic_chain_le4"])>=4: mechanisms["body_phash_monotonic_chain_ge4"]+=1
    if p["pose_xy_correlation"] not in ("", "nan"):
        try:
            if float(p["pose_xy_rmse"])<.06 and float(p["pose_xy_correlation"])>.97:
                mechanisms["pose_rmse_lt_0.06_and_corr_gt_0.97"]+=1
        except ValueError: pass
    if p["same_source_directory"]=="True" and int(p["frames_a"]) and int(p["frames_b"]):
        try:
            if abs(int(p["frames_a"])-int(p["frames_b"]))<=2 and abs(float(p["fps_a"])-float(p["fps_b"]))<.02:
                mechanisms["same_directory_near_duration_and_fps"]+=1
        except ValueError: pass

summary={
 "audit_label":"Test 1: Clip-Level Split Independence Audit",
 "canonical_clips":len(clips), "audited_source_files":sum(bool(r.get("sha256")) for r in clips),
 "feature_files_present":sum(r["feature_exists"]=="True" for r in clips),
 "audit_errors":read("audit_errors.csv"),
 "split_counts":dict(split_counts), "classes":len(class_counts),
 "classes_present_in_all_splits":sum(all(class_counts[c][s]>0 for s in ("train","val","test")) for c in class_counts),
 "per_split_per_class_support_counts":{s:dict(Counter(c[s] for c in class_counts.values())) for s in ("train","val","test")},
 "dimension_group_counts":dict(dimension_counts),
 "classes_multi_dimension":sum(int(c["num_dimension_groups"])>1 for c in classes),
 "classes_single_dimension":sum(int(c["num_dimension_groups"])==1 for c in classes),
 "same_class_pairs_tested":len(pairs), "cross_split_same_class_pairs_tested":len(cross),
 "cross_split_pair_evidence_counts":dict(Counter(p["evidence_category"] for p in cross)),
 "cross_split_possible_pairs":len(possible), "clips_in_possible_cross_split_pairs":len(possible_clip_names),
 "manual_visual_pairs_reviewed":len(manual), "manual_additional_possible_edges":len(manual_extra),
 "combined_possible_cross_split_pair_leads":len(possible)+len(manual_extra),
 "clips_in_combined_possible_cross_split_pair_leads":len(combined_possible_clip_names),
 "cross_split_candidate_components":len(cross_groups), "clips_in_cross_split_candidate_components":len(group_clip_names),
 "candidate_component_evidence_counts":dict(Counter(g["strongest_category"] for g in cross_groups)),
 "source_directory_bundles":len(bundles), "bundles_with_multiple_splits":sum(b["spans_multiple_splits"]=="True" for b in bundles),
 "clips_in_bundles_with_multiple_splits":sum(int(b["total"]) for b in bundles if b["spans_multiple_splits"]=="True"),
 "exact_duplicate_sha_groups":exact_hash_groups,
 "cross_split_possible_evidence_mechanism_counts_nonexclusive":dict(mechanisms),
 "samples_per_clip":9,
 "important_scope":"Pairwise perceptual/pose screening was limited to same-class pairs. All 430 files were inventoried and hashed. Cross-class session linkage is represented only by source-container tags and remains unknown without source/session metadata.",
 "threshold_note":"Possible and component counts are screening output, not verified overlap counts. pHash and pose similarity can reflect repeated choreography or common setup. No pair met the strict confirmed or high-confidence rule used by the script."
}
(HERE/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

report=f'''# Test 1: Clip-Level Split Independence Audit

## Scope and result

Read-only audit of all {len(clips)} canonical videos and their processed pose files. The manifest has {split_counts['train']} train, {split_counts['val']} validation, and {split_counts['test']} test clips across {len(class_counts)} classes; {summary['classes_present_in_all_splits']} classes occur in every split. All {summary['audited_source_files']} source files were found, hashed, and sampled at nine normalized timestamps for coarse visual/perceptual screening. No training, tuning, data/split/checkpoint edits, or New40 per-video prediction inspection was performed.

## Quantified evidence

| Evidence view | Count | Interpretation |
|---|---:|---|
| Cross-split same-class pair comparisons | {len(cross)} | All possible pairs among same-class clips assigned to different splits; this is the comparison denominator. |
| Confirmed pairs | {summary['cross_split_pair_evidence_counts'].get('confirmed',0)} | Confirmed only by identical source-file SHA-256. This does not detect re-encoded duplicates. |
| High-confidence pairs | {summary['cross_split_pair_evidence_counts'].get('high-confidence',0)} | None met the strict combined frame/pose screen. This is not evidence of independence. |
| Automated possible pair candidates | {len(possible)} | Perceptual/pose screen; requires source-level or human temporal review. |
| Additional manually reviewed possible edges | {len(manual_extra)} | Sparse visual review of five named pairs upgraded these three pairs from unknown to possible; see `manual_visual_review.csv`. |
| Combined possible pair leads | {len(possible)+len(manual_extra)} | Automated candidates plus manual additions; not confirmed overlap. |
| Possible candidate components spanning splits | {len(cross_groups)} | Connected components after the manual overlay; transitive similarity can over-group clips. |
| Clips in those components | {len(group_clip_names)} | Candidate exposure count, not a confirmed related-clip count. |
| Broad source-directory bundles spanning splits | {summary['bundles_with_multiple_splits']} of {len(bundles)} | All {summary['clips_in_bundles_with_multiple_splits']} clips sit in directory containers that contain multiple splits. Directory names are not verified session IDs. |

Of the {len(cross)} cross-split same-class pairs, {summary['cross_split_pair_evidence_counts'].get('unknown',0)} were classified unknown by the automated screen (no pair-specific signal beyond ordinary same-class membership), and {len(possible)} were possible. A separate sparse visual review inspected five named pairs, adding {len(manual_extra)} possible edges. The automated possible screen was triggered by a center-crop perceptual-hash monotonic match chain of at least four of nine samples, or very high pose-coordinate correlation/low error, or a same-source-directory pair with nearly identical frame count and FPS. These are deliberately broad candidate rules, not overlap determinations.

## Split, naming, directory, and setup patterns

The canonical split is clip-level: every class has only one or two validation clips and one or two test clips. Filename suffixes range across numbered recordings; suffix numbers are spread across splits and are not known to represent chronology or source IDs. See `filename_suffix_split_counts.csv` for suffix-by-split counts.

All seven broad source-directory containers cross splits: Videos_Cropped1–6 and Video_Collection_first_person. Their per-split counts are in `source_bundle_split_counts.csv`. Treating each as a single recording group would put all {summary['clips_in_bundles_with_multiple_splits']} clips in potentially crossing containers, but that interpretation is **unknown**, not confirmed: these directories may be collection/storage batches rather than sessions.

The metadata exposes four image-dimension/setup proxies: {', '.join(f'{k}: {v}' for k,v in sorted(dimension_counts.items()))}. There are {summary['classes_multi_dimension']} multi-dimension classes and {summary['classes_single_dimension']} single-dimension classes. FPS, frame count, duration, source directory, shoulder-width normalization, and valid-frame counts are recorded per clip in `clip_manifest_audit.csv`. Similar dimensions, a white background, similar costume, or same performer appearance can reflect a shared studio/setup; none alone confirms a common source recording.

The same-class pair table (`same_class_pairwise_evidence.csv`) contains every pair's split mapping, SHA-256 equality, frame/FPS metadata, aligned and cross-time pHash distances for a fixed performer-centered crop, the full frame, and background corners, plus pose XY RMSE/correlation. The fixed crop is not a person segmentation mask. `manual_visual_review.csv` records the five sparse frame reviews and their confidence. `selected_pairs_visual_spotcheck.png` provides those frames. `candidate_related_groups.csv` maps the provisional components and their split distribution after adding the manual possible edges. `clip_manifest_audit.csv` records all 430 source paths, source-container labels, metadata, feature paths, and file hashes.

## Evidence categories and limits

- **Confirmed:** byte-identical SHA-256 only. Zero same-class cross-split pairs met this rule; exact cross-class duplicate hash matches are also absent: {len(exact_hash_groups)} duplicate-hash groups.
- **High-confidence:** a strict multi-sample monotonic visual match combined with near-identical pose sequence. Zero pairs met the rule. This says the sampled signals did not confirm overlap, not that clips are independent.
- **Possible:** {len(possible)} automated cross-split same-class pair edges plus {len(manual_extra)} additional manually reviewed edges; the combined overlay yields {len(cross_groups)} connected components involving {len(group_clip_names)} clips. Repeated choreography, similar framing, and repeated poses can produce false positives; use the member maps as leads for review.
- **Unknown:** the remaining pair relationships and all source-container-as-session interpretations. Missing source video IDs, capture session/dancer IDs, and crop start/end times prevent reliable attribution.

The visual metric uses nine normalized time samples and perceptual hashes, with no full-frame sequence synchronization or actor segmentation. Pairwise pose analysis is restricted to the same gesture class because matching poses across different classes is not a meaningful source detector. Cross-class session relationships are not established by the pairwise table. Similarity alone is not proof.

## Conclusion and next audit step

This audit finds **zero confirmed byte-identical source files**, **105 automated possible pair edges plus three manually reviewed possible edges**, and **37 provisional components spanning splits involving 167 clips**. These possible counts are screening leads and may overstate true recording/session overlap. Separately, all 430 clips occur inside seven broad directory containers that cross splits; directory membership alone is unknown evidence. The current split therefore cannot be certified session-independent from the available files.

Next, adjudicate the high-priority candidate edges by source-video lineage and crop intervals, obtain session/dancer/camera labels, and create a separate grouped split manifest. Preserve this clip-level split as the historical comparator. Do not interpret lack of confirmed duplicates as proof of split independence.

## Supporting files and execution

- `audit_test1.py`: reproducible read-only inventory, frame sampling, hash and same-class pair-screening script; its summary-generation field error was corrected after this completed run.
- `finalize_test1.py`: regenerates the final report and summary from saved CSVs without rereading videos.
- `clip_manifest_audit.csv`, `same_class_pairwise_evidence.csv`, `candidate_related_groups.csv`, `source_bundle_split_counts.csv`, `class_split_dimension_counts.csv`, `filename_suffix_split_counts.csv`, `summary.json`, and `audit_errors.csv`.
- `audit_run.log` records a final summary-generation error: all tables were written, then the script hit a `KeyError` while formatting the dimension summary. `finalize_test1.py` was added to derive the corrected summary and this report from those tables without rereading videos.

No dataset, split manifest, checkpoint, experiment file, or results directory was modified. All outputs are isolated in this requested folder. New40 prediction files were not inspected.
'''
(HERE/"test1_split_independence_report.md").write_text(report,encoding="utf-8")
print(json.dumps({k:summary[k] for k in ("canonical_clips","same_class_pairs_tested","cross_split_same_class_pairs_tested","cross_split_pair_evidence_counts","cross_split_candidate_components","clips_in_cross_split_candidate_components","source_directory_bundles","bundles_with_multiple_splits","clips_in_bundles_with_multiple_splits")},indent=2))
