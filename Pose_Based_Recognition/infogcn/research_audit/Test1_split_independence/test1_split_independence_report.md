# Test 1: Clip-Level Split Independence Audit

## Scope and result

Read-only audit of all 430 canonical videos and their processed pose files. The manifest has 300 train, 65 validation, and 65 test clips across 41 classes; 41 classes occur in every split. All 430 source files were found, hashed, and sampled at nine normalized timestamps for coarse visual/perceptual screening. No training, tuning, data/split/checkpoint edits, or New40 per-video prediction inspection was performed.

## Quantified evidence

| Evidence view | Count | Interpretation |
|---|---:|---|
| Cross-split same-class pair comparisons | 1045 | All possible pairs among same-class clips assigned to different splits; this is the comparison denominator. |
| Confirmed pairs | 0 | Confirmed only by identical source-file SHA-256. This does not detect re-encoded duplicates. |
| High-confidence pairs | 0 | None met the strict combined frame/pose screen. This is not evidence of independence. |
| Automated possible pair candidates | 105 | Perceptual/pose screen; requires source-level or human temporal review. |
| Additional manually reviewed possible edges | 3 | Sparse visual review of five named pairs upgraded these three pairs from unknown to possible; see `manual_visual_review.csv`. |
| Combined possible pair leads | 108 | Automated candidates plus manual additions; not confirmed overlap. |
| Possible candidate components spanning splits | 37 | Connected components after the manual overlay; transitive similarity can over-group clips. |
| Clips in those components | 167 | Candidate exposure count, not a confirmed related-clip count. |
| Broad source-directory bundles spanning splits | 7 of 7 | All 430 clips sit in directory containers that contain multiple splits. Directory names are not verified session IDs. |

Of the 1045 cross-split same-class pairs, 940 were classified unknown by the automated screen (no pair-specific signal beyond ordinary same-class membership), and 105 were possible. A separate sparse visual review inspected five named pairs, adding 3 possible edges. The automated possible screen was triggered by a center-crop perceptual-hash monotonic match chain of at least four of nine samples, or very high pose-coordinate correlation/low error, or a same-source-directory pair with nearly identical frame count and FPS. These are deliberately broad candidate rules, not overlap determinations.

## Split, naming, directory, and setup patterns

The canonical split is clip-level: every class has only one or two validation clips and one or two test clips. Filename suffixes range across numbered recordings; suffix numbers are spread across splits and are not known to represent chronology or source IDs. See `filename_suffix_split_counts.csv` for suffix-by-split counts.

All seven broad source-directory containers cross splits: Videos_Cropped1–6 and Video_Collection_first_person. Their per-split counts are in `source_bundle_split_counts.csv`. Treating each as a single recording group would put all 430 clips in potentially crossing containers, but that interpretation is **unknown**, not confirmed: these directories may be collection/storage batches rather than sessions.

The metadata exposes four image-dimension/setup proxies: 1030x1080: 44, 1900x1710: 84, 1900x2060: 82, 2300x2160: 220. There are 21 multi-dimension classes and 20 single-dimension classes. FPS, frame count, duration, source directory, shoulder-width normalization, and valid-frame counts are recorded per clip in `clip_manifest_audit.csv`. Similar dimensions, a white background, similar costume, or same performer appearance can reflect a shared studio/setup; none alone confirms a common source recording.

The same-class pair table (`same_class_pairwise_evidence.csv`) contains every pair's split mapping, SHA-256 equality, frame/FPS metadata, aligned and cross-time pHash distances for a fixed performer-centered crop, the full frame, and background corners, plus pose XY RMSE/correlation. The fixed crop is not a person segmentation mask. `manual_visual_review.csv` records the five sparse frame reviews and their confidence. `selected_pairs_visual_spotcheck.png` provides those frames. `candidate_related_groups.csv` maps the provisional components and their split distribution after adding the manual possible edges. `clip_manifest_audit.csv` records all 430 source paths, source-container labels, metadata, feature paths, and file hashes.

## Evidence categories and limits

- **Confirmed:** byte-identical SHA-256 only. Zero same-class cross-split pairs met this rule; exact cross-class duplicate hash matches are also absent: 0 duplicate-hash groups.
- **High-confidence:** a strict multi-sample monotonic visual match combined with near-identical pose sequence. Zero pairs met the rule. This says the sampled signals did not confirm overlap, not that clips are independent.
- **Possible:** 105 automated cross-split same-class pair edges plus 3 additional manually reviewed edges; the combined overlay yields 37 connected components involving 167 clips. Repeated choreography, similar framing, and repeated poses can produce false positives; use the member maps as leads for review.
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
