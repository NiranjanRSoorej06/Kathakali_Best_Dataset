# Test 2A grouping rules and evidence categories

## Purpose and unit of grouping

The intended leakage unit is the **recording session/source lineage**, not the gesture label, filename suffix, directory, resolution, or visual appearance. When a verified source recording is divided into clips, all derived clips from that recording should share a group, even when they have different class labels or non-overlapping crop intervals. If evaluation is intended to generalize to unseen dancers, dancer identity is a separate grouping dimension and should be reported as such.

Each ledger row is a clip. `clip_id` is the stable `class_name/video_stem` identifier. `original_source_video_id`, `recording_session_id`, and `dancer_id` are explicitly `UNKNOWN` for all 430 rows because the inspected metadata does not supply verified values. `source_bundle` records a directory-container clue only; it is not a session ID.

## Evidence categories

| Category | Rule | Test 2A status |
|---|---|---|
| **Confirmed link** | Same source-file SHA-256, or a curator/source record verifies the same original recording/session and, where relevant, overlapping source intervals. Use as a hard must-link for split construction. | No confirmed relationships were found. All 430 source hashes are available; there are no duplicate hashes. |
| **Strong candidate** | Multiple independent indicators support a common take/session (for example, same source directory plus matching frame count/FPS/duration and a similar sampled pose/visual sequence), but no source lineage or overlap interval confirms it. Preserve as a candidate edge; do not treat as ground truth. | One strongest candidate: Kashtam_8 vs Kashtam_11. SHA-256 differs. It remains unconfirmed. |
| **Weak candidate** | A Test 1 same-class pose/perceptual screen or sparse visual review suggests similarity, while ordinary repeated choreography/common setup could explain it. Keep as a review lead and sensitivity-only edge. | 107 weak candidate links. |
| **Unknown provenance** | No verified session/source/dancer assignment or sufficient direct relation evidence. “Unknown” is not equivalent to independent. | Session/source/dancer provenance remains unknown for all 430 clips; remaining pair relations are unknown. |

Confidence labels qualify the candidate evidence, not the probability that two files are the same recording. The Test 1 pHash and pose screens are heuristic. A high score, common wall, costume, camera, performer appearance, or trajectory alone cannot create a confirmed link.

## Defensible grouping rules

1. **Hard must-link:** identical source bytes; verified identical original-video ID; verified overlapping source-frame/time intervals; or a curator-verified session/source ID, according to the stated evaluation target.
2. **Cross-class grouping:** if a recording session includes multiple gestures, group all clips from that session across classes. Do not build separate same-class-only groups when the goal is session independence.
3. **Ambiguous links:** keep strong and weak candidates in the relationship table with evidence, confidence, and review status. Do not silently union them into the primary grouping. For sensitivity analysis only, compare strict confirmed-only links, confirmed-plus-strong candidates, and a deliberately conservative candidate-union scenario.
4. **Never infer a hard group solely from:** filename prefix/suffix/number, class label, directory membership, dimensions/resolution, FPS, similar backgrounds, camera framing, costume, face/body appearance, or pose/choreography similarity.
5. **Directory/setup tags:** `Videos_Cropped1`–`Videos_Cropped6` and `Video_Collection_first_person` are recorded as organizational containers. They cross splits, but are not verified sessions. Use them only as explicitly labeled exploratory stress tests.
6. **Resolution/setup tags:** dimensions are setup covariates, not source IDs. Any leave-one-dimension-out analysis is a domain/setup evaluation, not a session-group split.
7. **No arbitrary split:** this audit does not assign final source/session IDs, generate a new split, or select model-training folds from candidate-only links.

## Evidence provenance

The 108 direct candidate edges in `candidate_relationships.csv` combine the 105 Test 1 automated possible edges with three additional manually reviewed possible edges. Of those, one is classed as a strong candidate and 107 as weak candidates. The five manual visual reviews are copied here as `test1_manual_visual_review.csv` and `test1_selected_pairs_visual_spotcheck.png`; see the report for why the Kashtam pair is the strongest candidate. The Test 1 connected-component map is copied as `test1_candidate_related_groups.csv`. These are screening components only; their transitive membership is not a source/session assignment.
