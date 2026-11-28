#!/usr/bin/env python3
"""Build Test 2A ledger and grouping documents from the read-only Test 1 artifacts."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
TEST1=HERE.parent/"Test1_split_independence"

def read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def write(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

clips=read(TEST1/"clip_manifest_audit.csv")
pairs=read(TEST1/"same_class_pairwise_evidence.csv")
manual=read(TEST1/"manual_visual_review.csv")
components=read(TEST1/"candidate_related_groups.csv")
bundle_rows=read(TEST1/"source_bundle_split_counts.csv")

def pkey(cls,a,b): return (cls,*sorted((a,b)))
machine={}
for p in pairs:
    if p["cross_split"]=="True" and p["evidence_category"]=="possible":
        machine[pkey(p["class_name"],p["video_a"],p["video_b"])]=p
manual_map={pkey(p["class_name"],p["video_a"],p["video_b"]):p for p in manual}
all_keys=sorted(set(machine)|set(manual_map))
strong_key=pkey("Kashtam","Kashtam_8.mp4","Kashtam_11.mp4")
component_for={}
for g in components:
    if g["spans_multiple_splits"]!="True": continue
    for name in g["members"].split(";"):
        component_for[(g["class_name"],name)]=g["candidate_group_id"]

edges=[]
for n,key in enumerate(all_keys,1):
    cls,a,b=key
    m=machine.get(key); v=manual_map.get(key)
    level="strong_candidate" if key==strong_key else "weak_candidate"
    confidence="medium-high" if key==strong_key else (v["confidence"] if v else "low-medium")
    split_a=next((r["split"] for r in clips if r["class_name"]==cls and r["video_name"]==a),"unknown")
    split_b=next((r["split"] for r in clips if r["class_name"]==cls and r["video_name"]==b),"unknown")
    comp=component_for.get((cls,a),"") if component_for.get((cls,a),"")==component_for.get((cls,b),"") else ""
    if m:
        technical=(f"Test1 same-class screen: center-crop pHash chain={m['pHash_body_monotonic_chain_le4']}/9; "
                   f"samples with pHash distance<=4={m['pHash_body_samples_with_match_le4']}; "
                   f"pose XY RMSE={m['pose_xy_rmse']}; correlation={m['pose_xy_correlation']}; "
                   f"background-corner pHash aligned mean={m['pHash_background_aligned_mean']}; "
                   f"same source directory={m['same_source_directory']}; same broad bundle={m['same_bundle']}; SHA match={m['same_sha256']}.")
    else:
        technical="No automated Test1 possible edge; added from sparse manual frame review."
    manual_text=v["evidence_summary"] if v else ""
    limitation=v["limitation"] if v else "Automated visual/pose similarity may reflect repeated choreography or a common setup; no source/session ID or interval match."
    edges.append({
        "relationship_id":f"REL{n:03d}","class_name":cls,
        "clip_id_a":f"{cls}/{Path(a).stem}","video_a":a,"split_a":split_a,
        "clip_id_b":f"{cls}/{Path(b).stem}","video_b":b,"split_b":split_b,
        "relationship_category":level,"confidence":confidence,
        "evidence_sources":"Test1 automated screen"+("; Test1 sparse visual review" if v else ""),
        "test1_screen_component_id":comp,
        "automated_evidence":technical,"manual_visual_evidence":manual_text,
        "limitation":limitation,"confirmed_same_source":False,
    })

by_clip=defaultdict(list)
for e in edges:
    by_clip[e["clip_id_a"]].append(e); by_clip[e["clip_id_b"]].append(e)
for r in clips:
    r["clip_id"]=f"{r['class_name']}/{Path(r['video_name']).stem}"
    r["original_source_video_id"]="UNKNOWN"
    r["recording_session_id"]="UNKNOWN"
    r["dancer_id"]="UNKNOWN"
    links=by_clip.get(r["clip_id"],[])
    statuses={x["relationship_category"] for x in links}
    r["provenance_status"]="strong_candidate" if "strong_candidate" in statuses else ("weak_candidate" if links else "unknown")
    r["confidence"]="medium-high candidate" if r["provenance_status"]=="strong_candidate" else ("low-medium candidate" if links else "unknown")
    r["candidate_relationship_ids"]=";".join(x["relationship_id"] for x in links)
    r["candidate_related_clip_ids"]=";".join(sorted({x["clip_id_b"] if x["clip_id_a"]==r["clip_id"] else x["clip_id_a"] for x in links}))
    r["test1_screen_component_ids"]=";".join(sorted({component_for[(r["class_name"],r["video_name"])] for _ in [0] if (r["class_name"],r["video_name"]) in component_for}))
    r["provenance_note"]="No verified source/session/dancer identifiers in inspected metadata; source_bundle is an organizational container only."

ledger_fields=["clip_id","video_name","class_name","label_index","split","source_path","source_directory","source_bundle",
               "metadata_width","metadata_height","decoded_width","decoded_height","decoded_fps","decoded_frames","decoded_duration",
               "sha256","feature_path","feature_exists","valid_frames","normalization_valid_frames","reference_shoulder_width",
               "original_source_video_id","recording_session_id","dancer_id","provenance_status","confidence",
               "candidate_relationship_ids","candidate_related_clip_ids","test1_screen_component_ids","provenance_note"]
write(HERE/"provenance_ledger.csv",clips,ledger_fields)
edge_fields=list(edges[0].keys()) if edges else ["relationship_id","relationship_category"]
write(HERE/"candidate_relationships.csv",edges,edge_fields)

# Independent groups are not observed. Class support is reported alongside
# necessary (not sufficient) group-count conditions for planned protocols.
class_counts=defaultdict(Counter)
for r in clips: class_counts[r["class_name"]][r["split"]]+=1
support=[]
for cls,c in sorted(class_counts.items()):
    support.append({"class_name":cls,"total_clips":sum(c.values()),"train_clips":c["train"],"validation_clips":c["val"],"test_clips":c["test"],
                    "verified_independent_groups_known":"NO","minimum_groups_for_3way_train_val_test":"3 necessary; not sufficient",
                    "minimum_groups_for_5_outer_test_folds":"5 necessary for each class to occur in every outer test fold; nested validation adds support constraints",
                    "feasibility":"UNKNOWN until session/source groups are verified"})
write(HERE/"class_support_feasibility.csv",support,list(support[0].keys()))

summary={
 "clips":len(clips),"classes":len(class_counts),"clip_manifest_sources_found":sum(r["file_exists"]=="True" for r in clips),
 "feature_files_found":sum(r["feature_exists"]=="True" for r in clips),"split_counts":dict(Counter(r["split"] for r in clips)),
 "all_classes_in_all_splits":sum(all(class_counts[c][s]>0 for s in ("train","val","test")) for c in class_counts),
 "clip_support_per_class_by_split":{s:dict(Counter(c[s] for c in class_counts.values())) for s in ("train","val","test")},
 "confirmed_relationships":0,"strong_candidate_relationships":sum(e["relationship_category"]=="strong_candidate" for e in edges),
 "weak_candidate_relationships":sum(e["relationship_category"]=="weak_candidate" for e in edges),
 "candidate_relationships_total":len(edges),"clips_with_candidate_links":sum(bool(by_clip.get(r["clip_id"])) for r in clips),
 "clips_with_unknown_session_provenance":len(clips),"verified_session_groups":0,
 "test1_transitive_candidate_components_spanning_splits":sum(g["spans_multiple_splits"]=="True" for g in components),
 "test1_component_members_are_not_session_assignments":True,
 "broad_source_containers_crossing_splits":sum(b["spans_multiple_splits"]=="True" for b in bundle_rows),
 "clips_in_broad_source_containers_crossing_splits":sum(int(b["total"]) for b in bundle_rows if b["spans_multiple_splits"]=="True"),
 "grouped_evaluation_ready":False,
 "reason":"Verified recording/session/source-video IDs and clip intervals are absent; group counts/support per class cannot be established.",
 "new40_per_video_predictions_inspected":False,
}
(HERE/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
