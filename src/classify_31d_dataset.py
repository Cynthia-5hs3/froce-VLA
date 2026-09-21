import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/franka_single_left_31d")
OUT = Path("annotations/31d_classification")
OUT.mkdir(parents=True, exist_ok=True)
episodes = pd.concat(
    [pd.read_parquet(path) for path in sorted((ROOT / "meta/episodes").glob("**/*.parquet"))],
    ignore_index=True,
).sort_values("episode_index")

def group(row):
    success = bool(row["episode_success"])
    reason = str(row.get("failure_reason") or "")
    length = int(row["length"])
    if success:
        return "sft_candidate" if length >= 111 else "sft_review_short"
    if reason == "operator marked failure":
        return "failure_candidate" if length >= 111 else "failure_review_short"
    if reason.startswith("Franka command success rate") or reason.startswith("Franky poll fault"):
        return "excluded_controller_fault"
    if reason.startswith("camera/recording"):
        return "excluded_recording_fault"
    return "excluded_other"

def batch(row):
    index = int(row["episode_index"])
    if index >= 152:
        return "followup_collection_2026-09-16"
    return "initial_collection"

groups = {name: [] for name in (
    "sft_candidate", "sft_review_short", "failure_candidate", "failure_review_short",
    "excluded_controller_fault", "excluded_recording_fault", "excluded_other",
)}
for _, row in episodes.iterrows():
    group_name = group(row)
    groups[group_name].append(int(row["episode_index"]))

for name, indices in groups.items():
    (OUT / f"{name}.json").write_text(json.dumps({
        "dataset": str(ROOT), "raw_data_modified": False, "episode_indices": indices,
        "count": len(indices), "selection_rule": "derived from episode_success, failure_reason, and length; review before training",
    }, ensure_ascii=False, indent=2) + "\n")

summary = {
    "dataset": str(ROOT),
    "raw_data_modified": False,
    "total_committed_episodes": len(episodes),
    "minimum_window_frames": 111,
    "groups": {name: {"count": len(indices), "episode_indices": indices} for name, indices in groups.items()},
    "success_total": int(episodes["episode_success"].eq(True).sum()),
    "success_at_least_111_frames": len(groups["sft_candidate"]),
    "operator_failure_at_least_111_frames": len(groups["failure_candidate"]),
    "collection_batches": {
        name: {
            "episode_indices": [int(row["episode_index"]) for _, row in episodes.iterrows() if batch(row) == name],
            "success_count": int(sum(bool(row["episode_success"]) and group(row).startswith("sft_") for _, row in episodes.iterrows() if batch(row) == name)),
            "excluded_count": int(sum(group(row).startswith("excluded_") for _, row in episodes.iterrows() if batch(row) == name)),
        }
        for name in sorted({batch(row) for _, row in episodes.iterrows()})
    },
    "explicit_manual_exclusion": {
        "episode_index": 170,
        "reason": "Franky poll fault; manual recovery required; libfranka cartesian_reflex",
        "classification": "excluded_controller_fault",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
