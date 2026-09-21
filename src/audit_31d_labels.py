import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/franka_single_left_31d")
episodes = pd.read_parquet(ROOT / "meta/episodes/chunk-000/file-000.parquet")
print("columns:", list(episodes.columns))
print("rows:", len(episodes))
for column in episodes.columns:
    if any(term in column.lower() for term in ("success", "failure", "label", "reason", "critical", "task", "episode")):
        values = episodes[column].map(lambda value: json.dumps(value.tolist() if hasattr(value, "tolist") else value, ensure_ascii=False))
        print("VALUE", column, values.value_counts(dropna=False).to_dict())
print(episodes.to_string(max_rows=20))

attempts = []
for path in (ROOT / ".recording_attempts").glob("*/attempt.json"):
    record = json.loads(path.read_text())
    record["attempt_dir"] = str(path.parent)
    for name in ("error.json", "metadata.json", "episode.json"):
        extra = path.parent / name
        if extra.exists():
            try:
                record[name.removesuffix(".json")] = json.loads(extra.read_text())
            except json.JSONDecodeError:
                record[name.removesuffix(".json")] = "<invalid json>"
    attempts.append(record)
print("attempts:", len(attempts))
print("attempt status:", pd.Series([x.get("status") for x in attempts]).value_counts(dropna=False).to_dict())
print("attempts with errors:")
for record in sorted(attempts, key=lambda x: x.get("episode_index", -1)):
    if "error" in record or record.get("status") != "committed":
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))

print("missing label values:")
missing = []
for _, row in episodes.iterrows():
    values = row.to_dict()
    candidates = {k: values.get(k) for k in values if any(t in k.lower() for t in ("success", "failure", "label", "reason"))}
    if not candidates or all(pd.isna(v) or v == "" for v in candidates.values()):
        episode_index = int(row.get("episode_index", -1))
        missing.append(episode_index)
        print(episode_index, candidates)

def reason_group(reason):
    if not reason:
        return "missing"
    if reason == "operator marked failure":
        return "operator_failure"
    if reason == "discard requested":
        return "discard_requested_text"
    if reason.startswith("Franka command success rate is low"):
        return "controller_low_success_rate"
    if reason.startswith("Franky poll fault"):
        return "controller_poll_fault"
    if reason.startswith("camera/recording"):
        return "recording_timing_fault"
    return "other"

report = {
    "dataset": str(ROOT),
    "raw_data_modified": False,
    "episode_count": int(len(episodes)),
    "missing_label_episode_indices": missing,
    "missing_label_count": len(missing),
    "success_count": int(episodes["episode_success"].eq(True).sum()),
    "failure_count": int(episodes["episode_success"].eq(False).sum()),
    "failure_reason_groups": episodes["failure_reason"].map(reason_group).value_counts(dropna=False).to_dict(),
    "success_with_failure_reason_text": episodes.loc[
        episodes["episode_success"].eq(True) & episodes["failure_reason"].notna(),
        "episode_index",
    ].astype(int).tolist(),
    "attempt_status_counts": pd.Series([x.get("status") for x in attempts]).value_counts(dropna=False).to_dict(),
    "stale_or_incomplete_attempts": [
        {"episode_index": x.get("episode_index"), "status": x.get("status"), "reason": x.get("reason"), "error": x.get("error")}
        for x in attempts if "error" in x or x.get("status") != "committed"
    ],
}
(Path("annotations") / "label_audit_31d.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
print("REPORT", json.dumps(report, ensure_ascii=False, indent=2, default=str))
