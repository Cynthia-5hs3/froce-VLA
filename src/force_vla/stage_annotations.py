"""Offline episode review, validated sidecar labels, and conservative window indices."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = "outputs/force_vla_prepared_v3/data_contract.json"
STAGES = {
    "approach": "接近/搬运", "misaligned_contact": "错误接触/卡住",
    "correction": "有效纠偏", "aligned_press": "对准下压",
    "seated_hold": "就位/压紧保持", "release_retreat": "释放/回升",
    "idle": "无操作等待", "uncertain": "不确定",
}
EVENTS = {"contact": "首次/再次接触", "correction_start": "开始纠偏",
          "aligned_press_start": "开始对准下压", "seated": "确认就位",
          "release": "释放", "other": "其他关键帧"}
DECISIONS = {"keep": "模仿", "exclude": "不模仿", "downsample": "降采样候选", "review": "待复核"}
SCHEMA = "force_vla_stage_annotations_v1"


def local_path(value, parent=None):
    path = (ROOT / value).resolve()
    scope = ROOT if parent is None else ROOT / parent
    if not path.is_relative_to(scope) or path == scope:
        raise ValueError(f"Path must be below {scope}: {path}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_context(contract_path):
    import pyarrow.parquet as parquet

    contract_path = local_path(contract_path, "outputs")
    contract = read_json(contract_path)
    windows = local_path(contract["windows"], "outputs")
    if digest(windows) != contract["sha256"]:
        raise ValueError("Windows changed since preparation")
    metadata = read_json(windows.with_name("metadata.json"))
    source = local_path(metadata["dataset_root"], "data")
    episodes = parquet.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return {"contract": contract, "contract_path": contract_path, "windows": windows,
            "source": source, "episodes": {int(row["episode_index"]): row for row in episodes}}


def episode_source(context, episode):
    row = context["episodes"][episode]
    source = context["source"]
    path = source / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
    videos = {}
    for camera in ("base", "left_wrist"):
        prefix = f"videos/observation.images.{camera}"
        if row.get(f"{prefix}/from_timestamp", 0) != 0:
            raise ValueError("Shared video offsets are not supported by this review format")
        videos[camera] = source / "videos" / f"observation.images.{camera}" / f"chunk-{row[prefix + '/chunk_index']:03d}" / f"file-{row[prefix + '/file_index']:03d}.mp4"
    if episode in context["contract"]["train_episodes"]:
        split = "train"
    elif episode in context["contract"]["validation_episodes"]:
        split = "validation"
    else:
        raise ValueError("Episode is outside the prepared split")
    identity = {"dataset": str(source.relative_to(ROOT)), "episode_index": episode,
                "frame_count": int(row["length"]), "split": split,
                "contract": str(context["contract_path"].relative_to(ROOT)),
                "contract_sha256": digest(context["contract_path"]),
                "data_path": str(path.relative_to(ROOT)), "data_sha256": digest(path),
                "videos": {camera: {"path": str(video.relative_to(ROOT)), "sha256": digest(video)}
                           for camera, video in videos.items()}}
    return identity, path, videos


def validate_document(document, expected_source):
    if document.get("schema") != SCHEMA or document.get("source") != expected_source:
        raise ValueError("Annotation schema or source identity differs")
    if not isinstance(document.get("annotator"), str) or not document["annotator"].strip():
        raise ValueError("Annotator name is required")
    if type(document.get("complete")) is not bool:
        raise ValueError("complete must be a boolean")
    count = expected_source["frame_count"]
    segments = document.get("segments")
    events = document.get("keyframes")
    if not isinstance(segments, list) or not isinstance(events, list):
        raise ValueError("segments and keyframes must be lists")
    previous_end = 0
    covered = 0
    for segment in sorted(segments, key=lambda item: item.get("start_frame", -1)):
        start, end = segment.get("start_frame"), segment.get("end_frame")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= count:
            raise ValueError("Invalid interval: expected 0 <= start < end <= frame_count")
        if start < previous_end:
            raise ValueError("Overlapping intervals are not allowed")
        if segment.get("stage") not in STAGES or segment.get("imitation") not in DECISIONS:
            raise ValueError("Unknown stage or imitation decision")
        confidence = segment.get("confidence")
        if confidence not in ("confirmed", "uncertain"):
            raise ValueError("Unknown confidence")
        if (confidence == "uncertain" or segment["stage"] == "uncertain") and segment["imitation"] != "review":
            raise ValueError("Uncertain segments must remain review")
        if not isinstance(segment.get("note"), str):
            raise ValueError("Segment note must be text")
        covered += end - start
        previous_end = end
    for event in events:
        frame = event.get("frame")
        if type(frame) is not int or not 0 <= frame < count or event.get("event") not in EVENTS:
            raise ValueError("Invalid keyframe")
        if not isinstance(event.get("note"), str):
            raise ValueError("Keyframe note must be text")
    if document["complete"] and covered != count:
        raise ValueError("Complete annotation must cover every frame without gaps")
    return {"frames": count, "covered_frames": covered, "unmarked_frames": count - covered,
            "segments": len(segments), "keyframes": len(events),
            "review_frames": sum(item["end_frame"] - item["start_frame"] for item in segments if item["imitation"] == "review")}


def frame_labels(document):
    count = document["source"]["frame_count"]
    stages, decisions = ["unmarked"] * count, ["review"] * count
    for segment in document["segments"]:
        for frame in range(segment["start_frame"], segment["end_frame"]):
            stages[frame] = segment["stage"]
            decisions[frame] = segment["imitation"]
    return stages, decisions


def window_review(anchor, future_frames, stages, decisions):
    action_frames = [int(frame) - 1 for frame in future_frames]
    if len(action_frames) != 50 or action_frames != list(range(anchor, anchor + 50)):
        raise ValueError("Unexpected future-action alignment")
    mask = [decisions[frame] in ("keep", "downsample") for frame in action_frames]
    approved = all(mask)
    weight = .1 if decisions[anchor] == "downsample" else 1.
    return {"anchor_stage": stages[anchor], "anchor_imitation": decisions[anchor],
            "action_target_mask": mask, "all_future_actions_approved": approved,
            "suggested_sampling_weight": weight if approved else 0.}


def prepare_review(args):
    raise SystemExit(
        "Stage annotation web UI has been retired. "
        "Existing label import, validation and training indices remain available. "
        "Restore instructions: archives/stage_cleanup_20261021/README.md"
    )


def canonical_name(source):
    source_key = hashlib.sha256(source["dataset"].encode()).hexdigest()[:10]
    return f"{source_key}_episode-{source['episode_index']:06d}.json"


def import_labels(args):
    context = load_context(args.contract)
    destination = local_path(args.annotations, "annotations")
    pending, seen = [], set()
    for value in args.files:
        path = local_path(value)
        document = read_json(path)
        identity, _, _ = episode_source(context, document["source"]["episode_index"])
        summary = validate_document(document, identity)
        target = destination / canonical_name(identity)
        if target.name in seen:
            raise ValueError("Multiple imports target the same episode")
        seen.add(target.name)
        if target.exists() and read_json(target) != document and not args.replace:
            raise FileExistsError(f"Existing label differs: {target}; use --replace to archive it")
        pending.append((target, document, summary))
    destination.mkdir(parents=True, exist_ok=True)
    for target, document, summary in pending:
        if target.exists() and read_json(target) != document:
            archive = destination / "history"
            archive.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shutil.copyfile(target, archive / f"{target.stem}_{stamp}_{uuid.uuid4().hex[:8]}.json")
        if not target.exists() or read_json(target) != document:
            write_json(target, document)
        print(json.dumps({"imported": str(target), **summary}, ensure_ascii=False))


def build_index(args):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    context = load_context(args.contract)
    annotation_dir = local_path(args.annotations, "annotations")
    labels, audits = {}, []
    for path in sorted(annotation_dir.glob("*.json")):
        document = read_json(path)
        identity, _, _ = episode_source(context, document["source"]["episode_index"])
        summary = validate_document(document, identity)
        episode = identity["episode_index"]
        if episode in labels:
            raise ValueError("Duplicate episode annotations")
        labels[episode] = frame_labels(document)
        audits.append({"episode": episode, "split": identity["split"], "path": str(path),
                       "sha256": digest(path), **summary})
    if not labels:
        raise ValueError("No imported annotations; import browser exports first")
    table = parquet.read_table(context["windows"], columns=["episode_index", "anchor_frame", "future_frame_indices"])
    rows, eligible, phases = [], [], Counter()
    for window_row, entry in enumerate(table.to_pylist()):
        episode, anchor = entry["episode_index"], entry["anchor_frame"]
        if episode not in context["contract"]["train_episodes"]:
            continue
        if episode in labels:
            stages, decisions = labels[episode]
        else:
            count = context["episodes"][episode]["length"]
            stages, decisions = ["unmarked"] * count, ["review"] * count
        reviewed = window_review(anchor, entry["future_frame_indices"], stages, decisions)
        rows.append({"source_window_row": window_row, "episode_index": episode, "anchor_frame": anchor, **reviewed})
        if reviewed["all_future_actions_approved"]:
            eligible.append(window_row)
            phases[reviewed["anchor_stage"]] += 1
    destination = local_path(args.output, "outputs")
    destination.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), destination / "window_review.parquet", compression="zstd")
    write_json(destination / "eligible_train_rows.json", {"windows": str(context["windows"]),
        "windows_sha256": context["contract"]["sha256"], "rows": eligible,
        "selection": "train split only; all 50 action target frames explicitly approved"})
    write_json(destination / "summary.json", {"annotations": audits, "training_windows": len(rows),
        "eligible_windows": len(eligible), "eligible_by_stage": dict(phases),
        "validation_episodes_unchanged": context["contract"]["validation_episodes"],
        "training_ready": False,
        "next_step": "Review coverage and stage balance, then prepare an explicit pi05.stage_sampling experiment. Ordinary training runs do not automatically consume this candidate index or its mask/weights.",
        "sampling_weights": "Suggestions only: approved downsample anchors=0.1, approved other anchors=1; unapproved windows=0"})
    print(json.dumps({"output": str(destination), "eligible_windows": len(eligible), "training_ready": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--contract", default=DEFAULT_CONTRACT)
    prepare.add_argument("--split", choices=("train", "validation"), default="train")
    prepare.add_argument("--episodes", type=int, nargs="+")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(run=prepare_review)
    importer = commands.add_parser("import")
    importer.add_argument("--contract", default=DEFAULT_CONTRACT)
    importer.add_argument("--files", nargs="+", required=True)
    importer.add_argument("--annotations", default="annotations/stages_v3")
    importer.add_argument("--replace", action="store_true")
    importer.set_defaults(run=import_labels)
    indexer = commands.add_parser("build-index")
    indexer.add_argument("--contract", default=DEFAULT_CONTRACT)
    indexer.add_argument("--annotations", default="annotations/stages_v3")
    indexer.add_argument("--output", required=True)
    indexer.set_defaults(run=build_index)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
