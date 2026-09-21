from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_HISTORY_OFFSETS = (-60, -53, -47, -40, -33, -27, -20, -13, -7, 0)
STATE_MODEL_INDICES = tuple(range(10))
STATE_WITH_VELOCITY_INDICES = tuple(range(10)) + tuple(range(24, 31))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_array(table: pa.Table, name: str, dtype) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def nearest_indices(timestamps_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(timestamps_ns, query_ns, side="left")
    positions = np.clip(positions, 0, len(timestamps_ns) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = np.abs(timestamps_ns[previous] - query_ns) <= np.abs(timestamps_ns[positions] - query_ns)
    indices = np.where(use_previous, previous, positions)
    distances = np.abs(timestamps_ns[indices] - query_ns)
    return indices.astype(np.int64), distances.astype(np.int64)


def episode_task_text(tasks_table: pa.Table, episode_row: dict) -> str:
    task_indices = episode_row.get("tasks", [])
    if isinstance(task_indices, np.ndarray):
        task_indices = task_indices.tolist()
    if not isinstance(task_indices, list):
        task_indices = [task_indices]
    if task_indices and isinstance(task_indices[0], str):
        return task_indices[0]
    if task_indices and "task" in tasks_table.column_names:
        task_index = int(task_indices[0])
        task_rows = tasks_table.to_pylist()
        if 0 <= task_index < len(task_rows):
            return str(task_rows[task_index]["task"])
    return ""


def data_path(dataset_root: Path, episode_row: dict) -> Path:
    chunk = int(episode_row["data/chunk_index"])
    file_index = int(episode_row["data/file_index"])
    path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def video_path(dataset_root: Path, camera: str, episode_row: dict) -> str:
    chunk = int(episode_row[f"videos/observation.images.{camera}/chunk_index"])
    file_index = int(episode_row[f"videos/observation.images.{camera}/file_index"])
    return str(Path("videos") / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")


def classify_episode(
    table: pa.Table,
    history_offsets: np.ndarray,
    future_steps: int,
    fps: float,
    max_gap_s: float,
    timestamp_tolerance_s: float,
) -> tuple[dict, str | None]:
    state = as_array(table, "observation.state", np.float32)
    action = as_array(table, "action", np.float32)
    observed_ns = as_array(table, "complementary_info.observed_at_ns", np.int64)
    image_names = ("complementary_info.base_image_at_ns", "complementary_info.left_wrist_image_at_ns")
    if state.ndim != 2 or state.shape[1] != 31:
        return {}, f"state_shape={state.shape}"
    if action.ndim != 2 or action.shape[1] != 10:
        return {}, f"action_shape={action.shape}"
    if len(state) != len(action) or len(state) != len(observed_ns):
        return {}, "row_count_mismatch"
    if len(state) <= future_steps or len(state) <= abs(int(history_offsets[0])):
        return {}, "episode_too_short"
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        return {}, "nonfinite_state_or_action"
    if not np.all(np.diff(observed_ns) > 0):
        return {}, "observed_timestamps_not_strictly_increasing"
    if float(np.max(np.diff(observed_ns))) / 1e9 > max_gap_s:
        return {}, "observed_timestamp_gap"
    episode_index_values = as_array(table, "episode_index", np.int64)
    dataset_index_values = as_array(table, "index", np.int64)
    phase_values = as_array(table, "complementary_info.phase", np.float32)
    base_image_ns = as_array(table, "complementary_info.base_image_at_ns", np.int64)
    wrist_image_ns = as_array(table, "complementary_info.left_wrist_image_at_ns", np.int64)
    for name in image_names:
        if not np.all(as_array(table, name, np.int64) > 0):
            return {}, f"invalid_{name}"

    tolerance_ns = int(timestamp_tolerance_s * 1e9)
    history_query_offsets_ns = np.rint(history_offsets * 1e9 / fps).astype(np.int64)
    future_query_offsets_ns = np.arange(1, future_steps + 1, dtype=np.int64) * int(round(1e9 / fps))
    rows = []
    for anchor in range(len(state)):
        query_history = observed_ns[anchor] + history_query_offsets_ns
        history_indices, history_distances = nearest_indices(observed_ns, query_history)
        query_future = observed_ns[anchor] + future_query_offsets_ns
        future_indices, future_distances = nearest_indices(observed_ns, query_future)
        if history_indices[0] < 0 or history_indices[-1] > anchor:
            continue
        if future_indices[0] <= anchor or future_indices[-1] >= len(state):
            continue
        if np.any(history_distances > tolerance_ns) or np.any(future_distances > tolerance_ns):
            continue
        if np.any(np.diff(history_indices) <= 0) or np.any(np.diff(future_indices) <= 0):
            continue
        if np.any(np.diff(observed_ns[history_indices]) <= 0) or np.any(np.diff(observed_ns[future_indices]) <= 0):
            continue
        if not np.isfinite(state[history_indices]).all() or not np.isfinite(state[future_indices]).all():
            continue
        action_indices = future_indices - 1
        if action_indices[0] != anchor or np.any(np.diff(action_indices) != 1):
            continue
        rows.append({
            "episode_index": int(episode_index_values[0]),
            "episode_success": True,
            "source_failure_reason": "",
            "anchor_frame": int(anchor),
            "anchor_phase": float(phase_values[anchor]),
            "anchor_dataset_index": int(dataset_index_values[anchor]),
            "task_text": "",
            "base_video_path": "",
            "left_wrist_video_path": "",
            "base_video_frame": int(anchor),
            "left_wrist_video_frame": int(anchor),
            "anchor_observed_at_ns": int(observed_ns[anchor]),
            "base_image_at_ns": int(base_image_ns[anchor]),
            "left_wrist_image_at_ns": int(wrist_image_ns[anchor]),
            "history_frame_indices": history_indices.tolist(),
            "future_frame_indices": future_indices.tolist(),
            "state": state[anchor, list(STATE_MODEL_INDICES)].tolist(),
            "state_with_joint_velocity": state[anchor, list(STATE_WITH_VELOCITY_INDICES)].tolist(),
            "observation_state_31d": state[anchor].tolist(),
            "joint_torque_history": state[history_indices, 10:17].reshape(-1).tolist(),
            "external_joint_torque_history": state[history_indices, 17:24].reshape(-1).tolist(),
            "future_actions": action[action_indices].reshape(-1).tolist(),
            "future_joint_torque": state[future_indices, 10:17].reshape(-1).tolist(),
            "future_external_joint_torque": state[future_indices, 17:24].reshape(-1).tolist(),
        })
    if not rows:
        return {}, "no_valid_windows"
    return {"rows": rows, "frame_count": len(state), "valid_window_count": len(rows)}, None


def build_windows(
    dataset_root: Path,
    classification_path: Path,
    output_dir: Path,
    history_offsets: tuple[int, ...] = DEFAULT_HISTORY_OFFSETS,
    future_steps: int = 50,
    fps: float = 30.0,
    max_gap_s: float = 0.1,
    timestamp_tolerance_s: float | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_json(dataset_root / "meta/info.json")
    episodes_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    tasks_table = pq.read_table(dataset_root / "meta/tasks.parquet")
    episodes = {int(row["episode_index"]): row for row in episodes_table.to_pylist()}
    selection = read_json(classification_path)
    selected_indices = [int(index) for index in selection["episode_indices"]]
    tolerance = timestamp_tolerance_s if timestamp_tolerance_s is not None else (1.0 / fps - 1e-4)
    history_array = np.asarray(history_offsets, dtype=np.int64)
    all_rows = []
    episode_reports = []
    for episode_index in selected_indices:
        row = episodes.get(episode_index)
        if row is None:
            episode_reports.append({"episode_index": episode_index, "status": "excluded", "reason": "missing_episode_metadata"})
            continue
        table = pq.read_table(data_path(dataset_root, row))
        result, reason = classify_episode(table, history_array, future_steps, fps, max_gap_s, tolerance)
        if reason is not None:
            episode_reports.append({"episode_index": episode_index, "status": "excluded", "reason": reason})
            continue
        task_text = episode_task_text(tasks_table, row)
        for window in result["rows"]:
            window["task_text"] = task_text
            window["episode_success"] = bool(row["episode_success"])
            window["source_failure_reason"] = str(row.get("failure_reason") or "")
            window["base_video_path"] = video_path(dataset_root, "base", row)
            window["left_wrist_video_path"] = video_path(dataset_root, "left_wrist", row)
        all_rows.extend(result["rows"])
        episode_reports.append({"episode_index": episode_index, "status": "included", "frame_count": result["frame_count"], "valid_window_count": result["valid_window_count"]})

    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        pq.write_table(table, output_dir / "windows.parquet", compression="zstd")
    metadata = {
        "dataset_root": str(dataset_root),
        "classification_path": str(classification_path),
        "source_info": {"total_episodes": info["total_episodes"], "total_frames": info["total_frames"], "fps": info["fps"]},
        "selected_episode_count": len(selected_indices),
        "included_episode_count": sum(report["status"] == "included" for report in episode_reports),
        "excluded_episode_count": sum(report["status"] == "excluded" for report in episode_reports),
        "window_count": len(all_rows),
        "history_offsets_frames": list(map(int, history_offsets)),
        "history_timestamps_seconds": [round(offset / fps, 6) for offset in history_offsets],
        "future_action_offsets_frames": list(range(future_steps)),
        "future_torque_offsets_frames": list(range(1, future_steps + 1)),
        "state_features": "TCP pose/gripper from the 31D observation: 10D; joint velocity is retained in state_with_joint_velocity",
        "state_shape": [10],
        "state_with_joint_velocity_shape": [17],
        "joint_torque_history_shape": [len(history_offsets), 7],
        "external_joint_torque_history_shape": [len(history_offsets), 7],
        "future_action_shape": [future_steps, 10],
        "future_joint_torque_shape": [future_steps, 7],
        "future_external_joint_torque_shape": [future_steps, 7],
        "normalization": "not applied; fit statistics on training episodes only",
        "validity_checks": [
            "selected from the success manifest",
            "31D state and 10D action shapes",
            "finite state/action values",
            "strictly increasing observation timestamps",
            f"observation timestamp gaps <= {max_gap_s} seconds",
            f"history/future timestamp lookup error <= {tolerance} seconds",
            "causal history and action/torque alignment within one episode",
        ],
        "visual_representation": "base_video_path and left_wrist_video_path point to the source MP4; frame fields identify the anchor image",
        "effort_policy": "measured joint torque is the first model effort; external joint torque is retained for ablation",
        "raw_data_modified": False,
        "episode_reports": episode_reports,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/franka_single_left_31d"))
    parser.add_argument("--classification", type=Path, default=Path("annotations/31d_classification/sft_candidate.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/force_vla_31d_windows"))
    parser.add_argument("--future-steps", type=int, default=50)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    metadata = build_windows(args.dataset_root, args.classification, args.output_dir, future_steps=args.future_steps, fps=args.fps)
    print(json.dumps({key: value for key, value in metadata.items() if key != "episode_reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
