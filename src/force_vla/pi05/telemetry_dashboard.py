"""Publish a local browser dashboard from JSONL telemetry, without robot access."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import shutil
import time

from .torque_monitor import finite_vector


ROOT = Path(__file__).resolve().parents[3]


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def pose_channels(action):
    if action is None:
        return None, None
    xyz = finite_vector(action[:3], 3)
    position = [1000 * value for value in xyz] if xyz is not None else None
    rotation = finite_vector(action[3:9], 6)
    if rotation is None:
        return position, None
    first, second = rotation[:3], rotation[3:]
    length = math.hypot(*first)
    if length < 1e-8:
        return position, None
    first = [value / length for value in first]
    projection = sum(left * right for left, right in zip(first, second))
    second = [value - projection * axis for value, axis in zip(second, first)]
    length = math.hypot(*second)
    if length < 1e-8:
        return position, None
    second = [value / length for value in second]
    third = [first[1] * second[2] - first[2] * second[1],
             first[2] * second[0] - first[0] * second[2],
             first[0] * second[1] - first[1] * second[0]]
    horizontal = math.hypot(first[0], first[1])
    pitch = math.atan2(-first[2], horizontal)
    if horizontal > 1e-8:
        roll = math.atan2(second[2], third[2])
        yaw = math.atan2(first[1], first[0])
    else:
        roll = math.atan2(-third[1], second[1])
        yaw = 0.
    return position, [math.degrees(value) for value in (roll, pitch, yaw)]


def project_record(record):
    timestamp = finite_number(record.get("control_at_ns"))
    measured = finite_vector(record.get("measured_tau_J"), 7)
    if timestamp is None or measured is None:
        return None
    raw = finite_vector(record.get("model_action"), 10)
    if raw is None:
        xyz = finite_vector(record.get("model_xyz"), 3)
        if xyz is not None:
            raw = [*xyz, *([None] * 6), finite_number(record.get("model_gripper"))]
    state = finite_vector(record.get("measured_state"), 10)
    if state is None:
        debug = record.get("command_debug")
        xyz = finite_vector(debug.get("measured_xyz"), 3) if isinstance(debug, dict) else None
        state = [*(xyz or [None] * 3), *([None] * 7)]
    state[9] = finite_number(record.get("measured_gripper_open_fraction"))
    requested = finite_vector(record.get("action"), 10)
    pose = {}
    for label, value in (("raw", raw), ("requested", requested), ("measured", state)):
        pose[label + "_xyz_mm"], pose[label + "_rpy_deg"] = pose_channels(value)
    joints = finite_vector(record.get("measured_joint_positions"), 7)
    return {"at_ns": timestamp, "raw": raw, "requested": requested, **pose,
            "joint_degrees": [math.degrees(value) for value in joints] if joints is not None else None,
            "teleop_active": record.get("teleop_active") is True,
            "measured_state": state, "tau": measured,
            "predicted_tau": finite_vector(record.get("predicted_tau_J"), 7),
            "external_tau": finite_vector(record.get("measured_tau_ext"), 7)}


class TelemetryTail:
    def __init__(self, path, points=900):
        self.path = Path(path)
        self.rows = deque(maxlen=points)
        self.latest = {}
        self.stream = None
        self.identity = None
        self.modified_at = None
        self.epoch = 0
        self.invalid_lines = 0

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def clear(self):
        self.rows.clear()
        self.latest = {}
        self.epoch += 1

    def poll(self, byte_budget=2 * 1024 * 1024):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.close()
            return False
        identity = (stat.st_dev, stat.st_ino)
        if self.stream is None or self.identity != identity or stat.st_size < self.stream.tell():
            first_open = self.identity is None
            self.close()
            self.clear()
            self.stream = self.path.open("rb")
            self.identity = identity
            if first_open and stat.st_size > 4 * 1024 * 1024:
                self.stream.seek(stat.st_size - 4 * 1024 * 1024)
                self.stream.readline()
        self.modified_at = stat.st_mtime
        consumed = 0
        while consumed < byte_budget:
            position = self.stream.tell()
            line = self.stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                self.stream.seek(position)
                break
            consumed += len(line)
            try:
                record = json.loads(line, parse_constant=lambda constant: None)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.invalid_lines += 1
                continue
            if not isinstance(record, dict):
                self.invalid_lines += 1
                continue
            if record.get("event") in ("inference_restart", "deployment_session"):
                self.clear()
                self.latest = {"status": record["event"]}
                continue
            row = project_record(record)
            if row is None:
                continue
            if self.rows and (row["at_ns"] < self.rows[-1]["at_ns"] or (
                    finite_number(record.get("step")) is not None and
                    finite_number(self.latest.get("step")) is not None and record["step"] < self.latest["step"])):
                self.clear()
            self.rows.append(row)
            self.latest = {key: record.get(key) for key in (
                "step", "status", "inference_sequence", "action_index", "inference_ms", "chunk_age_ms",
                "motion_sent", "teleop_active")}
        return True

    def payload(self, exists):
        now = time.time()
        age = max(0., now - self.modified_at) if self.modified_at is not None else None
        if self.rows:
            monotonic_age = (time.monotonic_ns() - self.rows[-1]["at_ns"]) / 1e9
            age = max(age or 0., monotonic_age)
        status = "waiting_file" if not exists else "waiting_data" if not self.rows else "stale" if age > 2 else "live"
        origin = self.rows[0]["at_ns"] if self.rows else 0
        rows = [{**row, "seconds": (row["at_ns"] - origin) / 1e9} for row in self.rows]
        return {"schema": 1, "generated_at_ms": int(now * 1000), "log_age_s": age,
                "status": status, "log": str(self.path), "epoch": self.epoch,
                "invalid_lines": self.invalid_lines, "latest": self.latest, "rows": rows,
                "full_action_available": bool(rows and rows[-1]["raw"] and rows[-1]["raw"][3] is not None)}


def publish(destination, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    temporary = destination / "telemetry.js.tmp"
    temporary.write_text("window.receiveTelemetry(" + encoded + ");\n", encoding="utf-8")
    temporary.replace(destination / "telemetry.js")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("outputs/realtime_stage_A_3000.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/realtime_dashboard"))
    parser.add_argument("--points", type=int, default=900)
    parser.add_argument("--refresh-s", type=float, default=.25)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.points <= 3000 or not math.isfinite(args.refresh_s) or args.refresh_s < .1:
        parser.error("points must be 30..3000 and refresh-s must be finite and >= 0.1")
    destination = (ROOT / args.output).resolve()
    log = (ROOT / args.log).resolve()
    if not destination.is_relative_to(ROOT / "outputs") or destination == ROOT / "outputs":
        parser.error("output must be a subdirectory of force-VLA/outputs")
    if not log.is_relative_to(ROOT):
        parser.error("log must be inside force-VLA")
    destination.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name("dashboard_assets")
    for name in ("index.html", "dashboard.js", "dashboard.css"):
        shutil.copyfile(assets / name, destination / name)
    tail = TelemetryTail(log, args.points)
    print(f"Open in a browser: {destination / 'index.html'}", flush=True)
    print("Read-only curves. Ctrl+C stops this monitor, not the robot.", flush=True)
    try:
        while True:
            exists = tail.poll()
            publish(destination, tail.payload(exists))
            if args.once:
                return
            time.sleep(args.refresh_s)
    except KeyboardInterrupt:
        return
    finally:
        tail.close()


if __name__ == "__main__":
    main()
