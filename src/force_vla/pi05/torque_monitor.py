"""Read-only live action and torque display from an inference JSONL log."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import shutil
import time


LEVELS = " .:-=+*#%@"


def finite_vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        return None
    try:
        converted = [float(item) for item in value]
    except (ValueError, TypeError, OverflowError):
        return None
    return converted if all(math.isfinite(item) for item in converted) else None


def read_new_records(stream, measured, predicted, latest=None):
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            break
        if not line.endswith("\n"):
            stream.seek(position)
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("event") == "inference_restart":
            measured.clear()
            predicted.clear()
            if latest is not None:
                latest.clear()
                latest.update(record)
            continue
        measured_value = finite_vector(record.get("measured_tau_J"), 7)
        if measured_value is None:
            continue
        measured.append(measured_value)
        predicted.append(finite_vector(record.get("predicted_tau_J"), 7))
        if latest is not None:
            latest.clear()
            latest.update(record)


def sparkline(values, low, high, width):
    if not values:
        return " " * width
    selected = values[-width:]
    scale = max(high - low, 1e-6)
    result = "".join(" " if value is None else LEVELS[min(len(LEVELS) - 1, max(0, int((value - low) / scale * (len(LEVELS) - 1))))]
                     for value in selected)
    return result.rjust(width)


def format_vector(value, size):
    vector = finite_vector(value, size)
    return "[" + ", ".join(f"{item:.4f}" for item in vector) + "]" if vector is not None else "n/a"


def render(measured, predicted, width, latest=None):
    latest = latest or {}
    timestamp = latest.get("control_at_ns")
    age = max(0., (time.monotonic_ns() - timestamp) / 1e9) if isinstance(timestamp, int) else None
    freshness = f"log age={age:.2f}s" if age is not None else "waiting for telemetry"
    if age is not None and age > 2:
        freshness += " [STALE: no recent record]"
    lines = ["Force-VLA action + torque monitor (read-only); Ctrl+C closes this display",
             f"step={latest.get('step', 'n/a')} status={latest.get('status', latest.get('event', 'n/a'))} "
             f"inference={latest.get('inference_sequence')} action_index={latest.get('action_index')} "
             f"motion_sent={latest.get('motion_sent')} | {freshness}",
             "Action order: XYZ [m], rotation6D [unitless, not Euler angles], gripper [open fraction]",
             "Model raw : " + format_vector(latest.get("model_action"), 10),
             "Requested : " + format_vector(latest.get("action"), 10),
             "Measured  : " + format_vector(latest.get("measured_state"), 10)]
    if "model_action" not in latest and latest.get("model_xyz") is not None:
        lines.append("Old log: raw XYZ=" + format_vector(latest.get("model_xyz"), 3)
                     + "; full raw action unavailable (requested action is not substituted)")
    debug = latest.get("command_debug")
    if isinstance(debug, dict) and debug:
        lines.append("Controller target XYZ [m]: " + format_vector(debug.get("effective_xyz"), 3)
                     + f" | commanded gripper={debug.get('commanded_gripper', 'n/a')}")
    lines.extend(["External joint torque estimate [Nm]: " + format_vector(latest.get("measured_tau_ext"), 7),
                  "M=tau_J sensor [Nm]; P=selected model torque [Nm], not a commanded torque",
                  "Curves share record columns; P is a forecast, not an exactly synchronized error."])
    for joint in range(7):
        measured_values = [row[joint] for row in measured]
        predicted_values = [row[joint] if row is not None else None for row in predicted]
        combined = measured_values + [value for value in predicted_values if value is not None]
        if combined:
            low, high = min(combined), max(combined)
        else:
            low, high = 0.0, 1.0
        lines.append(f"J{joint + 1} M {sparkline(measured_values, low, high, width)} {measured_values[-1]:8.3f}" if measured_values
                     else f"J{joint + 1} M {' ' * width}      n/a")
        predicted_last = f"{predicted_values[-1]:8.3f}" if predicted_values and predicted_values[-1] is not None else "     n/a"
        lines.append(f"   P {sparkline(predicted_values, low, high, width)} {predicted_last}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--points", type=int, default=120)
    parser.add_argument("--refresh-s", type=float, default=0.25)
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--once", action="store_true", help="Read existing records, print one snapshot, and exit")
    args = parser.parse_args()
    if args.points < 10 or args.refresh_s <= 0:
        parser.error("points must be at least 10 and refresh-s must be positive")
    if args.once and not args.log.is_file():
        parser.error("--once requires an existing log")
    try:
        while not args.log.is_file():
            print(f"Waiting for {args.log} ...", end="\r", flush=True)
            time.sleep(args.refresh_s)
        measured, predicted = deque(maxlen=args.points), deque(maxlen=args.points)
        latest = {}
        with args.log.open("r", encoding="utf-8") as stream:
            if not (args.from_start or args.once):
                stream.seek(0, 2)
            while True:
                read_new_records(stream, measured, predicted, latest)
                width = max(20, min(args.points, shutil.get_terminal_size((120, 30)).columns - 18))
                prefix = "" if args.once else "\x1b[2J\x1b[H"
                print(prefix + render(list(measured), list(predicted), width, latest), flush=True)
                if args.once:
                    return
                time.sleep(args.refresh_s)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
