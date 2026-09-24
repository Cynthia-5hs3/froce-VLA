"""Live terminal curves for measured and predicted joint torque JSONL fields."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import shutil
import time


LEVELS = " .:-=+*#%@"


def read_new_records(stream, measured, predicted):
    while True:
        line = stream.readline()
        if not line:
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        measured_value = record.get("measured_tau_J")
        predicted_value = record.get("predicted_tau_J")
        if isinstance(measured_value, list) and len(measured_value) == 7:
            measured.append([float(value) for value in measured_value])
        if isinstance(predicted_value, list) and len(predicted_value) == 7:
            predicted.append([float(value) for value in predicted_value])


def sparkline(values, low, high, width):
    if not values:
        return " " * width
    selected = values[-width:]
    scale = max(high - low, 1e-6)
    result = "".join(LEVELS[min(len(LEVELS) - 1, max(0, int((value - low) / scale * (len(LEVELS) - 1))))]
                     for value in selected)
    return result.rjust(width)


def render(measured, predicted, width):
    lines = ["Force-VLA live torque: measured (M) and predicted (P), Ctrl+C exits"]
    for joint in range(7):
        measured_values = [row[joint] for row in measured]
        predicted_values = [row[joint] for row in predicted]
        combined = measured_values + predicted_values
        if combined:
            low, high = min(combined), max(combined)
        else:
            low, high = 0.0, 1.0
        lines.append(f"J{joint + 1} M {sparkline(measured_values, low, high, width)} {measured_values[-1]:8.3f}" if measured_values
                     else f"J{joint + 1} M {' ' * width}      n/a")
        lines.append(f"   P {sparkline(predicted_values, low, high, width)} {predicted_values[-1]:8.3f}" if predicted_values
                     else f"   P {' ' * width}      n/a")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--points", type=int, default=120)
    parser.add_argument("--refresh-s", type=float, default=0.25)
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()
    if args.points < 10 or args.refresh_s <= 0:
        parser.error("points must be at least 10 and refresh-s must be positive")
    while not args.log.is_file():
        print(f"Waiting for {args.log} ...", end="\r", flush=True)
        time.sleep(args.refresh_s)
    measured, predicted = deque(maxlen=args.points), deque(maxlen=args.points)
    with args.log.open("r", encoding="utf-8") as stream:
        if not args.from_start:
            stream.seek(0, 2)
        try:
            while True:
                read_new_records(stream, measured, predicted)
                width = max(20, min(args.points, shutil.get_terminal_size((120, 30)).columns - 18))
                print("\x1b[2J\x1b[H" + render(list(measured), list(predicted), width), flush=True)
                time.sleep(args.refresh_s)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
