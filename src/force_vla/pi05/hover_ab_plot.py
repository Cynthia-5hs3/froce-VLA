"""Plot saved offline predictions without importing Torch or loading models."""

import argparse
import json
from pathlib import Path
import sys

if sys.flags.no_site:
    sys.path.append("/usr/lib/python3/dist-packages")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    destination = args.directory.resolve()
    manifest = json.loads((destination / "manifest.json").read_text())
    selections = manifest["selections"]
    saved = {label: np.load(destination / f"{label}_predictions.npz") for label in manifest["checkpoints"]}
    reference = next(iter(saved.values()))
    states, targets = reference["states"], reference["targets"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    timeline = np.arange(50) / 30
    for axis, category in zip(axes, ("down", "z_hold", "up")):
        indices = [index for index, item in enumerate(selections) if item["target_class20"] == category
                   and (not args.fixture_only or (.17 <= item["xyz"][0] <= .23 and -.48 <= item["xyz"][1] <= -.42))]
        if not indices:
            continue
        target_delta = 1000 * (targets[indices, :, 2] - states[indices, None, 2])
        axis.plot(timeline, target_delta.mean(axis=0), color="black", label="Demonstration")
        for label, data in saved.items():
            delta = 1000 * (data["actions"][indices, :, :, 2] - states[indices, None, None, 2])
            mean = delta.mean(axis=(0, 1))
            line = axis.plot(timeline, mean, label=label)[0]
            seed_std = delta.std(axis=1).mean(axis=0)
            axis.fill_between(timeline, mean - seed_std, mean + seed_std,
                              alpha=.15, color=line.get_color())
        axis.axhline(0, color="gray", linewidth=.5)
        axis.set(title=f"Reference at step 20: {category}", xlabel="Action offset (s)", ylabel="Z displacement (mm)")
        axis.legend()
    figure.suptitle("Paired observations; shading = mean within-observation seed standard deviation")
    basename = "fixture_vertical_predictions" if args.fixture_only else "vertical_predictions"
    figure.savefig(destination / f"{basename}.png", dpi=180)
    figure.savefig(destination / f"{basename}.pdf")
    plt.close(figure)
    for data in saved.values():
        data.close()
    print(destination / f"{basename}.png")


if __name__ == "__main__":
    main()
