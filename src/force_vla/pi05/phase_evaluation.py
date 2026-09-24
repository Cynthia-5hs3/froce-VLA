"""Evaluate human-marked windows offline, or regroup existing evaluations."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as parquet

from .evaluation import evaluate_loss, evaluate_samples, summarize_phases
from .phases import phase_values
from .preparation import local_path, output_path, write_json


def regroup_existing(dataset, evaluation):
    phases = phase_values(dataset)
    labels = {(int(episode), int(frame)): float(phase) for episode, frame, phase in zip(
        dataset.table["episode_index"].to_pylist(), dataset.table["anchor_frame"].to_pylist(), phases, strict=True)}
    groups = evaluation.get("variants", {"loss": evaluation})
    result = {}
    for name, group in groups.items():
        records = [dict(record, anchor_phase=labels[(record["episode"], record["frame"])])
                   for record in group["records"]]
        fields = list(group["episode_macro_mean"])
        result[name] = summarize_phases(records, fields)
    return {"by_variant": result, "new_model_inference": False,
            "note": "Existing sampled windows only. Unmarked does not mean no contact; counts indicate coverage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/pi05_force_sft_30k/checkpoint-030000")
    parser.add_argument("--existing", help="Regroup this existing loss/inference JSON without loading the model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-phase-per-episode", type=int, default=16)
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()
    checkpoint = local_path(args.checkpoint)
    run = json.loads((checkpoint / "run.json").read_text())
    windows = local_path(run["data_contract"]["windows"])
    destination = output_path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    if args.existing:
        table = parquet.read_table(windows, columns=["episode_index", "anchor_frame", "anchor_phase"])
        existing = json.loads(local_path(args.existing).read_text())
        result = regroup_existing(SimpleNamespace(table=table), existing)
        result.update(checkpoint=str(checkpoint), source_evaluation=args.existing)
    else:
        from transformers import AutoTokenizer

        from .training import restore_for_inference
        from .window_dataset import ForceVLAWindowDataset

        dataset = ForceVLAWindowDataset(windows, include_images=True)
        indices = np.flatnonzero(np.isin(np.asarray(dataset.table["episode_index"]),
                                         run["data_contract"]["validation_episodes"]))
        policy, normalizer = restore_for_inference(checkpoint, args.device)
        tokenizer = AutoTokenizer.from_pretrained(str(local_path(run["settings"]["tokenizer"])), local_files_only=True)
        if args.samples:
            result = evaluate_samples(policy, dataset, indices, normalizer, tokenizer, args.device, {
                "phase_balanced_evaluation": True, "inference_windows_per_episode": args.per_phase_per_episode,
                "inference_seeds": [12345, 23456], "inference_steps": 10, "history_variants": ["measured"]})
        else:
            result = evaluate_loss(policy, dataset, indices, normalizer, tokenizer, args.device,
                                   args.per_phase_per_episode, phase_balanced=True)
        result.update(checkpoint=str(checkpoint), new_model_inference=True, hardware_tested=False)
    write_json(destination, result)
    print(json.dumps({"output": str(destination), "new_model_inference": result["new_model_inference"]}))


if __name__ == "__main__":
    main()
