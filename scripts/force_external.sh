#!/usr/bin/env bash
set -euo pipefail
FORCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$FORCE_ROOT"
VARIANT="${1:-}"
MODE="${2:-}"
case "$VARIANT" in
    success_30k)
        PREPARED=outputs/pi05_external_success_30k/external_success_30k_prepared
        TRAINED=outputs/pi05_external_success_30k
        EVALUATION="$TRAINED/eval_external_success_30k"
        TELEMETRY="$TRAINED/realtime_external_success_30k.jsonl"
        STEP=030000
        BASELINE=outputs/pi05_force_tavla_aligned_30k_resume/checkpoint-030000
        WINDOWS=outputs/force_vla_31d_windows/windows.parquet
        ;;
    stage_b)
        PREPARED=outputs/external_stage_b_prepared
        TRAINED=outputs/pi05_external_stage_b_3k
        EVALUATION=outputs/eval_external_stage_b
        TELEMETRY=outputs/realtime_external_stage_b.jsonl
        STEP=003000
        BASELINE=outputs/pi05_stage_measured_3k/checkpoint-003000
        WINDOWS=outputs/force_vla_31d_windows_v3/windows.parquet
        ;;
    *) echo "Usage: bash scripts/force_external.sh {success_30k|stage_b} {prepare|check|train|offline|model-check|observe|execute} [options]" >&2; exit 2 ;;
esac
if [[ $# -lt 2 ]]; then
    echo "An explicit mode is required; nothing was started." >&2
    exit 2
fi
shift 2
case "$MODE" in
    prepare|check)
        exec bash scripts/run.sh python -m force_vla.pi05.external_experiments --variant "$VARIANT" "--$MODE" "$@" ;;
    train)
        exec /usr/bin/flock --nonblock runtime/force_pi05_train.lock \
            bash scripts/run.sh --gpu python -u -m force_vla.pi05.training --train \
            --config "$PREPARED/config.json" --output "$TRAINED" --device cuda:0 "$@" ;;
    offline)
        exec bash scripts/run.sh --gpu python -m force_vla.pi05.hover_ab_evaluation \
            --checkpoint-a "$BASELINE" --checkpoint-b "$TRAINED/checkpoint-$STEP" \
            --windows "$WINDOWS" --output "$EVALUATION" --device cuda:0 "$@" ;;
    model-check)
        exec bash scripts/run_force_realtime.sh --check --checkpoint "$TRAINED/checkpoint-$STEP" \
            --robot-config configs/deployment_force_vla_smooth_test.yaml --device cuda:0 "$@" ;;
    observe|execute)
        exec bash scripts/run_force_realtime.sh --mode "$MODE" --allow-hardware \
            --checkpoint "$TRAINED/checkpoint-$STEP" \
            --robot-config configs/deployment_force_vla_smooth_test.yaml \
            --log "$TELEMETRY" --device cuda:0 "$@" ;;
    *) echo "Unknown mode: $MODE; nothing was started." >&2; exit 2 ;;
esac
