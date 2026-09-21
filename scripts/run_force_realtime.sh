#!/usr/bin/env bash
set -euo pipefail
FORCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" == "--check" ]]; then
    shift
    exec bash "$FORCE_ROOT/scripts/run.sh" --gpu python -m force_vla.pi05.realtime --mode check "$@"
fi
if [[ "${FORCE_VLA_HARDWARE:-0}" != "1" ]]; then
    echo "Hardware access requires FORCE_VLA_HARDWARE=1" >&2
    exit 2
fi
export PYTHONPATH="$FORCE_ROOT/src:$FORCE_ROOT/reference/evo-rlt/src"
exec env -i \
    PATH="$FORCE_ROOT/env/bin:/usr/bin:/bin" \
    HOME="$FORCE_ROOT/runtime/home" \
    LANG=C.UTF-8 \
    CONDA_PREFIX="$FORCE_ROOT/env" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PYTHONPATH" \
    HF_HOME="$FORCE_ROOT/runtime/cache/huggingface" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME="$FORCE_ROOT/runtime/cache" \
    TOKENIZERS_PARALLELISM=false \
    "$FORCE_ROOT/env/bin/python" -u -m force_vla.pi05.realtime "$@"
