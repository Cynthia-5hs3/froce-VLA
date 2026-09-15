#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then
    shift
    for device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-caps; do
        if [[ -e "$device" ]]; then
            GPU_ARGS+=(--dev-bind "$device" "$device")
        fi
    done
fi
if [[ $# -eq 0 ]]; then
    set -- "$ROOT/env/bin/python" "$ROOT/src/lab.py" inspect
fi
exec /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 /usr/bin/bwrap \
    --ro-bind / / --bind "$ROOT" "$ROOT" \
    --ro-bind "$ROOT/datasets" "$ROOT/datasets" \
    --ro-bind "$ROOT/models" "$ROOT/models" \
    --ro-bind "$ROOT/reference" "$ROOT/reference" \
    --proc /proc --dev /dev "${GPU_ARGS[@]}" --tmpfs /tmp \
    --unshare-net --die-with-parent --new-session --chdir "$ROOT" \
    /usr/bin/env -i \
    PATH="$ROOT/env/bin:/usr/bin:/bin" HOME="$ROOT/runtime/home" LANG=C.UTF-8 \
    CONDA_PREFIX="$ROOT/env" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$ROOT/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    HF_HOME="$ROOT/runtime/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME="$ROOT/runtime/cache" XDG_CONFIG_HOME="$ROOT/runtime/config" \
    XDG_DATA_HOME="$ROOT/runtime/data" TORCH_HOME="$ROOT/runtime/cache/torch" \
    TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime/cache/torchinductor" \
    TRITON_CACHE_DIR="$ROOT/runtime/cache/triton" CUDA_CACHE_PATH="$ROOT/runtime/cache/cuda" \
    NUMBA_CACHE_DIR="$ROOT/runtime/cache/numba" MPLCONFIGDIR="$ROOT/runtime/cache/matplotlib" \
    PIP_CACHE_DIR="$ROOT/runtime/cache/pip" UV_CACHE_DIR="$ROOT/runtime/cache/uv" \
    CONDA_PKGS_DIRS="$ROOT/runtime/conda-pkgs" CONDA_ENVS_PATH="$ROOT/runtime/conda-envs" \
    CONDARC="$ROOT/runtime/condarc" TMPDIR="$ROOT/runtime/tmp" \
    WANDB_MODE=disabled WANDB_DIR="$ROOT/runtime/cache/wandb" \
    TOKENIZERS_PARALLELISM=false "$@"
