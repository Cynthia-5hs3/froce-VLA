#!/usr/bin/env bash
set -euo pipefail
FORCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec /usr/bin/flock --nonblock "$FORCE_ROOT/runtime/force_pi05_train.lock" \
    bash "$FORCE_ROOT/scripts/run.sh" --gpu python -u -m force_vla.pi05.extended_training "$@"
