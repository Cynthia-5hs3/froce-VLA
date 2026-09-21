#!/usr/bin/env bash
set -euo pipefail
FORCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$FORCE_ROOT/scripts/run.sh" --gpu python -m force_vla.pi05.train_pi05_force "$@"
