# Force-VLA-Lab isolation contract

- Work only in this directory. Never modify `/data0/sht/Evo-RLT`, `/data0/miniconda3/envs/evo-rlt`, the base Conda installation, or the original shared model directories.
- Run Python and tests only through `scripts/run.sh`. It mounts everything outside this directory read-only, clears inherited environment variables, disables network access, and hides robot devices.
- The environment is `./env`, not the original Conda environment. Do not activate or install packages into the original environment. Never use its package cache for writes.
- `datasets`, `models`, and `reference` are independent copied snapshots and are mounted read-only during experiments. Put annotations in `annotations`, new code in `src`, and generated artifacts in `outputs`.
- Do not create hardlinks or source-pointing symlinks for mutable files. Do not add the old workspace or ROS paths to Python's import path.
- Treat copied historical configuration paths as provenance, not executable runtime configuration. Use explicit local checkpoint/tokenizer paths.
- Do not start robot collectors, controllers, ROS nodes, or hardware connections. GPU access is opt-in via `scripts/run.sh --gpu`.
- Do not guess which 80 RLT episodes the user meant. All 113 are preserved; an explicit selection manifest is required for an 80-episode experiment.
- The new 100 demonstrations have placeholder language `111`. Do not fabricate episode labels.
- The auxiliary force MLP smoke test is not VLA training and must never be described as such.
- The user explicitly authorized this independent directory to be initialized and published to `https://github.com/Cynthia-5hs3/froce-VLA` on 2026-09-15. Use only that repository for this project's remote. The separate source repository `/data0/sht/Evo-RLT` retains its sole approved remote `https://github.com/MINT-SJTU/Evo-RLT`; do not change it.
