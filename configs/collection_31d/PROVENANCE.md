# 31D collection provenance

- Data source: `/data0/sht/Evo-RLT/datacollection/franka_robotiq_single_left/rollout_dataset_franky_31d`
- Data content: LeRobot v3 dataset with 184 committed episodes, 177384 frames, 31D observation state, 10D action, two camera videos, success/failure episode metadata, and command traces.
- Follow-up batch: episodes 152-183; 31 success episodes and one controller-reflex anomaly at episode 170. Classification is maintained under `annotations/31d_classification/`.
- Data use: force-feedback VLA preprocessing and training experiments.
- Format source: `/data0/sht/Evo-RLT/datacollection/franka_robotiq_single_left/schema_31d.py`
- Collection source: copied 31D gamepad/Franky recorder files from `/data0/sht/Evo-RLT/datacollection/start_teleop/`
- Configuration source: copied `single_left_gamepad_31d.yaml` and `single_left_31d.yaml`.
- Training note: the copied 10D dataset-feature YAML is provenance only; it must not be used to convert the 31D dataset without a dedicated 31D training configuration.
