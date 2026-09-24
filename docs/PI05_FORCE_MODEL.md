# Torque-aware PI0.5

`src/force_vla/pi05/modeling_pi05_base.py` is a local copy of the PI0.5
implementation used by the Evo-RLT environment. The force variant subclasses
that copy in `modeling_pi05_force.py`; no file under `/data0/sht/Evo-RLT` or
the shared Conda environment is imported for mutation.

The implementation adapts TA-VLA's `EXPERT_HIS_C_FUT` to PyTorch PI0.5.
TA-VLA uses separate effort/state tokens (DePost). The original experiment
and existing 30k checkpoints use the default `conditioning_layout="fused"`,
while retaining PI0.5's state text prompt:

```text
state (10) ───────────────┐
                           ├─ concat ─ projection ─ state/torque condition token
10 history frames × 7 Nm ─┘

future action (10) + future joint torque (7)
    └─ action padding to 32 ─ 39D flow matching input ─ 50 expert tokens
```

The condition token is prepended to the action expert suffix. Its attention
mask is `[1, 1, 0, ..., 0]`: the visual/language prefix cannot read the
condition, the condition cannot read flow tokens, and all 50 flow tokens
attend bidirectionally within their block. The output has 32 padded action
dimensions plus seven future torque dimensions. Training uses the first ten
action dimensions and a weighted seven-dimensional future torque loss
(`future_torque_loss_weight=0.1`). The predicted torque is an auxiliary target;
it is never sent to the robot as a command.

The old PI0.5 action projections seed the new flow action projections. The
torque adapter, state fusion, and torque output layers are new parameters. Use
`load_pi05_backbone` to require every backbone tensor with the correct shape,
then seed the new action projections after loading. New parameter names are
recorded explicitly. Per-joint train-only normalization preserves torque
magnitude; the adapter does not apply per-window LayerNorm.

## Window contract

`outputs/force_vla_31d_windows/windows.parquet` is read by
`ForceVLAWindowDataset`. Each row supplies:

- `observation.state`: current 10D TCP pose/rotation/夹爪 state;
- `joint_torque_history`: causal `10×7` measured `tau_J` history;
- `action`: future `50×10` action target;
- `future_joint_torque`: future `50×7` measured torque target;
- task text and episode/frame provenance.

The history and future windows are created without crossing episode boundaries.
The model's `ForcePI05Policy` expects tokenized language and image tensors in
the same format as PI0.5, plus the two custom torque fields above. Set
`include_images=True` on `ForceVLAWindowDataset` and call
`attach_visual_language_inputs` to decode the referenced RGB frames and
tokenize each task string. The loader contract can be checked without loading
the large VLM:

```bash
bash scripts/run.sh python -m force_vla.pi05.train_pi05_force --check-data
```

The current first version uses measured joint torque only. External torque is
retained in the window parquet for a later ablation and is deliberately
rejected by `ForcePI05Config.use_external_torque` until a separate comparison
is defined.

## Paper-aligned configuration

`configs/pi05_force_tavla_aligned.json` selects `conditioning_layout="separate"`:

```text
10 samples over 2 seconds × 7 Nm → MLP → torque token
10D state → projection → state token
decoder suffix = [torque token, state token, 50 noisy action/torque tokens]
```

The suffix block mask is `[1, 1, 1, 0, ..., 0]`. The torque token cannot see
state/actions, the state token can see torque, and action tokens can see both.
This follows the official `EXPERT_HIS_C_FUT` sequence and mask. Concatenation
here is along the **token sequence**, not merging both features into one state
token. Paper Sec. 4.1 also evaluates DePre, which concatenates torque into state,
but Sec. 6 adopts DePost-1 Token. The legacy fused model is a separate experiment.

The new `training.trainable="lora"` mode trains low-rank updates in both the
PaliGemma language transformer and the action expert, plus the torque/state/flow
projections and time MLP. Q/K/V/O and gate/up/down projections receive LoRA.
Encoder rank/alpha=16/16 and expert rank/alpha=32/32 follow the official Gemma
variant settings. Base transformer parameters and SigLIP stay frozen; this is
parameter-efficient fine-tuning, not full fine-tuning. The local PI0.5/SigLIP
freezing, two cameras and TCP state remain embodiment/implementation adaptations
to the paper's main PI0 setup, not a bit-identical JAX reproduction.

New token layout and LoRA parameters require a new training run. Do not load
the old fused adapter as a complete separate-token model. Old configurations
without `conditioning_layout` continue to restore as fused; the checkpoint
loader requires an exact match of trainable parameter names.

Sparse critical marks can select training windows and evaluation groups. They
are not needed at inference and are not labels of physical contact/no-contact.
See [alignment and phase evaluation](TA_VLA_ALIGNMENT.md).

独立环境、数据准备、训练和保存/恢复步骤见 [训练说明](FORCE_TRAINING.md)。
