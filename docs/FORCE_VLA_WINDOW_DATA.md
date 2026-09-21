# 力矩 VLA 窗口数据

构建命令：

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh python src/build_force_vla_windows.py
```

输入使用 `data/franka_single_left_31d` 和 `annotations/31d_classification/sft_candidate.json`，因此只使用已标记成功、未列入控制器或录制异常的 episode。原始31D数据保持不变。

输出目录为 `outputs/force_vla_31d_windows/`。`windows.parquet` 每一行是一个锚点窗口，包含：

- `base_video_path`、`left_wrist_video_path` 及锚点帧号，训练读取器据此解码两路视觉；
- `task_text`，当前任务文本来自 LeRobot 的任务元数据；
- `state`，锚点的10D模型状态：TCP位置3、rot6d姿态6、夹爪1；这与10D动作维度匹配 TA-VLA 的状态接口；
- `state_with_joint_velocity`，额外保留17D状态（10D TCP/夹爪 + 7D关节速度），供后续模型消融，不作为第一版 TA-VLA 主状态；
- `joint_torque_history`，锚点前的10个时间采样点，形状为 `10×7`，最后一个点是当前时刻；
- `future_actions`，锚点时刻到后49步，形状为 `50×10`；
- `future_joint_torque`，锚点后1步到后50步，形状为 `50×7`；
- `external_joint_torque_history` 和 `future_external_joint_torque`，与关节力矩相同的时间窗口，作为可选消融信号；
- `observation_state_31d`、时间戳、原始数据帧号和 episode 成功标签，便于追溯。

动作与未来力矩的错位是有意设计：TA-VLA 代码将动作序列读取为 `t..t+49`，将未来 effort 读取为 `t+1..t+50`，然后拼接为联合监督目标。历史偏移在30Hz下为 `[-60,-53,-47,-40,-33,-27,-20,-13,-7,0]` 帧，约覆盖过去2秒；历史和未来窗口绝不跨 episode。

第一版建议使用 TA-VLA 的 `EXPERT_HIS_C_FUT`：只使用 measured joint torque（`tau_J`）作为7D effort 输入和未来力矩监督。它与论文中的关节 effort 定义直接对应。`tau_ext_hat_filtered` 是根据动力学估计的外部关节力矩，可能与 measured torque 重复且含估计噪声；先保存、不参与第一版主损失，待关节力矩基线跑通后做外部力矩消融。若做外部力矩实验，应重新计算训练统计并保持 train/validation 的归一化一致。

当前生成统计写入 `outputs/force_vla_31d_windows/metadata.json`：101条成功 episode、101883个窗口。该文件是窗口索引与数组的离线产物，不代表已经完成 TA-VLA 模型训练。
