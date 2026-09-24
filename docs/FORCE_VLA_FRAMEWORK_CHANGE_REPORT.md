# Force-VLA 框架与代码改造报告

更新时间：2026-09-23

本文记录 `/data0/wx/force-VLA` 当前用于力反馈 VLA 的独立框架、关键改造文件、数据处理产物和未启用的实验分支。

## 1. 当前总体结构

```text
31D Franka 采集
  -> 成功/失败/控制故障分类
  -> 30Hz 时间对齐
  -> 过去2秒10帧 torque history
  -> 未来50步 action + measured torque target
  -> 训练集归一化和 episode 划分
  -> PI0.5 + torque adapter + LoRA
  -> 50步动作预测和未来力矩辅助损失
  -> 独立 force-VLA 实时推理入口
```

当前机器人执行输出仍为10D TCP/夹爪动作。预测的未来7D力矩只用于辅助训练和日志，不直接发送为机器人力控命令。

## 2. 采集格式改造

### 2.1 31D状态定义

文件：

- `src/collection_31d/schema_31d.py`
- `src/collection_31d/robot_31d.py`
- `src/collection_31d/gamepad_record_31d.py`
- `src/collection_31d/gamepad_recording_31d.py`
- `configs/collection_31d/single_left_gamepad_31d.yaml`

状态维度为：

```text
TCP位置                 3D
TCP rot6d姿态            6D
夹爪开度                 1D
Franka tau_J             7D
Franka tau_ext_hat_filtered 7D
关节速度 dq              7D
总计                    31D
```

动作仍为10D：TCP位置3D、rot6d姿态6D和夹爪开度1D。

`robot_31d.py` 从 libfranka/Franky 的 `RobotState` 读取：

- `tau_J`
- `tau_ext_hat_filtered`
- `dq`

### 2.2 与原采集流程的关系

31D采集保留原 Evo-RLT 的 Franka、相机、手柄和视频写入流程，只在 force-VLA 副本中增加31D状态字段。采集器内部导入路径已改为使用本地 `collection_31d` 副本；原 Evo-RLT 采集文件不变。

`RB` 产生的是人工关键阶段分割：

- `complementary_info.phase`
- `critical_intervals`
- `critical_human_frames`

它不是自动接触检测，也不是力矩标签。它表示操作者标记的纠正、接触或关键操作区间。

## 3. 数据分类与窗口处理

### 3.1 分类文件

当前最新分类位于：

- `annotations/31d_classification_v3/sft_candidate.json`
- `annotations/31d_classification_v3/failure_candidate.json`
- `annotations/31d_classification_v3/excluded_controller_fault.json`
- `annotations/31d_classification_v3/excluded_recording_fault.json`
- `annotations/31d_classification_v3/summary.json`

最新数据共40条：

```text
成功SFT候选             31条
人工失败                  1条
控制器/机器人异常          5条
相机/录制异常              3条
```

新增的 `episode 36` 因 Franka command success rate 异常排除；`episode 37、38、39` 纳入成功样本。

失败和控制故障数据仍保留在原始数据集中，只是不进入普通成功SFT窗口。它们可用于后续恢复能力评估和失败分类实验。

### 3.2 滑动窗口构建

实现文件：

- `src/build_force_vla_windows.py`
- `src/force_vla/pi05/window_dataset.py`
- `src/force_vla/pi05/preparation.py`

窗口契约：

```text
采样频率                  30Hz
历史力矩                  过去2秒，10个时间点，10×7
未来动作                  50×10
未来 measured torque      50×7
未来 torque 对齐           anchor之后的1到50帧
```

窗口构建会检查状态/动作维度、有限值、观测时间戳、最大时间间隔、历史采样误差、未来采样误差和 episode 边界。

最新产物：

- `outputs/force_vla_31d_windows_v3/windows.parquet`
- `outputs/force_vla_31d_windows_v3/metadata.json`
- `outputs/force_vla_prepared_v3/data_contract.json`

统计结果：57362个有效窗口，51718个训练窗口，5644个验证窗口。归一化统计只使用训练episode，避免验证轨迹泄漏。

## 4. PI0.5 力反馈模型改造

### 4.1 配置和模型入口

主要文件：

- `configs/pi05_force_tavla_aligned.json`
- `src/force_vla/pi05/configuration_pi05_force.py`
- `src/force_vla/pi05/modeling_pi05_force.py`
- `src/force_vla/pi05/torque_adapter.py`
- `src/force_vla/pi05/temporal.py`

当前配置：

```text
state_dim                  10
torque_dim                  7
torque_history_steps       10
future_steps                50
action_dim                  10
future_torque_loss_weight  0.1
conditioning_layout        separate
trainable                  lora
batch_size                  4
```

### 4.2 Torque Adapter

文件：`src/force_vla/pi05/torque_adapter.py`

`TorqueAdapter` 接收 `[B, 10, 7]` 的过去力矩历史，将其压缩为一个条件token。该token与状态投影一起注入 PI0.5 action expert 的条件序列。

当前 `separate` 布局为两个条件token：

```text
state -> state_proj -> state token
10×7 torque history -> TorqueAdapter -> torque token
```

旧的 `fused` 布局仍保留兼容路径，但当前对齐配置使用 `separate`。

### 4.3 动作与未来力矩联合预测

文件：`src/force_vla/pi05/modeling_pi05_force.py`

模型的流匹配输入和目标扩展为：

```text
未来动作10D + 未来 measured torque 7D = 17D
```

输出拆分为：

- `flow_action_out_proj`：未来50步10D动作
- `torque_out_proj`：未来50步7D力矩

联合损失为：

```text
loss = action_loss + 0.1 × torque_loss
```

未来力矩目标用于约束时序和接触相关预测，不作为机器人执行指令。

## 5. 当前训练方式

训练入口：

- `scripts/train_force_pi05.sh`
- `src/force_vla/pi05/train_pi05_force.py`
- `src/force_vla/pi05/training.py`
- `src/force_vla/pi05/lora.py`

当前不是全量微调，而是 PI0.5 基座上的 LoRA 和新增适配器训练：

```text
可训练：torque_adapter、state_proj、融合层、动作/力矩投影、LoRA参数
冻结：PI0.5视觉语言主体和未选中的基础模型参数
```

`training.py` 中的 `configure_trainable()` 控制三种模式：

- `adapters`：只训练新增适配器
- `expert`：训练 action expert
- `lora`：当前使用，训练新增适配器和LoRA参数

当前30k训练会话：

```text
tmux session: force-vla-train-v3
output: outputs/pi05_force_v3_run
log: outputs/force_vla_v3_train.log
```

启动命令：

```bash
bash scripts/train_force_pi05.sh \
  --train \
  --steps 30000 \
  --config configs/pi05_force_tavla_aligned.json \
  --windows outputs/force_vla_31d_windows_v3/windows.parquet \
  --contract outputs/force_vla_prepared_v3/data_contract.json \
  --output outputs/pi05_force_v3_run
```

## 6. 外部力矩/末端六维力分支状态

数据窗口已经保留：

```text
external_joint_torque_history
future_external_torque
```

但当前模型配置为：

```json
"use_external_torque": false
```

`configuration_pi05_force.py` 当前将外部力矩作为独立消融分支保留，尚未接入当前训练主路径。因此“数据已采集/窗口已保存”不等于“当前模型已经使用外部力矩”。

若后续启用末端六维外力，需要在 force-VLA 副本中新增：

1. 6D外力窗口的模型读取和归一化。
2. 6D外力适配器或与7D关节外力矩的融合层。
3. 实时推理阶段的同步观测输入。
4. measured torque、external joint torque、末端六维力的消融评估。

不需要修改 Evo-RLT 原始模型和控制接口。

## 7. 实时推理改造

主要文件：

- `src/force_vla/pi05/realtime.py`
- `src/force_vla/pi05/realtime_control.py`
- `src/force_vla/pi05/teleop_assist.py`
- `scripts/run_force_realtime.sh`
- `configs/deployment_force_vla.yaml`
- `configs/deployment_force_vla_speed_test.yaml`

当前实时输入为两路RGB、10D状态和过去2秒的10×7 measured torque历史。主要控制改造包括：

- 异步推理和动作块消费。
- 8Hz推理请求上限配置。
- 首次推理前等待完整10帧真实力矩历史。
- 观测短暂超时时重试，不使用过期图像继续发动作。
- 夹爪 `--startup-open` 初始打开流程。
- `q`、`x`、`Esc` 键盘退出。
- `--teleop-assist` 手柄辅助修正。
- 动作块衔接和平滑控制。
- 预测未来力矩写入日志但不发送为力控命令。

部署入口不会复用原 Evo-RLT 的标准 PI0.5部署器，因为原部署器不识别 force adapter checkpoint 和力矩历史。

## 8. 启动器和隔离边界

文件：`scripts/run.sh`

该启动器负责：

- 使用 `/data0/wx/force-VLA/env`。
- 清除继承的 ROS/Python 路径污染。
- 只读挂载数据、模型和参考源码。
- 将输出、缓存和HOME留在 force-VLA。
- 可选挂载GPU设备。

已增加条件挂载：旧 `datasets/` 被删除时不再导致 bwrap 启动失败。训练和离线预处理可继续使用 `scripts/run.sh`；真实硬件采集和推理仍需使用各自明确的硬件入口。

## 9. 当前结论与后续实验

当前版本已经完成：

- 31D Franka采集格式。
- 成功/失败/控制故障分类。
- 10帧历史力矩窗口。
- 50步动作和未来力矩联合目标。
- PI0.5 torque adapter和LoRA训练链路。
- 独立实时推理入口。

当前版本尚未完成：

- 外部关节力矩正式接入主模型。
- 末端六维力正式接入主模型。
- 自动接触事件检测。
- 基于力矩的放置成功率验证。

推荐消融顺序：

```text
视觉+状态
视觉+状态+measured torque history
视觉+状态+measured torque history+external joint torque
视觉+状态+关节力矩+末端六维力
```

每个版本都应使用相同episode划分，并比较动作误差、未来力矩误差、关键阶段误差和真机放置成功率。
