# Force-VLA 真机推理测试

实时入口是 `src/force_vla/pi05/realtime.py`，只在 `/data0/wx/force-VLA` 内运行。它读取 `tau_J`，按过去2秒的固定10个时间点构建7维力矩历史，接收两路224×224 RGB图像，预测50步10D TCP/夹爪动作。预测的7D未来力矩只用于日志，不发送给机器人。

实时日志由启动命令的 `--log` 指定。每个策略控制周期记录当前7维实测力矩 `measured_tau_J`，以及当前动作索引对应的7维预测力矩 `predicted_tau_J`。在第二个终端中可实时查看两组曲线：

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh python -m force_vla.pi05.torque_monitor \
  --log outputs/realtime_execute_v3_continuous.jsonl
```

监视器默认只显示启动后追加的数据；增加 `--from-start` 可从日志开头回放。按 `Ctrl+C` 只退出曲线监视器，不影响真机推理进程。

2026-09-21 起，观测采样与控制各自目标30Hz，推理请求默认8Hz，三者解耦。每次按观测时间戳跳过已过期的动作索引，顺序消费动作块；新块默认用3步过渡衔接（夹爪不平均）。`--chunk-blend-steps 0` 关闭过渡。`--rate-hz` 仅控制推理请求上限，不改变示范/执行时间尺度。`--async-inference` 保留为兼容参数，默认已经异步。

启动和遥操释放后等待完整2秒实测历史，不复制当前力矩填充；首次CUDA预热可能额外增加等待。首次有效预测前不发送动作。后续模型未返回有效块或块超过50步时保持当前位姿与上次夹爪目标，避免用正在变化的实测开度反向覆盖开合指令。没有新鲜观测时不发送新动作，且不会重复发送上一条动作；已经启动的有限时长运动段仍可能执行到结束。RealSense或夹爪短暂超时时，观测线程默认每20ms重试，最多等待2秒；超过该安全窗口才退出并保留底层异常。控制日志记录 `action_index`、`chunk_age_ms`、`skipped_steps_on_arrival` 和实际 `history_timestamps_ns`，可检查是否持续推进。旧30k checkpoint 可直接使用这一时间对齐路径，无需先重训。

原 Evo-RLT 的标准 PI0.5 部署器不能加载 `force-VLA` 的 adapter 增量，也不会提供力矩历史，所以这里使用独立入口。参考机器人、相机和 libfranka 代码来自 `reference/evo-rlt` 的只读副本；原 Evo-RLT 文件和环境不变。

## 候选 checkpoint

完整30k训练已经结束。`checkpoint-027000` 按10条验证轨迹的宏平均动作损失选出；`checkpoint-030000` 的联合损失和力矩损失略低，但动作损失略高。第一次真机测试优先使用 `027000`，便于沿用选择指标；两者都可恢复。

当前30k离线采样的测量力矩结果为：50步位置RMSE `14.55 mm`，前10步位置RMSE `8.61 mm`，姿态MAE `0.54°`，力矩RMSE `1.55 Nm`。此前真机观测模式首次推理约`3.3 s`（CUDA warmup），之后约`175–184 ms`；因此8Hz请求不保证每秒8次新预测，控制循环使用尚未过期的块内后续动作。历史观测结果保存在 `outputs/realtime_observe_027000.jsonl`。新实现尚未真机测试。

## 1. 只加载模型，不接硬件

```bash
cd /data0/wx/force-VLA
bash scripts/run_force_realtime.sh --check \
  --device cpu \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000
```

## 2. 真机观测，不发送动作

确认机器人没有被其他程序控制、急停可用、双臂工作区清空后执行。此模式需要访问 FCI、Robotiq 和两台 RealSense，但代码不会调用 `send_action`，不需要输入 `ARM`：

```bash
cd /data0/wx/force-VLA
FORCE_VLA_HARDWARE=1 bash scripts/run_force_realtime.sh \
  --mode observe --allow-hardware \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000 \
  --robot-config configs/deployment_force_vla.yaml \
  --device cuda --max-steps 30 --rate-hz 5 --inference-steps 10 \
  --log outputs/realtime_observe_027000_long.jsonl
```

`motion_sent` 必须始终为 `false`。任何相机、力矩或状态错误都会在发送动作前退出。

## 3. 单步动作测试

这一步会真正发送一个受限的绝对TCP动作，必须在交互终端输入精确的 `ARM`。默认只执行1步，速度和动力学因子已降到测试配置；仍需人工保持急停和接管准备：

进入动作循环后，按 `r` 会丢弃旧动作块和正在计算的旧推理结果，清空力矩历史并重新等待2秒历史；该操作不重连机器人、不调用 `robot.stop()`，等待期间不发送新的模型动作。按 `q`、`x` 或 `Esc` 会停止后续推理和动作发送，并执行 `robot.stop()` 清理；`Ctrl+C` 也可退出。键盘退出属于软件停止，不能替代机械臂急停按钮。

```bash
cd /data0/wx/force-VLA
FORCE_VLA_HARDWARE=1 bash scripts/run_force_realtime.sh \
  --mode execute --allow-hardware \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000 \
  --robot-config configs/deployment_force_vla.yaml \
  --device cuda --max-steps 1 --rate-hz 5 --inference-steps 10 \
  --log outputs/realtime_execute_027000_step1.jsonl
```

代码拒绝非有限动作、工作空间外动作、位置跳变超过3cm或姿态跳变超过0.2rad的输出，并将夹爪裁剪到`[0,1]`。原始模型目标和块间过渡后的目标均通过安全门。`--max-steps` 计数实际策略/遥操控制周期（目标30Hz），不计启动等待；单步窗口很短，不保证肉眼可见位移。日志中的 `command_debug` 可查看限幅后的速度和实际目标差值。

如果需要由操作者决定停止时机，可使用独立的 `configs/deployment_force_vla_motion_test.yaml` 和 `--hold-after-steps`。达到 `--max-steps` 后程序保持连接，不自动调用 `robot.stop()`；按 `q`、`x` 或 `Esc` 时才停止。异常和 `Ctrl+C` 仍会执行清理，避免控制线程遗留。

持续推理使用 `--continuous`，它不设置步数上限，直到操作者按 `q`、`x` 或 `Esc` 才停止：

```bash
cd /data0/wx/force-VLA
FORCE_VLA_HARDWARE=1 bash scripts/run_force_realtime.sh \
  --mode execute --allow-hardware --startup-open --continuous --async-inference \
  --teleop-assist \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000 \
  --robot-config configs/deployment_force_vla_speed_test.yaml \
  --device cuda --rate-hz 8 --inference-steps 10 \
  --log outputs/realtime_execute_continuous.jsonl
```

如果夹爪当前是闭合的，在 `--mode execute` 后增加 `--startup-open`。程序连接 Robotiq 后会先发送打开目标，并等待实测开度达到配置的`0.95`，再提示输入 `ARM`：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/run_force_realtime.sh \
  --mode execute --allow-hardware --startup-open --continuous --async-inference \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000 \
  --robot-config configs/deployment_force_vla_speed_test.yaml \
  --device cuda --rate-hz 5 --inference-steps 10 \
  --log outputs/realtime_execute_gripper_open.jsonl
```

当前夹爪动作采用滞回二值控制：输出大于`0.875`打开，小于等于`0.75`闭合，中间保持上一状态。日志中的 `measured_gripper_open_fraction` 是实际反馈开度；如果启动打开成功但后续模型输出长期低于`0.75`，夹爪再次闭合表示模型预测了闭合动作，而不是开度反馈失效。

`--teleop-assist` 使用本机 Xbox 手柄 `/dev/input/by-id/usb-Microsoft_Controller_3039373130303635393336353232-event-joystick`。左摇杆修正 XY，`Y/A` 修正 Z，右摇杆修正倾斜，`X/B` 修正滚转，左/右扳机请求打开/闭合。手柄有输入时暂停模型预测并丢弃旧动作块；在途预测也被作废。松开后重新收集2秒实测力矩历史，再接收新预测，不回到旧目标。输入设备由 `evdev` 直接读取，不启动 ROS Joy 节点。

需要提高连续测试速度时使用 `configs/deployment_force_vla_speed_test.yaml`。该配置把最大平移速度设为约`50mm/s`，单次目标限幅仍由动作安全门和`30mm`跳变上限控制；它不把模型动作强行放大到`50mm`。模型当前训练得到的是绝对TCP位姿，验证数据的相邻位置步长约`5–6mm`，若要求单次位移`50mm`，需要按更大时间跨度重新构建数据并训练。

如果出现高速下的顿挫、犹豫或电流声，先使用 `configs/deployment_force_vla_smooth_test.yaml`。它把运动段延长到`350ms`以覆盖连续推理周期，将平移速度降为`30mm/s`，并提高目标与速度平滑；单次目标限幅约为`10.5mm`。这份配置用于恢复轨迹连续性，原有配置不变。

该测试配置将单次平移限幅从约`1.5mm`提高到约`3mm`，运动窗口改为`150ms`，动力学因子为`0.05`。它只用于独立运动测试，原 `deployment_force_vla.yaml` 不变。

## 4. 短闭环测试

只有单步动作确认方向、幅度和夹爪行为安全后，才增加到3–5步；每次测试都使用独立日志和小的 `--max-steps`，不使用旧的 Evo-RLT rollout socket：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/run_force_realtime.sh \
  --mode execute --allow-hardware \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-027000 \
  --robot-config configs/deployment_force_vla.yaml \
  --device cuda --max-steps 3 --rate-hz 5 --inference-steps 10 \
  --log outputs/realtime_execute_027000_step3.jsonl
```

这不是任务成功率测试：它只验证实时观测、力矩历史、推理延迟、动作方向和安全限制。每次任务是否成功仍需人工标注；任何机械臂卡顿、相机延迟、力矩异常或动作方向错误都应立即停止并保留日志。
